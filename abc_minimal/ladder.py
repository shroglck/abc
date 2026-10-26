"""Pick/place ladder: 14 single-target tasks on ABC's YAM scene, one declarative table.

Why a ladder: the only scored ABC sim task (put one bottle in the bin, zero-shot) sits
near 0-20% for every LAP checkpoint, which is too coarse to rank models or attribute
failures. The ladder spreads the same robot/table/cameras/lighting over 10 pick and
4 place rungs, each adding one difficulty axis on top of the previous:

  graspability   lego block -> bottle -> rolling fruit -> mug -> thin pen
  spread         the b1g cell (x 0.45-0.65, y +-0.10) -> most of the reachable table
  distractors    one object -> the target among other objects / same-shape colours
  precision      big bin -> small tray -> block-on-block stacking

Like `mug_flip`, this does NOT fork the XML: it loads `put_bottle.xml`, deletes the
bottle/bin bodies, and grafts the task's objects in, so every appearance decision of
the scored scene is shared. Bottles and the bin are CLONES of the scene's own bodies
(same meshes and contact parameters); legos/fruit/pens are MuJoCo primitives.

Prompts are ABC-130k `/subtask-annotation` strings verbatim (counts in the table
comments), except p09's colour word, which appears nowhere in ABC-130k: that rung is
out-of-distribution language on purpose.

`l01_bottle_in_bin_tight` draws its layout with the same RNG stream and acceptance
rule as `SimEnv.reset` on the b1g cell, so it reproduces the b1g put-bottles layouts
seed-for-seed (see scripts/ladder_smoke.py --b1g-check).
"""

from __future__ import annotations

import copy
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np

from abc_minimal.config import LadderSimConfig, MugFlipSimConfig, PutBottlesSimConfig
from abc_minimal.mug_flip import _add_mug_assets, _add_mug_body, _quat_mul, _quat_yaw

ObjKind = Literal["bottle", "mug", "box", "sphere", "capsule", "msobj", "cloth"]

# Cloth ("cloth" kind) is a MuJoCo flexcomp grid. Constraints that shaped it:
#  * MuJoCo caps flex contacts at 50 per flex (mjMAXCONPAIR), so a cloth with more than
#    50 vertices sinks through the table. 6x8 @ 6 cm = 48 vertices is the ceiling.
#  * MuJoCo-Warp has no flex support: cloth rungs need --vanilla-physics and the OpenGL
#    renderer (eval_policy switches both on when the model has a flex), so their frames
#    are NOT the ray-traced look of the rigid rungs.

# MolmoSpaces objaverse assets (copied into abc/assets/molmospaces/<id>/, see README.json
# there). Each is a self-contained MJCF: visual OBJ + texture + convex collider OBJs,
# y-up, origin at the mesh centre, mesh scale "-1 1 1". Grafted as freejoint bodies
# with the ladder's own contact parameters; the collider AABB gives footprint/height.
MS_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "molmospaces"
MS_YUP_TO_ZUP = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])  # +90 deg about x
# RoboDojo assets (assets/robodojo/, Apache-2.0, imported by scripts/import_robodojo_asset.py)
# are already z-up and live in their own directory, so they skip the y-up rotation.
RD_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "robodojo"
# Assets whose baked texture atlas has large unpainted (black) regions that MuJoCo's
# OBJ UV convention lands on -> speckled render. Use a flat colour for these instead.
MS_FLAT_RGBA: dict[str, tuple[float, float, float, float]] = {
    "7c265f6c42ef40b8b3e9c42560c28a97": (0.88, 0.88, 0.86, 1.0),  # plate, white ceramic
    "a37c32fbdbbb4ab19c8b1cccc557f52d": (0.55, 0.56, 0.58, 1.0),  # tray, brushed metal
}
# Assets whose visual OBJ is too sparse for the ray tracer (renders as a dotted outline):
# draw their convex collider hulls in the flat colour and drop the visual mesh.
MS_SHOW_COLLIDERS = {"a37c32fbdbbb4ab19c8b1cccc557f52d", "7c265f6c42ef40b8b3e9c42560c28a97",  # tray, plate
                    "rd_toolbox", "rd_basket_a", "rd_basket_b"}  # RoboDojo: decimated visuals render speckled
# RoboDojo imports carry no texture (the converter drops UVs when it decimates), so each
# gets a flat colour picked to match its description.json caption.
RD_RGBA = {
    "rd_basket_a": (0.22, 0.23, 0.25, 1.0),   # matte black storage bin
    "rd_basket_b": (0.55, 0.42, 0.28, 1.0),   # wicker-ish, so the two baskets differ
    "rd_plate": (0.90, 0.90, 0.88, 1.0),
    "rd_bowl": (0.85, 0.82, 0.75, 1.0),
    "rd_mug": (0.80, 0.80, 0.82, 1.0),
    "rd_can": (0.75, 0.25, 0.20, 1.0),
    "rd_toolbox": (0.20, 0.35, 0.55, 1.0),
}


# Task table.


@dataclass(frozen=True)
class ObjectSpec:
    name: str  # body name; its free joint is f"{name}_joint"
    kind: ObjKind
    label: str = ""  # words the prompt uses for it, e.g. "red lego"
    # box: half-extents xyz; sphere: (r,); capsule: (r, half_length); bottle/mug: ()
    size: tuple[float, ...] = ()
    rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    mass: float = 0.02
    scale_range: tuple[float, float] = (1.0, 1.0)
    asset: str = ""  # msobj: directory name under assets/molmospaces
    # cloth: grid (nx, ny), vertex spacing, vertex radius
    cloth_count: tuple[int, int] = (6, 8)
    cloth_spacing: float = 0.06
    cloth_radius: float = 0.006
    # box only: a visual-only layer over the top 30% in this colour (the sponge's scrub side).
    top_rgba: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class SpawnRegion:
    xy: tuple[float, float, float, float]  # x0 x1 y0 y1, bounds on the FOOTPRINT (b1g semantics)
    yaw: tuple[float, float] = (-math.pi, math.pi)
    min_sep: float = 0.04  # gap between footprint circles (= bottle_collision_margin)


@dataclass(frozen=True)
class ContainerSpec:
    kind: Literal["bin", "tray", "msobj", "spatula"]
    xy: tuple[float, float, float, float]  # centre range
    asset: str = ""  # msobj: directory name under assets/molmospaces
    yaw: tuple[float, float] = (-0.75, 0.75)
    scale_range: tuple[float, float] = (1.0, 1.0)  # bin mesh scale; ignored by the tray
    tray_inner: tuple[float, float] = (0.07, 0.07)  # cavity half-extents x, y
    tray_wall: float = 0.006
    tray_height: float = 0.035  # wall height above the floor
    rgba: tuple[float, float, float, float] = (0.35, 0.36, 0.40, 1.0)
    # spatula: a flat steel blade (half-extents) with a round handle along -x that rests on
    # the table too. Heavy enough (0.4 kg) that a sponge stroke does not shove it away.
    blade_half: tuple[float, float, float] = (0.055, 0.040, 0.003)
    handle_len: float = 0.16
    handle_r: float = 0.009


@dataclass(frozen=True)
class LadderTask:
    name: str
    # place: target inside the container's AABB (bin/tray cavity). on: target resting
    # on top of the container's upper surface (plate/tray) within its xy extent.
    # fold: the cloth target's xy footprint shrinks to <= fold_area_ratio of its initial
    # area while lying on the table, after having been lifted.
    # align: the target objects end up collinear and evenly spaced (RoboDojo "Align Blocks").
    goal: Literal["pick", "place", "stack", "on", "fold", "align", "wipe"]
    prompt: str  # may contain {color}
    objects: tuple[ObjectSpec, ...]
    # One name, or a tuple to draw one per world (p09).
    target: str | tuple[str, ...]
    region: SpawnRegion
    container: ContainerSpec | None = None  # goal == "place"
    # goal == "stack": the body the target must rest on. A tuple draws one per world
    # (excluding the target), which is what makes a stack instruction language-conditioned.
    support: str | tuple[str, ...] | None = None
    # Open/language rungs: prompt per drawn target, overriding `prompt` (which stays the
    # template recorded in summary.json so shards of one rung still compare equal).
    target_prompts: dict[str, str] = field(default_factory=dict)
    # Per-object region overrides (object name -> region).
    regions: dict[str, SpawnRegion] = field(default_factory=dict)
    # wipe: a grid of dirt spots the target (a pad) must be rubbed over. On the table when
    # wipe_patch (centre range x0 x1 y0 y1) is set, else on the container's top surface
    # (a disc of wipe_disc_frac x its half-width). Spots are wipe_spacing apart.
    wipe_patch: tuple[float, float, float, float] | None = None
    wipe_half: tuple[float, float] = (0.10, 0.06)
    wipe_spacing: float = 0.028
    wipe_disc_frac: float = 0.62
    wipe_dot_half: float | None = None  # stain half-width; None = DIRT_HALF (a continuous smear)
    # Total xy distance the pad must have been rubbed over the dirt (summed over all contact
    # strokes) for success. Needed wherever ONE pad footprint already covers >= wipe_frac of
    # the stains (then "place the pad on it and let it slide 2 cm" would pass). 0 = off, the
    # w01/w02 v1 rule, kept so their runs stay comparable.
    wipe_min_total_stroke: float = 0.0
    # > 0: rubbing counts only while the container is held at least this high above where it
    # spawned (one arm holds the plate up, the other wipes it). Strokes are then measured in
    # the CONTAINER frame, so carrying plate and pad together is not rubbing. 0 = off.
    wipe_hold_lift: float = 0.0
    # Bimanual (w04b): rubbing also needs one arm TOUCHING the container and the OTHER arm
    # GRIPPING the pad (>= 2 of its bodies in contact). Closes the w04 loophole: one arm
    # tilts the plate while an ungripped sponge rides on it and slides over the stains.
    wipe_require_bimanual: bool = False


TIGHT = SpawnRegion(xy=(0.45, 0.65, -0.10, 0.10))  # the b1g cell
SPREAD = SpawnRegion(xy=(0.35, 0.80, -0.45, 0.45))
MID = SpawnRegion(xy=(0.42, 0.72, -0.25, 0.25), min_sep=0.06)
STACK_CELL = SpawnRegion(xy=(0.45, 0.65, -0.15, 0.15), min_sep=0.06)
# Clutter rungs: 5-6 objects, so a wider cell than MID but kept clear of the far edge.
CLUTTER = SpawnRegion(xy=(0.40, 0.76, -0.36, 0.36), min_sep=0.05)
# Clutter next to a receptacle (bin/plate cells sit at x >= 0.64): objects on the near side.
CLUTTER_NEAR = SpawnRegion(xy=(0.38, 0.62, -0.36, 0.36), min_sep=0.05)

# b1g's bin cell and scale range, from eval_abc_bottles_ckpt.sbatch.
BIN_B1G = ContainerSpec("bin", xy=(0.70, 0.78, 0.15, 0.25), scale_range=(0.95, 1.05))
BIN_SPREAD = ContainerSpec("bin", xy=(0.55, 0.75, -0.30, 0.30), scale_range=(0.95, 1.05))
TRAY_TIGHT = ContainerSpec("tray", xy=(0.66, 0.74, 0.12, 0.22), yaw=(-0.4, 0.4))

# MolmoSpaces receptacles/objects (see assets/molmospaces/README.json).
MS_PLATE = "7c265f6c42ef40b8b3e9c42560c28a97"  # round white plate, 18 cm across, 2 cm tall
MS_TRAY = "a37c32fbdbbb4ab19c8b1cccc557f52d"  # metal tray with raised edges, 21 x 19 x 5 cm
MS_APPLE = "0098150095bb4b0f80a7dce8764ed392"  # apple, 8 cm
PLATE_TIGHT = ContainerSpec("msobj", xy=(0.64, 0.72, 0.10, 0.22), yaw=(-0.5, 0.5), asset=MS_PLATE)
TRAY_MS_TIGHT = ContainerSpec("msobj", xy=(0.64, 0.72, 0.10, 0.22), yaw=(-0.4, 0.4), asset=MS_TRAY)
# RoboDojo receptacles (assets/robodojo, Apache-2.0).
TOOLBOX = ContainerSpec("msobj", xy=(0.62, 0.70, 0.14, 0.26), yaw=(-0.3, 0.3), asset="rd_toolbox")
# Further out for the bottle: a 16 cm bottle plus the 21 cm tray do not both fit next to
# the TIGHT cell (sampler raised on 2/10 seeds with TRAY_MS_TIGHT).
TRAY_MS_FAR = ContainerSpec("msobj", xy=(0.70, 0.78, 0.16, 0.28), yaw=(-0.4, 0.4), asset=MS_TRAY)

RED = (0.85, 0.10, 0.10, 1.0)
BLUE = (0.10, 0.25, 0.85, 1.0)
YELLOW = (0.95, 0.80, 0.10, 1.0)


def lego(name: str, color: str, rgba: tuple[float, float, float, float]) -> ObjectSpec:
    # 3.2 x 3.2 x 2.4 cm: a duplo-sized block, well inside the ~8.2 cm open gripper.
    return ObjectSpec(name, "box", label=f"{color} lego", size=(0.016, 0.016, 0.012), rgba=rgba, mass=0.02)


BOTTLE = ObjectSpec("bottle", "bottle", label="plastic bottle", mass=0.05, scale_range=(0.9, 1.1))
# Scaled so the body (0.093 m at scale 1) fits the gripper; the rim/handle are graspable anyway.
MUG = ObjectSpec("mug", "mug", label="cup", mass=0.2, scale_range=(0.8, 0.9))
FRUIT = ObjectSpec("fruit", "sphere", label="fake fruit", size=(0.03,), rgba=(0.90, 0.35, 0.08, 1.0), mass=0.06)
# Capsule, not cylinder: MJWarp runs cylinder contacts through GJK, less stable for thin bodies.
PEN = ObjectSpec("pen", "capsule", label="pen", size=(0.0075, 0.06), rgba=(0.08, 0.10, 0.55, 1.0), mass=0.01)
APPLE = ObjectSpec("apple", "msobj", label="fake fruit", mass=0.15, scale_range=(0.9, 1.0), asset=MS_APPLE)
# Wipe pads. Both are graspable across their short side (6.4 / 5.6 cm < the ~8.2 cm open gripper).
TOWEL_PAD = ObjectSpec("towel_pad", "box", label="towel", size=(0.05, 0.032, 0.012), rgba=(0.52, 0.70, 0.88, 1.0), mass=0.05)
SPONGE = ObjectSpec("sponge", "box", label="sponge", size=(0.045, 0.028, 0.015), rgba=(0.97, 0.82, 0.22, 1.0),
                    top_rgba=(0.13, 0.50, 0.22, 1.0), mass=0.02)
# v2: w01's 1.2 cm pad was never lifted (0/120 episodes, two checkpoints); a thicker fold,
# still under the ~8.2 cm open gripper across its 6.4 cm side, as graspable as a lego is tall.
TOWEL_PAD_THICK = ObjectSpec("towel_pad", "box", label="towel", size=(0.05, 0.032, 0.0125), rgba=(0.52, 0.70, 0.88, 1.0), mass=0.06)
DIRT_RGBA = (0.26, 0.17, 0.07, 1.0)
DIRT_HALF = 0.0155  # square stain half-width: overlaps its 2.8 cm-spaced neighbours, and each
# stain gets a seeded random yaw, so the patch reads as one smear rather than a tiled grid.
# 30 x 42 cm towel, 48 vertices (see the cloth note at the top).
TOWEL = ObjectSpec("towel", "cloth", label="towel", rgba=(0.25, 0.45, 0.85, 1.0), mass=0.12)

_TASKS = (
    # ---------------- pick ----------------
    LadderTask("p01_lego_tight", "pick", "pick up the lego from the center of the table",  # 9711
               (lego("lego", "red", RED),), "lego", TIGHT),
    LadderTask("p02_bottle_tight", "pick", "pick up the plastic bottle at the center of the board",  # 2660
               (BOTTLE,), "bottle", TIGHT),
    LadderTask("p03_fruit_tight", "pick", "pick up the fake fruit from the center of the board",  # 190
               (FRUIT,), "fruit", TIGHT),
    LadderTask("p04_cup_tight", "pick", "pick up the cup at the center of the board",  # 858
               (MUG,), "mug", TIGHT),
    # Spread rungs keep the "center of" phrasing so only the spatial axis changes.
    LadderTask("p05_lego_spread", "pick", "pick up the lego from the center of the table",
               (lego("lego", "red", RED),), "lego", SPREAD),
    LadderTask("p06_bottle_spread", "pick", "pick up the plastic bottle at the center of the board",
               (BOTTLE,), "bottle", SPREAD),
    LadderTask("p07_lego_vs_bottles", "pick", "pick up the lego from the center of the table",
               (lego("lego", "red", RED), replace(BOTTLE, name="bottle_a"), replace(BOTTLE, name="bottle_b")),
               "lego", SPREAD),
    LadderTask("p08_bottle_among_mixed", "pick", "pick up the plastic bottle at the center of the board",
               (BOTTLE, lego("lego", "blue", BLUE), FRUIT, MUG), "bottle", SPREAD),
    # No colour words anywhere in ABC-130k: out-of-distribution language by design.
    LadderTask("p09_colored_lego", "pick", "pick up the {color} lego from the center of the table",
               (lego("lego_red", "red", RED), lego("lego_blue", "blue", BLUE), lego("lego_yellow", "yellow", YELLOW)),
               ("lego_red", "lego_blue", "lego_yellow"), MID),
    LadderTask("p10_pen_spread", "pick", "pick up the pen from the center of the board",  # 12309
               (PEN, lego("lego", "red", RED), FRUIT), "pen", SPREAD),
    # ---------------- place ----------------
    LadderTask("l01_bottle_in_bin_tight", "place", "place the plastic bottle in the trash bin",  # 2574
               (BOTTLE,), "bottle", TIGHT, container=BIN_B1G),
    LadderTask("l02_lego_in_tray_tight", "place", "place the lego in the container",  # 18111
               (lego("lego", "red", RED),), "lego", TIGHT, container=TRAY_TIGHT),
    LadderTask("l03_fruit_in_bin_spread", "place", "place the fake fruit in the bin",  # 2245
               (FRUIT,), "fruit", SPREAD, container=BIN_SPREAD),
    LadderTask("l04_lego_stack", "stack", "place the lego on the lego",  # 48
               (lego("lego_red", "red", RED), lego("lego_blue", "blue", BLUE)), "lego_red", STACK_CELL,
               support="lego_blue"),
    # MolmoSpaces receptacles: no walls to clear, so easier than the bin.
    LadderTask("l05_fruit_on_plate", "on", "place the fake fruit on the plate",  # 1007
               (APPLE,), "apple", TIGHT, container=PLATE_TIGHT),
    LadderTask("l06_cup_on_tray", "on", "place the cup on the tray",  # 463
               (MUG,), "mug", TIGHT, container=TRAY_MS_TIGHT),
    # 2026-09-16 additions. Prompts marked ~ are not verbatim ABC-130k strings (nearest:
    # "place the lego in the container", "place the cup on the tray").
    LadderTask("l07_lego_on_plate", "on", "place the lego on the plate",  # ~
               (lego("lego", "red", RED),), "lego", TIGHT, container=PLATE_TIGHT),
    LadderTask("l08_fruit_on_tray", "on", "place the fake fruit on the tray",  # ~
               (APPLE,), "apple", TIGHT, container=TRAY_MS_TIGHT),
    LadderTask("l09_bottle_on_tray", "on", "place the plastic bottle on the tray",  # ~
               (BOTTLE,), "bottle", TIGHT, container=TRAY_MS_FAR),
    # ---------------- compositional picks in clutter ----------------
    LadderTask("p11_fruit_clutter", "pick", "pick up the fake fruit from the center of the board",  # 190
               (APPLE, BOTTLE, lego("lego", "blue", BLUE), MUG, PEN), "apple", CLUTTER),
    LadderTask("p12_bottle_clutter", "pick", "pick up the plastic bottle at the center of the board",  # 2660
               (BOTTLE, APPLE, replace(APPLE, name="apple_b"), lego("lego_red", "red", RED),
                lego("lego_yellow", "yellow", YELLOW), PEN), "bottle", CLUTTER),
    LadderTask("p13_lego_clutter", "pick", "pick up the lego from the center of the table",  # 9711
               (lego("lego", "red", RED), APPLE, BOTTLE, MUG, PEN), "lego", CLUTTER),
    # ---------------- pick-and-place in clutter ----------------
    LadderTask("l10_fruit_on_plate_clutter", "on", "place the fake fruit on the plate",  # 1007
               (APPLE, BOTTLE, lego("lego", "blue", BLUE), PEN), "apple", CLUTTER_NEAR, container=PLATE_TIGHT),
    LadderTask("l11_lego_on_plate_clutter", "on", "place the lego on the plate",  # ~
               (lego("lego", "red", RED), APPLE, BOTTLE, PEN), "lego", CLUTTER_NEAR, container=PLATE_TIGHT),
    LadderTask("l12_bottle_in_bin_clutter", "place", "place the plastic bottle in the trash bin",  # 2574
               (BOTTLE, APPLE, lego("lego", "red", RED), PEN), "bottle", CLUTTER_NEAR, container=BIN_B1G),
    # ---------------- open / language-conditioned ----------------
    # The target (and for o02 the support) is drawn per world and named in the prompt, so the
    # scene alone does not say what to do -- the RoboDojo "Open" dimension. Prompts are the
    # per-object ABC-130k strings; {color} rungs are OOD language (ABC has no colour words).
    LadderTask("o01_pickup_by_language", "pick", "pick up the object",
               (BOTTLE, APPLE, lego("lego", "red", RED), PEN), ("bottle", "apple", "lego", "pen"), CLUTTER,
               target_prompts={
                   "bottle": "pick up the plastic bottle at the center of the board",  # 2660
                   "apple": "pick up the fake fruit from the center of the board",  # 190
                   "lego": "pick up the lego from the center of the table",  # 9711
                   "pen": "pick up the pen from the center of the board",  # 12309
               }),
    LadderTask("o03_tools_in_toolbox", "place", "place the screwdriver in the toolbox",  # 70
               (PEN, replace(PEN, name="pen_b", rgba=(0.6, 0.15, 0.1, 1.0)), lego("lego", "red", RED)),
               "pen", CLUTTER_NEAR, container=TOOLBOX),
    LadderTask("o04_align_blocks", "align", "place the legos in a row at the center of the table",  # ~
               (lego("lego_red", "red", RED), lego("lego_blue", "blue", BLUE), lego("lego_yellow", "yellow", YELLOW)),
               "lego_red", MID),
    LadderTask("o02_stack_by_language", "stack", "place the {color} lego on the {support_color} lego",  # 48
               (lego("lego_red", "red", RED), lego("lego_blue", "blue", BLUE), lego("lego_yellow", "yellow", YELLOW)),
               ("lego_red", "lego_blue", "lego_yellow"), MID,
               support=("lego_red", "lego_blue", "lego_yellow")),
    # ---------------- deformable ----------------
    LadderTask("f01_fold_towel", "fold", "fold the towel",  # 1502
               (TOWEL,), "towel", SpawnRegion(xy=(0.34, 0.84, -0.34, 0.34), yaw=(-0.4, 0.4))),  # footprint bounds; the towel is 30x42 cm
    # ---------------- wipe (tool use; EXPERIMENTAL until validated) ----------------
    # ABC-130k has exactly one wipe segment ("wipe the center of the board with the
    # towel", 1 segment) and no scrub/rub/wash at all, so the verb is zero-shot. w01 uses
    # that string verbatim with a folded towel; w02 is a sponge on a plate.
    LadderTask("w01_wipe_board_towel", "wipe", "wipe the center of the board with the towel",  # 1
               (TOWEL_PAD,), "towel_pad", SpawnRegion(xy=(0.38, 0.74, -0.36, 0.36), min_sep=0.04),
               wipe_patch=(0.55, 0.60, -0.05, 0.05), wipe_half=(0.10, 0.06)),
    LadderTask("w02_wipe_plate_sponge", "wipe", "wipe the plate with the sponge",  # 0
               (SPONGE,), "sponge", SpawnRegion(xy=(0.38, 0.74, -0.36, 0.36), min_sep=0.04),
               container=ContainerSpec("msobj", xy=(0.54, 0.62, -0.06, 0.06), yaw=(-0.5, 0.5), asset=MS_PLATE)),
    # v2 of w01/w02 (2026-09-18): v1 let "place the pad on the dirt, let it slide 2 cm" score
    # wherever one footprint covers >= wipe_frac of the stains (w02: 8/13). v2 requires 10 cm of
    # total rubbing; w01b also uses the thicker towel. v1 stays defined for its existing runs.
    LadderTask("w01b_wipe_board_towel", "wipe", "wipe the center of the board with the towel",  # 1
               (TOWEL_PAD_THICK,), "towel_pad", SpawnRegion(xy=(0.38, 0.74, -0.36, 0.36), min_sep=0.04),
               wipe_patch=(0.55, 0.60, -0.05, 0.05), wipe_half=(0.10, 0.06), wipe_min_total_stroke=0.10),
    LadderTask("w02b_wipe_plate_sponge", "wipe", "wipe the plate with the sponge",  # 0
               (SPONGE,), "sponge", SpawnRegion(xy=(0.38, 0.74, -0.36, 0.36), min_sep=0.04),
               container=ContainerSpec("msobj", xy=(0.54, 0.62, -0.06, 0.06), yaw=(-0.5, 0.5), asset=MS_PLATE),
               wipe_min_total_stroke=0.10),
    # Utensil: a spatula (1858 ABC segments as an object, never wiped). Its stains ride on
    # the blade, so wiping it on the table and holding it up with the other arm both work.
    LadderTask("w03_wipe_spatula_sponge", "wipe", "wipe the spatula with the sponge",  # 0
               (SPONGE,), "sponge", SpawnRegion(xy=(0.38, 0.74, -0.36, 0.36), min_sep=0.04),
               container=ContainerSpec("spatula", xy=(0.54, 0.62, -0.06, 0.06), yaw=(-math.pi, math.pi)),
               wipe_spacing=0.022, wipe_dot_half=0.009,  # separate blotches: bare steel must show between
               # The sponge covers all 12 blade stains at once, so coverage alone proves nothing:
               # require ~2 back-and-forth passes of rubbing.
               wipe_min_total_stroke=0.10),
    # Bimanual (2026-10-09): w02b plus "the plate must be held up while it is wiped".
    LadderTask("w04_hold_plate_wipe_sponge", "wipe", "hold the plate with one hand and wipe it with the sponge",  # 0
               (SPONGE,), "sponge", SpawnRegion(xy=(0.38, 0.74, -0.36, 0.36), min_sep=0.04),
               container=ContainerSpec("msobj", xy=(0.54, 0.62, -0.06, 0.06), yaw=(-0.5, 0.5), asset=MS_PLATE),
               wipe_min_total_stroke=0.10, wipe_hold_lift=0.03),
    LadderTask("w04b_hold_plate_wipe_sponge", "wipe", "hold the plate with one hand and wipe it with the sponge",  # 0
               (SPONGE,), "sponge", SpawnRegion(xy=(0.38, 0.74, -0.36, 0.36), min_sep=0.04),
               container=ContainerSpec("msobj", xy=(0.54, 0.62, -0.06, 0.06), yaw=(-0.5, 0.5), asset=MS_PLATE),
               wipe_min_total_stroke=0.10, wipe_hold_lift=0.03, wipe_require_bimanual=True),
    # Layout-OOD evals for w02b (2026-10-26, flow-RL study): identical to w02b except the plate
    # centre is drawn 6-14 cm beyond its training box (y in [-0.06, 0.06]) on one side, with x
    # widened to [0.48, 0.68]; sponge region and plate yaw unchanged. Left/right are mirrored so
    # neither arm is favoured. Eval only -- never train on these.
    LadderTask("w02b_ood_left", "wipe", "wipe the plate with the sponge",  # 0
               (SPONGE,), "sponge", SpawnRegion(xy=(0.38, 0.74, -0.36, 0.36), min_sep=0.04),
               container=ContainerSpec("msobj", xy=(0.48, 0.68, 0.12, 0.20), yaw=(-0.5, 0.5), asset=MS_PLATE),
               wipe_min_total_stroke=0.10),
    LadderTask("w02b_ood_right", "wipe", "wipe the plate with the sponge",  # 0
               (SPONGE,), "sponge", SpawnRegion(xy=(0.38, 0.74, -0.36, 0.36), min_sep=0.04),
               container=ContainerSpec("msobj", xy=(0.48, 0.68, -0.20, -0.12), yaw=(-0.5, 0.5), asset=MS_PLATE),
               wipe_min_total_stroke=0.10),
)

LADDER_TASKS: dict[str, LadderTask] = {t.name: t for t in _TASKS}

# Rungs dropped from the active ladder because they scored 0/10 for abc_yam_29k on the
# 4-light scene (put_bottle_4light.xml, sweep 3885413, 2026-09-15). Kept defined so
# old results stay reproducible (`--ladder-task` still accepts them); only removed
# from LADDER_TASK_NAMES, i.e. from what `TASK=all` runs. A 0/10 still allows up to
# ~28% (95% Wilson), and the cut was made on ONE checkpoint.
RETIRED: frozenset[str] = frozenset({
    "p01_lego_tight", "p03_fruit_tight", "p04_cup_tight", "p05_lego_spread",
    "p07_lego_vs_bottles", "p10_pen_spread", "l02_lego_in_tray_tight", "l04_lego_stack",
    "l06_cup_on_tray",  # 0/10 for all three ckpts; the cup is never picked (as p04)
    # 2026-09-16, user call: single-bottle picks are covered by p08/p12 and the l01/l09 places.
    "p02_bottle_tight", "p06_bottle_spread",
})
# Active rungs, easy -> hard; the sbatch array order.
# Defined and runnable by name (--ladder-task / TASK=<rung>) but not yet in TASK=all:
# appended at the end of the table so no existing array index moves when they join.
EXPERIMENTAL: frozenset[str] = frozenset({"w01_wipe_board_towel", "w02_wipe_plate_sponge", "w03_wipe_spatula_sponge",
                                           "w01b_wipe_board_towel", "w02b_wipe_plate_sponge",
                                           "w04_hold_plate_wipe_sponge", "w04b_hold_plate_wipe_sponge",
                                           "w02b_ood_left", "w02b_ood_right"})
LADDER_TASK_NAMES: tuple[str, ...] = tuple(n for n in LADDER_TASKS if n not in RETIRED and n not in EXPERIMENTAL)


def get_task(name: str) -> LadderTask:
    if name not in LADDER_TASKS:
        raise ValueError(f"unknown --ladder-task {name!r}; one of: {', '.join(LADDER_TASK_NAMES)}")
    task = LADDER_TASKS[name]
    names = [o.name for o in task.objects]
    assert len(set(names)) == len(names), f"{name}: duplicate object names"
    assert sum(o.kind == "mug" for o in task.objects) <= 1, f"{name}: at most one mug per scene"
    return task


# Geometry of each object kind, relative to its body origin.


_MS_BOUNDS: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def is_rd(asset: str) -> bool:
    return asset.startswith("rd_")


def ms_asset_dir(asset: str) -> Path:
    d = (RD_ASSETS if is_rd(asset) else MS_ASSETS) / asset
    if not d.is_dir():
        raise FileNotFoundError(f"MolmoSpaces asset not found: {d}")
    return d


def ms_bounds_zup(asset: str) -> tuple[np.ndarray, np.ndarray]:
    """Collider AABB (min, max) of an asset in OUR frame: mesh x-flip applied, y-up
    rotated to z-up (x, y, z) <- (-x_obj, z_obj... i.e. (x, -z_obj, y_obj))."""
    if asset not in _MS_BOUNDS:
        vs = []
        for f in sorted(f for f in ms_asset_dir(asset).glob("*_collider*.obj") if not f.name.endswith("_mx.obj")):
            with open(f) as fh:
                for line in fh:
                    if line.startswith("v "):
                        vs.append([float(x) for x in line.split()[1:4]])
        if is_rd(asset):
            v = np.asarray(vs, dtype=np.float64)  # RoboDojo: z-up, no mirror
        else:
            v = np.asarray(vs, dtype=np.float64) * np.array([-1.0, 1.0, 1.0])  # the MJCF's mirror
            # +90 deg about x: (x, y, z) -> (x, -z, y)
            v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
        _MS_BOUNDS[asset] = (v.min(axis=0), v.max(axis=0))
    return _MS_BOUNDS[asset]


def _scaled_size(spec: ObjectSpec, scale: float) -> np.ndarray:
    return np.asarray(spec.size, dtype=np.float64) * scale


def footprint(spec: ObjectSpec, scale: float, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """xy min/max offsets of the resting object's footprint from its body origin."""
    if spec.kind == "bottle":
        from abc_minimal.eval_policy import bottle_xy_footprint

        return bottle_xy_footprint(PutBottlesSimConfig(), 0, scale, yaw)
    if spec.kind == "mug":
        r = (MugFlipSimConfig().mug_handle_x + MugFlipSimConfig().mug_handle_radius) * scale
        return np.array([-r, -r]), np.array([r, r])
    if spec.kind == "msobj":
        lo, hi = ms_bounds_zup(spec.asset)
        r = float(np.max(np.abs(np.concatenate([lo[:2], hi[:2]])))) * scale  # yaw-safe disc
        return np.array([-r, -r]), np.array([r, r])
    if spec.kind == "cloth":
        nx, ny = spec.cloth_count
        hx, hy = 0.5 * (nx - 1) * spec.cloth_spacing + spec.cloth_radius, 0.5 * (ny - 1) * spec.cloth_spacing + spec.cloth_radius
        corners = np.array([[-hx, -hy], [-hx, hy], [hx, -hy], [hx, hy]], dtype=np.float64)
        c, s_ = math.cos(yaw), math.sin(yaw)
        offsets = corners @ np.array([[c, -s_], [s_, c]]).T
        return offsets.min(axis=0), offsets.max(axis=0)
    size = _scaled_size(spec, scale)
    if spec.kind == "sphere":
        return np.array([-size[0]] * 2), np.array([size[0]] * 2)
    if spec.kind == "box":
        hx, hy = size[0], size[1]
    else:  # capsule lying along its (rotated) x axis
        hx, hy = size[1] + size[0], size[0]
    corners = np.array([[-hx, -hy], [-hx, hy], [hx, -hy], [hx, hy]], dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    offsets = corners @ np.array([[c, -s], [s, c]]).T
    return offsets.min(axis=0), offsets.max(axis=0)


def half_height(spec: ObjectSpec, scale: float) -> float:
    """Height of the body origin above the table when the object rests on it."""
    if spec.kind == "bottle":
        return PutBottlesSimConfig().bottle_side_radii[0] * scale
    if spec.kind == "mug":
        return MugFlipSimConfig().mug_half_height * scale
    if spec.kind == "msobj":
        # Origin is the mesh centre; resting height is how far the collider extends below it.
        return float(-ms_bounds_zup(spec.asset)[0][2]) * scale
    if spec.kind == "cloth":
        return spec.cloth_radius
    size = _scaled_size(spec, scale)
    return float(size[2] if spec.kind == "box" else size[0])


def rest_quat(spec: ObjectSpec, yaw: float) -> np.ndarray:
    if spec.kind in ("bottle", "capsule"):
        # Lying flat: body z rotated onto the table plane, then yawed.
        from abc_minimal.eval_policy import _flat_bottle_quat

        return _flat_bottle_quat(yaw)
    if spec.kind == "msobj":
        if is_rd(spec.asset):
            return _quat_yaw(yaw)
        q = _quat_mul(_quat_yaw(yaw), MS_YUP_TO_ZUP)
        return q / np.linalg.norm(q)
    return _quat_yaw(yaw)


# Per-world draw (doubles as the world's `randomization` record).


@dataclass
class ObjectDraw:
    kind: str
    pos: list[float]
    quat: list[float]
    yaw: float
    scale: float


@dataclass
class ContainerDraw:
    kind: str
    pos: list[float]
    quat: list[float]
    yaw: float
    scale: float


@dataclass
class LadderDraw:
    seed: int
    task: str
    goal: str
    target: str
    prompt: str
    objects: dict[str, ObjectDraw]
    container: ContainerDraw | None
    bottle_scale: float  # bottle clones share ONE mesh scale (mesh scale is an asset property)
    support: str | None = None  # goal == "stack": the drawn support body
    wipe_center: list[float] | None = None  # goal == "wipe" on the table: patch centre xy


def _region(scene: LadderSimConfig, task: LadderTask, spec: ObjectSpec) -> SpawnRegion:
    region = task.regions.get(spec.name, task.region)
    if scene.region_xy is not None:
        region = replace(region, xy=tuple(scene.region_xy))
    return region


def _container_occupied(scene: LadderSimConfig, c: ContainerSpec, draw: ContainerDraw) -> float:
    if c.kind == "bin":
        return scene.bin_occupied_radius * draw.scale
    if c.kind == "msobj":
        lo, hi = ms_bounds_zup(c.asset)
        return float(np.max(np.abs(np.concatenate([lo[:2], hi[:2]])))) * draw.scale
    if c.kind == "spatula":
        return c.blade_half[0] + c.handle_len + c.handle_r  # disc about the blade centre
    outer = np.asarray(c.tray_inner) + c.tray_wall
    return float(np.linalg.norm(outer))


def sample_draw(rng: np.random.Generator, scene: LadderSimConfig, task: LadderTask, seed: int = 0) -> LadderDraw:
    """Draw one world. RNG order: object scales, container scale/yaw/x/y, object poses, target.

    That order is `SimEnv.reset`'s for put-bottles (bottle scales, bin scale, bin
    yaw/x/y, bottle poses), and the pose loop below consumes yaw, x, y per attempt
    with the same acceptance rule as `sample_bottle_pose` -- which is what makes
    l01 reproduce the b1g layouts.
    """
    # 1. Scales. Bottles share the first bottle's draw (one mesh asset for all clones).
    scales: dict[str, float] = {}
    bottle_scale = None
    for spec in task.objects:
        if spec.kind == "bottle" and bottle_scale is not None:
            scales[spec.name] = bottle_scale
            continue
        scales[spec.name] = float(rng.uniform(*spec.scale_range))
        if spec.kind == "bottle":
            bottle_scale = scales[spec.name]
    bottle_scale = 1.0 if bottle_scale is None else bottle_scale

    # 2. Container.
    occupied: list[tuple[np.ndarray, float]] = []
    container = None
    if task.container is not None:
        c = task.container
        c_scale = float(rng.uniform(*c.scale_range))
        yaw = float(rng.uniform(*c.yaw))
        x0, x1, y0, y1 = scene.container_xy if scene.container_xy is not None else c.xy
        x, y = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
        if c.kind == "bin":
            z = scene.bin_z_scale * c_scale
            quat = _quat_mul(_quat_yaw(yaw), np.asarray(scene.bin_base_quat, dtype=np.float64))
        elif c.kind == "msobj":
            z = scene.table_z - ms_bounds_zup(c.asset)[0][2] * c_scale + scene.spawn_clearance
            if is_rd(c.asset):
                quat = _quat_yaw(yaw)
            else:
                quat = _quat_mul(_quat_yaw(yaw), MS_YUP_TO_ZUP)
                quat = quat / np.linalg.norm(quat)
        elif c.kind == "spatula":
            z = scene.table_z + c.blade_half[2] + scene.spawn_clearance
            quat = _quat_yaw(yaw)
        else:
            z = scene.table_z + 0.001
            quat = _quat_yaw(yaw)
        container = ContainerDraw(c.kind, [x, y, float(z)], quat.tolist(), yaw, c_scale)
        occupied.append((np.array([x, y]), _container_occupied(scene, c, container)))

    # 2b. Wipe patch on the table (only wipe rungs draw here, so no other rung's RNG moves).
    wipe_center = None
    if task.goal == "wipe" and task.wipe_patch is not None:
        x0, x1, y0, y1 = task.wipe_patch
        wipe_center = [float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))]
        occupied.append((np.asarray(wipe_center), float(np.linalg.norm(task.wipe_half))))

    # 3. Object poses, rejection-sampled on footprints. Objects are placed in table
    # order, so a crowded cell can leave no room for the last one; then the whole
    # layout is redrawn (up to `layout_attempts`) rather than failing the episode.
    bases = np.asarray(scene.arm_base_xy, dtype=np.float64).reshape(2, 2)
    occupied_base = list(occupied)
    for layout_attempt in range(scene.layout_attempts):
        occupied = list(occupied_base)
        objects, failed = _place_objects(rng, scene, task, scales, occupied, bases)
        if failed is None:
            break
    else:
        raise RuntimeError(
            f"{task.name}: could not place {failed} after {scene.layout_attempts} layout "
            f"attempts x {scene.sample_attempts} draws (cell too tight for its objects)"
        )

    # 4. Target (drawn last so a fixed-target task's layouts do not depend on it).
    if isinstance(task.target, tuple):
        target = task.target[int(rng.integers(len(task.target)))]
    else:
        target = task.target
    support = task.support
    if isinstance(support, tuple):
        options = [n for n in support if n != target]
        support = options[int(rng.integers(len(options)))]
    prompt = fill_prompt(task.prompt, task, target, support)
    return LadderDraw(seed, task.name, task.goal, target, prompt, objects, container, bottle_scale, support, wipe_center)


def _place_objects(rng, scene, task, scales, occupied, bases):
    """One layout attempt: returns (objects, None) or (partial, name_of_object_that_failed)."""
    objects: dict[str, ObjectDraw] = {}
    for spec in task.objects:
        region, scale = _region(scene, task, spec), scales[spec.name]
        placed = None
        for _ in range(scene.sample_attempts):
            yaw = float(rng.uniform(*region.yaw))
            lo, hi = footprint(spec, scale, yaw)
            rx0, rx1, ry0, ry1 = region.xy
            x_low, x_high = rx0 - lo[0], rx1 - hi[0]
            y_low, y_high = ry0 - lo[1], ry1 - hi[1]
            if x_low > x_high or y_low > y_high:
                continue
            x, y = float(rng.uniform(x_low, x_high)), float(rng.uniform(y_low, y_high))
            center = np.array([x, y]) + 0.5 * (lo + hi)
            radius = float(0.5 * np.linalg.norm(hi - lo))
            if any(np.linalg.norm(center - c) <= radius + r + region.min_sep for c, r in occupied):
                continue
            if np.min(np.linalg.norm(bases - center, axis=1)) <= scene.arm_keepout_radius:
                continue
            placed = (x, y, yaw, center, radius)
            break
        if placed is None:
            return objects, spec.name
        x, y, yaw, center, radius = placed
        z = scene.table_z + half_height(spec, scale) + scene.spawn_clearance
        objects[spec.name] = ObjectDraw(spec.kind, [x, y, float(z)], rest_quat(spec, yaw).tolist(), yaw, scale)
        occupied.append((center, radius))
    return objects, None


def wipe_spots_xy(task: LadderTask, draw: LadderDraw) -> np.ndarray:
    """World xy of each dirt spot at spawn: a grid on the table patch, or a disc on the
    container's top surface (container frame yawed with the drawn container)."""
    s = task.wipe_spacing
    if task.wipe_patch is not None:
        hx, hy = task.wipe_half
        nx, ny = max(2, round(2 * hx / s)), max(2, round(2 * hy / s))
        xs = -hx + (np.arange(nx) + 0.5) * (2 * hx / nx)
        ys = -hy + (np.arange(ny) + 0.5) * (2 * hy / ny)
        return np.array([(x, y) for x in xs for y in ys]) + np.asarray(draw.wipe_center)
    c, d = task.container, draw.container
    if c is not None and d is not None and c.kind == "spatula":
        hx, hy = c.blade_half[0] - 0.006, c.blade_half[1] - 0.006  # keep stains off the edge
        nx, ny = max(2, round(2 * hx / s)), max(2, round(2 * hy / s))
        xs = -hx + (np.arange(nx) + 0.5) * (2 * hx / nx)
        ys = -hy + (np.arange(ny) + 0.5) * (2 * hy / ny)
        local = np.array([(x, y) for x in xs for y in ys])
        cy, sy = math.cos(d.yaw), math.sin(d.yaw)
        return local @ np.array([[cy, -sy], [sy, cy]]).T + np.asarray(d.pos[:2])
    if c is None or d is None or c.kind != "msobj":
        raise ValueError(f"{task.name}: a wipe rung needs wipe_patch, an msobj or a spatula container")
    lo, hi = ms_bounds_zup(c.asset)
    r = task.wipe_disc_frac * 0.5 * float(np.min(hi[:2] - lo[:2])) * d.scale
    n = int(np.ceil(r / s))
    local = np.array([(i * s, j * s) for i in range(-n, n + 1) for j in range(-n, n + 1)
                      if math.hypot(i * s, j * s) <= r])
    local = local + 0.5 * (lo[:2] + hi[:2]) * d.scale  # mesh centre offset
    cy, sy = math.cos(d.yaw), math.sin(d.yaw)
    return local @ np.array([[cy, -sy], [sy, cy]]).T + np.asarray(d.pos[:2])


def fill_prompt(template: str, task: LadderTask, target: str, support: str | None = None) -> str:
    """Per-target prompt (Open rungs), then `{color}` / `{support_color}` from the labels."""
    labels = {o.name: o.label for o in task.objects}
    text = task.target_prompts.get(target, template)
    if "{color}" in text:
        text = text.replace("{color}", labels[target].split()[0])
    if "{support_color}" in text and support:
        text = text.replace("{support_color}", labels[support].split()[0])
    return text


# Scene construction.


def _rename_tree(element: ET.Element, suffix: str) -> None:
    for node in element.iter():
        if node.get("name"):
            node.set("name", node.get("name") + suffix)


def _primitive_body(worldbody: ET.Element, spec: ObjectSpec, draw: ObjectDraw) -> None:
    body = ET.SubElement(
        worldbody, "body", {"name": spec.name, "pos": _fmt(draw.pos), "quat": _fmt(draw.quat)}
    )
    ET.SubElement(body, "freejoint", {"name": f"{spec.name}_joint"})
    size = _scaled_size(spec, draw.scale)
    # One geom that both renders (group 2, like the bottles' visual meshes) and
    # collides. Contact parameters are the bottles' collision set, so fingers grip
    # these the way they grip the scored object. `mass` lets MuJoCo derive the
    # inertia from the primitive's shape.
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{spec.name}_geom",
            "type": spec.kind,
            "size": _fmt(size),
            "rgba": _fmt(spec.rgba),
            "mass": f"{spec.mass:.8g}",
            "group": "2",
            "friction": "3.0 0.03 0.003",
            "condim": "6",
            "solref": "0.004 1",
            "solimp": "0.998 0.998 0.001",
            "priority": "1",
        },
    )
    if spec.top_rgba is not None and spec.kind == "box":
        hx, hy, hz = size
        ET.SubElement(body, "geom", {
            "name": f"{spec.name}_top", "type": "box",
            "size": _fmt([hx + 0.0004, hy + 0.0004, 0.3 * hz]), "pos": _fmt([0.0, 0.0, 0.7 * hz + 0.0004]),
            "rgba": _fmt(spec.top_rgba), "contype": "0", "conaffinity": "0", "group": "2", "density": "0",
        })


def _ms_body(
    root: ET.Element, worldbody: ET.Element, name: str, asset: str, pos, quat, scale: float, mass: float
) -> None:
    """Graft a MolmoSpaces asset as a freejoint body `name` (assets prefixed with it)."""
    d = ms_asset_dir(asset)
    src = ET.fromstring((d / f"{asset}.xml").read_text())
    asset_el = root.find("asset")
    assert asset_el is not None
    rename = {}
    for el in src.find("asset"):
        old = el.get("name"); new = f"{name}__{old}"; rename[old] = new
        el.set("name", new)
        if el.tag == "mesh" and is_rd(asset):
            el.set("file", str((d / el.get("file")).resolve()))
            el.set("scale", _fmt([scale] * 3))
        elif el.tag == "mesh":
            # The MJCF mirrors every mesh with scale "-1 1 1", which MuJoCo-Warp's ray
            # tracer renders as a speckled ghost (inverted winding). Use the pre-mirrored
            # `*_mx.obj` copies (x negated, faces reversed) at positive scale instead.
            el.set("file", str((d / el.get("file").replace(".obj", "_mx.obj")).resolve()))
            base = np.abs(np.asarray([float(v) for v in el.get("scale", "1 1 1").split()]))
            el.set("scale", _fmt(base * scale))
        elif el.get("file"):
            el.set("file", str((d / el.get("file")).resolve()))
        if el.tag == "material" and el.get("texture"):
            el.set("texture", rename[el.get("texture")])
        asset_el.append(el)
    body = ET.SubElement(worldbody, "body", {"name": name, "pos": _fmt(pos), "quat": _fmt(quat)})
    ET.SubElement(body, "freejoint", {"name": f"{name}_joint"})
    # Inertia from the collider hull at the given mass (the MJCF's own density is discarded).
    src_body = src.find("worldbody/body")
    ncol = sum(1 for g in src_body.findall("geom") if "collider" in (g.get("mesh") or ""))
    for g in src_body.findall("geom"):
        mesh = rename[g.get("mesh")]
        if "collider" in g.get("mesh"):
            show = asset in MS_SHOW_COLLIDERS
            flat = MS_FLAT_RGBA.get(asset) or RD_RGBA.get(asset) or (0.6, 0.6, 0.6, 1.0)
            ET.SubElement(body, "geom", {
                "name": f"{name}__{g.get('name').split('_', 1)[1]}", "type": "mesh", "mesh": mesh,
                "group": "2" if show else "3",
                "rgba": _fmt(flat) if show else "0 0 0 0",
                "mass": f"{mass / max(ncol, 1):.8g}",
                # Higher rolling friction than the bottles: scanned fruit has no flat
                # face and would roll a couple of cm on spawn before the policy acts.
                "friction": "3.0 0.03 0.05", "condim": "6", "solref": "0.004 1",
                "solimp": "0.998 0.998 0.001", "priority": "1",
            })
        elif asset in MS_SHOW_COLLIDERS:
            continue
        else:
            flat = MS_FLAT_RGBA.get(asset) or RD_RGBA.get(asset)
            look = {"rgba": _fmt(flat)} if flat else {"material": rename[g.get("material")]}
            ET.SubElement(body, "geom", {
                "name": f"{name}__visual", "type": "mesh", "mesh": mesh, **look,
                "contype": "0", "conaffinity": "0", "group": "2", "density": "0",
            })


def _cloth_flexcomp(root: ET.Element, worldbody: ET.Element, spec: ObjectSpec, draw: ObjectDraw) -> None:
    """A flexcomp grid towel named `spec.name` (vertex bodies `<name>_<i>`, 3 slide joints each)."""
    nx, ny = spec.cloth_count
    fc = ET.SubElement(worldbody, "flexcomp", {
        "name": spec.name, "type": "grid", "dim": "2", "count": f"{nx} {ny} 1",
        "spacing": _fmt([spec.cloth_spacing, spec.cloth_spacing, 0.02]), "radius": f"{spec.cloth_radius:.8g}",
        "mass": f"{spec.mass:.8g}", "pos": _fmt(draw.pos), "quat": _fmt(draw.quat), "rgba": _fmt(spec.rgba),
    })
    # Inextensible via edge-length EQUALITY constraints. The <elasticity> route was a
    # no-op here (Young 1e3 and 1e6 stretched identically: 6 cm edges -> 2 m under a 3 N
    # pull), so the towel behaved like rubber; with these constraints the same pull leaves
    # edges at 6.0-6.3 cm. (Bend-only elasticity below is allowed alongside the constraints.)
    ET.SubElement(fc, "edge", {"damping": "0.05", "equality": "true", "solref": "0.002 1", "solimp": "0.99 0.99 0.001"})
    # Bending stiffness (shell); elastic2d="bend" is REQUIRED (default = no elasticity at all,
    # which is why earlier Young values changed nothing). Bend-only combines with the edge
    # constraints. User's call (2026-09-16): Young 1e4 x 1 cm for a less flimsy towel, knowing it
    # costs fold success -- r101k: limp 10/16 & 8/16, 1e4 2/16, 3e4 1/16, 1e5 1/16 (the towel
    # slides as a sheet instead of draping into the gripper at higher stiffness).
    ET.SubElement(fc, "elasticity", {"young": "1e4", "poisson": "0.2", "thickness": "0.01", "damping": "0.5", "elastic2d": "bend"})
    # Bottle-class contact stiffness. Self-collision ON: without it a folded half falls through
    # the other half and the two layers end at the same height (the fold test needs a gap).
    ET.SubElement(fc, "contact", {"selfcollide": "auto", "internal": "false", "solref": "0.004 1", "solimp": "0.998 0.998 0.001"})
    size = root.find("size")
    if size is not None:
        size.set("nconmax", str(max(int(size.get("nconmax", "512")), 2000)))


def _spatula_body(worldbody: ET.Element, c: ContainerSpec, draw: ContainerDraw) -> None:
    hx, hy, hz = c.blade_half
    body = ET.SubElement(worldbody, "body", {"name": "container", "pos": _fmt(draw.pos), "quat": _fmt(draw.quat)})
    ET.SubElement(body, "freejoint", {"name": "container_joint"})
    contact = {"friction": "2.0 0.03 0.003", "condim": "6", "solref": "0.004 1",
               "solimp": "0.998 0.998 0.001", "priority": "1", "group": "2"}
    ET.SubElement(body, "geom", {"name": "spatula_blade", "type": "box", "size": _fmt([hx, hy, hz]),
                                 "rgba": "0.74 0.76 0.79 1", "mass": "0.30", **contact})
    # Handle: resting on the table (its axis sits handle_r above the table, the blade's hz).
    z = -hz + c.handle_r
    ET.SubElement(body, "geom", {"name": "spatula_handle", "type": "capsule",
                                 "fromto": _fmt([-hx - 0.004, 0.0, z, -hx - c.handle_len, 0.0, z]),
                                 "size": _fmt([c.handle_r]), "rgba": "0.07 0.07 0.08 1", "mass": "0.10", **contact})


def _tray_body(worldbody: ET.Element, c: ContainerSpec, draw: ContainerDraw) -> None:
    body = ET.SubElement(
        worldbody, "body", {"name": "container", "pos": _fmt(draw.pos), "quat": _fmt(draw.quat)}
    )
    ET.SubElement(body, "freejoint", {"name": "container_joint"})
    ix, iy = c.tray_inner
    w, h = c.tray_wall, c.tray_height
    # The bin's contact class values: a container, not a grasp target.
    common = {
        "type": "box",
        "rgba": _fmt(c.rgba),
        "group": "2",
        "condim": "3",
        "friction": "0.6 0.01 0.002",
        "solref": "0.004 1",
        "solimp": "0.998 0.998 0.001",
    }
    parts = {
        "floor": ((ix + w, iy + w, w / 2), (0.0, 0.0, w / 2), 0.3),
        "wall_px": ((w / 2, iy + w, h / 2), (ix + w / 2, 0.0, w + h / 2), 0.05),
        "wall_nx": ((w / 2, iy + w, h / 2), (-(ix + w / 2), 0.0, w + h / 2), 0.05),
        "wall_py": ((ix, w / 2, h / 2), (0.0, iy + w / 2, w + h / 2), 0.05),
        "wall_ny": ((ix, w / 2, h / 2), (0.0, -(iy + w / 2), w + h / 2), 0.05),
    }
    for part, (size, pos, mass) in parts.items():
        ET.SubElement(
            body,
            "geom",
            {"name": f"container_{part}", "size": _fmt(size), "pos": _fmt(pos), "mass": f"{mass:.8g}", **common},
        )


def _fmt(values) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def ladder_scene_xml(scene: LadderSimConfig, task: LadderTask, draw: LadderDraw) -> str:
    # The put-bottles SCENE_XML, so ABC_SCENE_XML lighting variants apply here too.
    from abc_minimal.eval_policy import ROOT, SCENE_XML, _brighten_surround

    root = ET.fromstring(SCENE_XML.read_text())
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((ROOT / "assets" / "put_bottles" / "assets").resolve()))
        compiler.set("texturedir", str((ROOT / "assets" / "put_bottles" / "assets").resolve()))
    worldbody = root.find("worldbody")
    assert worldbody is not None

    # Keep the scored scene's own bottle/bin bodies as templates, then strip them.
    # Their mesh/material assets and default classes stay: the clones reference them.
    bottle_tpl = worldbody.find("./body[@name='bottle_1']")
    bin_tpl = worldbody.find("./body[@name='bin_container']")
    for body in list(worldbody):
        name = body.get("name", "") if body.tag == "body" else ""
        if name == "bin_container" or re.fullmatch(r"bottle_\d+", name):
            worldbody.remove(body)

    bin_scale = draw.container.scale if draw.container is not None and draw.container.kind == "bin" else 1.0
    for mesh in root.findall("./asset/mesh"):
        name = mesh.get("name", "")
        base = np.asarray([float(v) for v in mesh.get("scale", "1 1 1").split()], dtype=np.float64)
        # The bottle_1 body uses the bottle_0 meshes (see `scene_xml` in eval_policy).
        if name.startswith("bottle_0_"):
            mesh.set("scale", _fmt(base * draw.bottle_scale))
        elif name.startswith("water_bottle_"):
            mesh.set("scale", _fmt(base * bin_scale))

    for spec in task.objects:
        d = draw.objects[spec.name]
        if spec.kind == "bottle":
            if bottle_tpl is None:
                raise ValueError("scene has no bottle_1 body to clone")
            body = copy.deepcopy(bottle_tpl)
            _rename_tree(body, f"__{spec.name}")
            body.set("name", spec.name)
            body.find("freejoint").set("name", f"{spec.name}_joint")
            body.set("pos", _fmt(d.pos))
            body.set("quat", _fmt(d.quat))
            worldbody.append(body)
        elif spec.kind == "msobj":
            _ms_body(root, worldbody, spec.name, spec.asset, d.pos, d.quat, d.scale, spec.mass)
        elif spec.kind == "cloth":
            _cloth_flexcomp(root, worldbody, spec, d)
        elif spec.kind == "mug":
            _add_mug_assets(root, np.array([d.scale]))
            _add_mug_body(worldbody, MugFlipSimConfig(), 0, d.scale)
            body = worldbody.find("./body[@name='mug_0']")
            body.set("name", spec.name)
            body.find("freejoint").set("name", f"{spec.name}_joint")
            body.set("pos", _fmt(d.pos))
            body.set("quat", _fmt(d.quat))
        else:
            _primitive_body(worldbody, spec, d)

    if task.container is not None:
        if task.container.kind == "bin":
            if bin_tpl is None:
                raise ValueError("scene has no bin_container body to clone")
            body = copy.deepcopy(bin_tpl)
            body.set("name", "container")
            body.find("freejoint").set("name", "container_joint")
            body.set("pos", _fmt(draw.container.pos))
            body.set("quat", _fmt(draw.container.quat))
            worldbody.append(body)
        else:
            if task.container.kind == "msobj":
                _ms_body(root, worldbody, "container", task.container.asset, draw.container.pos,
                         draw.container.quat, draw.container.scale, 0.5)
            elif task.container.kind == "spatula":
                _spatula_body(worldbody, task.container, draw.container)
            else:
                _tray_body(worldbody, task.container, draw.container)

    if task.goal == "wipe":
        # Visual-only mocap stains. The evaluator raycasts them onto the real surface at
        # reset (a plate is a dish, not a flat disc at its rim height) and moves each one
        # under the table once wiped, so the policy SEES the patch get cleaner.
        z = scene.table_z + 0.0006 if task.wipe_patch is not None else -5.0
        yaws = np.random.default_rng(draw.seed + 7919).uniform(-math.pi, math.pi, len(wipe_spots_xy(task, draw)))
        for i, (x, y) in enumerate(wipe_spots_xy(task, draw)):
            # A 0.1 mm height step per stain keeps overlapping coplanar boxes from z-fighting.
            b = ET.SubElement(worldbody, "body", {"name": f"dirt_{i}", "mocap": "true",
                                                  "pos": _fmt([x, y, z + 1e-4 * (i % 4)]), "quat": _fmt(_quat_yaw(yaws[i]))})
            ET.SubElement(b, "geom", {
                "name": f"dirt_{i}_geom", "type": "box", "size": _fmt([task.wipe_dot_half or DIRT_HALF, task.wipe_dot_half or DIRT_HALF, 0.0006]),
                "rgba": _fmt(DIRT_RGBA), "contype": "0", "conaffinity": "0", "group": "2",
            })

    if os.environ.get("ABC_SCENE_BRIGHT", "0") == "1":
        _brighten_surround(root)
    return ET.tostring(root, encoding="unicode")


# Metric.


def _joint_qpos_adr(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found: {name}")
    return int(model.jnt_qposadr[jid])


def _mesh_body_local_aabb(model: mujoco.MjModel, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    """AABB of a body's mesh geoms in its own frame (the bin: see PutBottlesEvaluator._bin_local_aabb)."""
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pts = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[adr : adr + num].reshape(-1, 3)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[geom])
        pts.append(verts @ rot.reshape(3, 3).T + model.geom_pos[geom])
    if not pts:
        raise ValueError(f"{body_name} has no mesh geoms to measure")
    allv = np.vstack(pts)
    return allv.min(axis=0), allv.max(axis=0)


class LadderEvaluator:
    """pick: target ever lifted >= lift_m. place: target inside the container AND ever
    lifted. stack: target resting on the support for stack_hold_evals evals AND ever
    lifted. Emits the key set the rollout loop, summary and shard merge read."""

    def __init__(self, model: mujoco.MjModel, scene: LadderSimConfig, task: LadderTask, draw: LadderDraw,
                 data: mujoco.MjData | None = None):
        self.model = model
        # wipe writes the dirt spots' mocap_pos into the LIVE MjData (the vanilla-physics
        # renderer copies it every frame), so it needs the env's data, not a qpos copy.
        self.data = data
        self.scene = scene
        self.task = task
        self.draw = draw
        self.names = [o.name for o in task.objects]
        self.specs = {o.name: o for o in task.objects}
        # Rigid objects: qpos address of the free joint. Cloth: per-vertex body pos + 3 slide
        # joints (world position = body_pos + qpos * axis), handled in cloth_vertices().
        self.addrs = np.asarray([-1 if self.specs[n].kind == "cloth" else _joint_qpos_adr(model, f"{n}_joint") for n in self.names], dtype=np.int32)
        self.cloth: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for n in self.names:
            if self.specs[n].kind == "cloth":
                bids = [b for b in range(model.nbody) if re.fullmatch(rf"{n}_\d+", model.body(b).name or "")]
                bids.sort(key=lambda b: int(model.body(b).name.rsplit("_", 1)[1]))
                adr = np.asarray([model.jnt_qposadr[model.body_jntadr[b]] for b in bids])
                axes = np.stack([model.jnt_axis[model.body_jntadr[b]: model.body_jntadr[b] + 3] for b in bids])  # (V,3,3)
                self.cloth[n] = (model.body_pos[bids].copy(), adr, axes)
        self.cloth_area0: float | None = None
        self.t = self.names.index(draw.target)
        self.container_adr = None
        if task.container is not None:
            self.container_adr = _joint_qpos_adr(model, "container_joint")
            c = task.container
            # Body-frame AABB. The bin and MolmoSpaces assets are y-up meshes carried by a
            # +90deg-x body quat, so in THEIR frame "up" is index 1; the primitive tray is z-up.
            self.c_up = 1 if c.kind in ("bin", "msobj") else 2
            self.c_plane = [0, 2] if self.c_up == 1 else [0, 1]
            if c.kind in ("bin", "msobj"):
                self.c_lo, self.c_hi = _mesh_body_local_aabb(model, "container")
            else:
                ix, iy = c.tray_inner
                # The cavity, floor to rim, in the tray body frame.
                self.c_lo = np.array([-ix, -iy, 0.0])
                self.c_hi = np.array([ix, iy, c.tray_wall + c.tray_height])
        self.s = None
        if task.goal == "on":
            specs = {o.name: o for o in task.objects}
            self.on_half_h = half_height(specs[draw.target], draw.objects[draw.target].scale)
        if task.goal == "wipe":
            n = len(wipe_spots_xy(task, draw))
            bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"dirt_{i}") for i in range(n)]
            self.dirt_mocap = np.asarray([model.body_mocapid[b] for b in bids], dtype=np.int32)
            self.pad_half = _scaled_size(self.specs[draw.target], draw.objects[draw.target].scale)
            self.container_bid = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "container")
                                  if task.container is not None else -1)
            # Spot anchors: world points (table) or container-frame points (plate), set by
            # place_dirt(); until then, the XML positions.
            self.dirt_ref = model.body_pos[bids].copy()
            self.dirt_in_container = task.wipe_patch is None
        if task.goal == "stack":
            support = draw.support or task.support
            self.s = self.names.index(support)
            specs = {o.name: o for o in task.objects}
            self.stack_dz = half_height(specs[draw.target], draw.objects[draw.target].scale) + half_height(
                specs[support], draw.objects[support].scale
            )
        self.reset()

    def reset(self) -> None:
        self.spawn: np.ndarray | None = None
        self.max_lift: np.ndarray | None = None
        self.ever_picked: np.ndarray | None = None
        self.hold = 0
        self.max_solved = 0
        self.ever_success = False
        self.cleaned: np.ndarray | None = None
        self.stroke_m = 0.0
        self.total_stroke_m = 0.0  # summed over all contact strokes; never reset in an episode
        self.last_pad_xy: np.ndarray | None = None
        self.container_z0: float | None = None

    def _bimanual_hold(self) -> bool:
        """One arm touches the container while the OTHER grips the pad (>= 2 of its bodies).
        Reads the live MjData contacts (evaluate runs right after mj_step)."""
        if self.data is None:
            return False
        m, d = self.model, self.data
        pad_root = int(m.body_rootid[m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{self.names[self.t]}_joint")]])
        c_root = int(m.body_rootid[m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "container_joint")]])
        pad: dict[str, set[int]] = {"left": set(), "right": set()}
        cont: dict[str, bool] = {"left": False, "right": False}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for a, b in ((b1, b2), (b2, b1)):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                arm = "left" if name.startswith("left_") else "right" if name.startswith("right_") else None
                if arm is None:
                    continue
                if int(m.body_rootid[a]) == pad_root:
                    pad[arm].add(b)
                elif int(m.body_rootid[a]) == c_root:
                    cont[arm] = True
        return any(len(pad[g]) >= 2 and cont[h] for g, h in (("left", "right"), ("right", "left")))

    def _container_pose(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = self.container_adr
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(qpos[a + 3 : a + 7], dtype=np.float64))
        return np.asarray(qpos[a : a + 3], dtype=np.float64), rot.reshape(3, 3)

    def place_dirt(self, data: mujoco.MjData) -> None:
        """Drop every dirt spot straight down onto its surface (table, or the container's
        own geoms) and record it as a world point or a container-frame point. Call after
        mj_forward at reset, before the first render."""
        m, xy = self.model, wipe_spots_xy(self.task, self.draw)
        yaw_quats = data.mocap_quat[self.dirt_mocap].copy()  # the XML's seeded per-stain yaw
        data.mocap_pos[self.dirt_mocap] = np.c_[xy, np.full(len(xy), -5.0)]  # out of the rays
        mujoco.mj_kinematics(m, data)
        gid = np.zeros(1, dtype=np.int32)

        def surface(x: float, y: float) -> np.ndarray | None:
            pnt = np.array([x, y, self.scene.table_z + 0.5])
            for _ in range(12):  # step past anything that is not the surface (a hovering gripper)
                dist = mujoco.mj_ray(m, data, pnt, np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
                if dist < 0:
                    return None
                pt, body = pnt - [0.0, 0.0, dist], int(m.geom_bodyid[gid[0]])
                ok = body == self.container_bid if self.dirt_in_container else int(m.body_weldid[body]) == 0
                if ok:
                    return pt
                pnt = pt - [0.0, 0.0, 1e-4]
            return None

        hits, quats, e = [], [], 0.006
        for i, (x, y) in enumerate(xy):
            hit = surface(x, y)
            hits.append(hit if hit is not None else np.array([x, y, self.scene.table_z]))
            # Tilt the stain onto the local surface (a plate is a dish): normal from 4 rays.
            px0, px1, py0, py1 = surface(x - e, y), surface(x + e, y), surface(x, y - e), surface(x, y + e)
            q = yaw_quats[i]
            if all(v is not None for v in (px0, px1, py0, py1)):
                n = np.cross(px1 - px0, py1 - py0)
                if n[2] > 1e-9:
                    n /= np.linalg.norm(n)
                    axis = np.cross([0.0, 0.0, 1.0], n)
                    ang = math.acos(float(np.clip(n[2], -1.0, 1.0)))
                    if np.linalg.norm(axis) > 1e-9 and ang > 1e-4:
                        tilt = np.zeros(4)
                        mujoco.mju_axisAngle2Quat(tilt, axis / np.linalg.norm(axis), ang)
                        q = _quat_mul(tilt, q)
            quats.append(q)
        world = np.asarray(hits) + [0.0, 0.0, 0.0006] + np.c_[np.zeros((len(xy), 2)), 1e-4 * (np.arange(len(xy)) % 4)]
        data.mocap_pos[self.dirt_mocap] = world
        data.mocap_quat[self.dirt_mocap] = np.asarray(quats)
        if self.dirt_in_container:
            c_pos, c_rot = self._container_pose(data.qpos)
            self.dirt_ref = (world - c_pos) @ c_rot  # container frame
        else:
            self.dirt_ref = world
        mujoco.mj_forward(m, data)

    def _dirt_world(self, qpos: np.ndarray) -> np.ndarray:
        if not self.dirt_in_container:
            return self.dirt_ref
        c_pos, c_rot = self._container_pose(qpos)
        return c_pos + self.dirt_ref @ c_rot.T

    def cloth_vertices(self, qpos: np.ndarray, name: str) -> np.ndarray:
        body_pos, adr, axes = self.cloth[name]
        disp = np.stack([qpos[a : a + 3] for a in adr])  # (V,3) slide displacements
        return body_pos + np.einsum("vj,vjk->vk", disp, axes)

    def evaluate(self, qpos: np.ndarray) -> dict[str, Any]:
        qpos = np.asarray(qpos, dtype=np.float64)
        verts = {n: self.cloth_vertices(qpos, n) for n in self.cloth}
        pos = np.stack([verts[n].mean(axis=0) if n in verts else qpos[a : a + 3] for n, a in zip(self.names, self.addrs)])
        if self.spawn is None:
            # First eval after a reset: the only point the objects are known at rest.
            self.spawn = pos.copy()
            if self.container_adr is not None:
                self.container_z0 = float(qpos[self.container_adr + 2])
            self.max_lift = np.zeros(len(self.names))
            self.ever_picked = np.zeros(len(self.names), dtype=bool)
        lift = pos[:, 2] - self.spawn[:, 2]
        # A cloth is "lifted" when any vertex rises lift_m (its centroid barely moves in a fold).
        for i, n in enumerate(self.names):
            if n in verts:
                lift[i] = float(verts[n][:, 2].max() - (self.spawn[i, 2]))
        self.max_lift = np.maximum(self.max_lift, lift)
        self.ever_picked |= self.max_lift >= self.scene.lift_m
        t = self.t
        extra: dict[str, Any] = {}

        if self.task.goal == "pick":
            solved = bool(self.ever_picked[t])
        elif self.task.goal == "place":
            c_pos = qpos[self.container_adr : self.container_adr + 3]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, qpos[self.container_adr + 3 : self.container_adr + 7])
            local = (pos - c_pos[None]) @ rot.reshape(3, 3)
            inside = np.all((local >= self.c_lo[None]) & (local <= self.c_hi[None]), axis=1)
            solved = bool(inside[t] and self.ever_picked[t])
            extra = {
                "inside_container": [bool(v) for v in inside.tolist()],
                "target_container_local_pos": [float(v) for v in local[t].tolist()],
                "container_local_lo": [float(v) for v in self.c_lo.tolist()],
                "container_local_hi": [float(v) for v in self.c_hi.tolist()],
                "distractor_in_container": bool(np.delete(inside, t).any()) if len(inside) > 1 else False,
            }
        elif self.task.goal == "align":
            # Collinear and evenly spaced: fit a line through the objects' xy centres by PCA,
            # require every off-line deviation <= align_line_tol, the gaps to differ by
            # <= align_gap_tol, and each object to sit on the table.
            xy = pos[:, :2]
            c = xy.mean(axis=0)
            u, sv, vt = np.linalg.svd(xy - c)
            direction = vt[0]
            along = (xy - c) @ direction
            off = np.abs((xy - c) @ vt[1])
            order = np.argsort(along)
            gaps = np.diff(along[order])
            on_table = bool(np.all(pos[:, 2] - self.scene.table_z < self.scene.fold_max_height))
            aligned_now = bool(off.max() <= self.scene.align_line_tol
                               and len(gaps) > 0 and (gaps.max() - gaps.min()) <= self.scene.align_gap_tol
                               and gaps.min() > 0.02 and on_table)
            self.hold = self.hold + 1 if aligned_now else 0
            solved = bool(self.hold >= self.scene.stack_hold_evals and self.ever_picked.any())
            extra = {"align_max_off_line_m": float(off.max()), "align_gap_spread_m": float(gaps.max() - gaps.min()) if len(gaps) else 0.0,
                     "align_gaps_m": [float(g) for g in gaps.tolist()], "aligned_now": aligned_now, "align_hold_steps": int(self.hold)}
        elif self.task.goal == "on":
            # Resting ON the container: xy inside its extent, centre height within tolerance
            # of "sitting on the top surface", held for stack_hold_evals evals. The AABB is
            # in the container's own frame; for a y-up asset rotated to z-up, local z is up.
            c_pos = qpos[self.container_adr : self.container_adr + 3]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, qpos[self.container_adr + 3 : self.container_adr + 7])
            local = (pos - c_pos[None]) @ rot.reshape(3, 3)
            lt = local[t]
            pl, up = self.c_plane, self.c_up
            in_xy = bool(np.all(lt[pl] >= self.c_lo[pl]) and np.all(lt[pl] <= self.c_hi[pl]))
            # Centre between the container's floor and "resting on its rim": a plate has
            # rim == floor, a tray with raised edges holds the object below the rim.
            # The upper bound excludes an object held above the container.
            floor, rim = float(self.c_lo[up]), float(self.c_hi[up])
            on_now = in_xy and floor <= lt[up] <= (rim + 2 * self.on_half_h + self.scene.on_z_tol)
            self.hold = self.hold + 1 if on_now else 0
            solved = bool(self.hold >= self.scene.stack_hold_evals and self.ever_picked[t])
            extra = {
                "on_now": on_now,
                "on_hold_steps": int(self.hold),
                "target_container_local_pos": [float(v) for v in lt.tolist()],
                "container_local_lo": [float(v) for v in self.c_lo.tolist()],
                "container_local_hi": [float(v) for v in self.c_hi.tolist()],
            }
        elif self.task.goal == "wipe":
            spots = self._dirt_world(qpos)
            if self.cleaned is None:
                self.cleaned = np.zeros(len(spots), dtype=bool)
            a = self.addrs[t]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, np.asarray(qpos[a + 3 : a + 7], dtype=np.float64))
            rot = rot.reshape(3, 3)
            p = pos[t]
            local = (spots - p) @ rot  # spot in the pad frame
            hx, hy, hz = self.pad_half
            mg = self.scene.wipe_margin
            # Inside the pad's 3D box (+margins), not an infinite slab: without the local-z bound a
            # sponge held ON ITS EDGE "covered" every stain along its (now horizontal) normal, e.g.
            # a plate 40 cm away (robotonly 26k w02, seed 20260514, 85% "wiped" with no contact).
            inside = ((np.abs(local[:, 0]) <= hx + mg) & (np.abs(local[:, 1]) <= hy + mg)
                      & (np.abs(local[:, 2]) <= hz + self.scene.wipe_z_tol))
            bottom_z = p[2] - float(np.abs(rot[2]) @ self.pad_half)  # lowest point of the box
            gap = bottom_z - spots[:, 2]
            under = inside & (gap <= self.scene.wipe_z_tol)
            hold_lift = float(self.task.wipe_hold_lift)
            c_lift = None
            pad_xy = p[:2]
            if hold_lift > 0:
                c_lift = float(qpos[self.container_adr + 2]) - float(self.container_z0)
                if c_lift < hold_lift:
                    under = np.zeros_like(under)  # plate not held up: rubbing does not count
                c_pos, c_rot = self._container_pose(qpos)
                pad_xy = ((p - c_pos) @ c_rot)[:2]  # relative to the plate, not the world
            bimanual_ok = None
            if self.task.wipe_require_bimanual:
                bimanual_ok = self._bimanual_hold()
                if not bimanual_ok:
                    under = np.zeros_like(under)
            in_contact = bool(under.any())
            # Stroke = xy path the pad has slid while continuously in contact with the dirt.
            if in_contact:
                if self.last_pad_xy is not None:
                    step_m = float(np.linalg.norm(pad_xy - self.last_pad_xy))
                    self.stroke_m += step_m
                    if self.ever_picked[t]:
                        self.total_stroke_m += step_m
                self.last_pad_xy = np.array(pad_xy, copy=True)
            else:
                self.stroke_m, self.last_pad_xy = 0.0, None
            if in_contact and self.ever_picked[t] and self.stroke_m >= self.scene.wipe_min_stroke:
                self.cleaned |= under
            frac = float(self.cleaned.mean()) if len(self.cleaned) else 0.0
            solved = bool(self.ever_picked[t] and frac >= self.scene.wipe_frac
                          and self.total_stroke_m >= self.task.wipe_min_total_stroke)
            if self.data is not None:
                shown = spots.copy()
                shown[self.cleaned, 2] = -5.0  # under the table: wiped
                self.data.mocap_pos[self.dirt_mocap] = shown
            extra = {
                "wipe_cleaned_frac": frac,
                "wipe_n_cleaned": int(self.cleaned.sum()),
                "wipe_n_dirt": int(len(self.cleaned)),
                "wipe_in_contact": in_contact,
                "wipe_stroke_m": float(self.stroke_m),
                "wipe_total_stroke_m": float(self.total_stroke_m),
                "wipe_pad_gap_m": float(gap[inside].min()) if inside.any() else None,
                "wipe_container_lift_m": c_lift,
                "wipe_container_held": None if c_lift is None else bool(c_lift >= hold_lift),
                "wipe_bimanual_ok": bimanual_ok,
            }
        elif self.task.goal == "fold":
            # FOLDED = lifted at some point + LAYERED now: a vertex is layered when another
            # vertex sits (nearly) on top of or under it -- within fold_layer_xy in the plane
            # and at least fold_layer_dz apart vertically. A flat towel has no such pairs; a
            # half-fold layers ~all vertices; a crumple layers some but rarely fold_layer_frac.
            # `lying` (vertices near the table) stops a folded towel held in the air scoring.
            v = verts[self.names[t]]
            dxy = np.linalg.norm(v[:, None, :2] - v[None, :, :2], axis=-1)
            dz = np.abs(v[:, None, 2] - v[None, :, 2])
            pair = (dxy <= self.scene.fold_layer_xy) & (dz >= self.scene.fold_layer_dz)
            np.fill_diagonal(pair, False)
            layered_frac = float(pair.any(axis=1).mean())
            ext = v[:, :2].max(axis=0) - v[:, :2].min(axis=0)
            area = float(ext[0] * ext[1])
            if self.cloth_area0 is None:
                self.cloth_area0 = area
            ratio = area / max(self.cloth_area0, 1e-9)
            lying = bool((v[:, 2] - self.scene.table_z < self.scene.fold_max_height).mean() >= 0.95)
            folded_now = bool(layered_frac >= self.scene.fold_layer_frac and lying)
            self.hold = self.hold + 1 if folded_now else 0
            solved = bool(self.hold >= self.scene.stack_hold_evals and self.ever_picked[t])
            extra = {"cloth_layered_frac": layered_frac, "fold_area_ratio": ratio, "cloth_lying": lying, "folded_now": folded_now,
                     "fold_hold_steps": int(self.hold), "cloth_extent_xy": [float(x) for x in ext.tolist()],
                     "cloth_max_z_above_table": float(v[:, 2].max() - self.scene.table_z)}
        else:  # stack
            d = pos[t] - pos[self.s]
            on_top = bool(
                np.linalg.norm(d[:2]) <= self.scene.stack_xy_tol
                and abs(d[2] - self.stack_dz) <= self.scene.stack_z_tol
            )
            self.hold = self.hold + 1 if on_top else 0
            solved = bool(self.hold >= self.scene.stack_hold_evals and self.ever_picked[t])
            extra = {
                "on_top_now": on_top,
                "support": self.names[self.s],
                "stack_hold_steps": int(self.hold),
                "target_minus_support": [float(v) for v in d.tolist()],
            }

        self.max_solved = max(self.max_solved, int(solved))
        self.ever_success = self.ever_success or solved
        others = np.delete(self.ever_picked, t)
        return {
            "reward": float(solved),
            "success": solved,
            "ever_success": bool(self.ever_success),
            "num_solved": int(solved),
            "num_active": 1,
            "max_solved_so_far": int(self.max_solved),
            "goal": self.task.goal,
            "target": self.names[t],
            "object_names": list(self.names),
            # PICKED/LIFTED: the TARGET got off the table. PLACED: the place/stack
            # predicate held; always False on pick rungs (there is nothing to place).
            "picked": bool(self.ever_picked[t]),
            "num_picked": int(self.ever_picked[t]),
            "lifted": bool(self.ever_picked[t]),
            "placed": bool(solved) if self.task.goal != "pick" else False,
            "num_placed": int(solved) if self.task.goal != "pick" else 0,
            "max_lift_m": float(self.max_lift[t]),
            "max_lift_per_object_m": [float(v) for v in self.max_lift.tolist()],
            "ever_picked_mask": [bool(v) for v in self.ever_picked.tolist()],
            "distractor_lifted": bool(others.any()) if others.size else False,
            "target_displaced_xy_m": float(np.linalg.norm(pos[t, :2] - self.spawn[t, :2])),
            "target_dropped": bool(self.ever_picked[t] and lift[t] < 0.01),
            **extra,
        }
