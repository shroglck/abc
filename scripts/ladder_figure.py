"""Compose the paper figure of the ABC pick/place ladder from rendered frames.

Input: outputs/ladder_figure/<rung>/{top,left,right}.png + meta.json (ladder_figure_render.py)
       outputs/ladder_figure_evalres/<rung>/... (224x168, what the policy sees) for the header.
Output: ladder_figure.pdf / .png (vector text, 300 dpi raster).

    .venv/bin/python scripts/ladder_figure.py --rungs p09,... --out outputs/ladder_figure/ladder_figure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from PIL import Image

# Short, paper-facing names per rung (falls back to the rung id).
TITLES = {
    "p08_bottle_among_mixed": "Bottle among distractors",
    "p09_colored_lego": "Colour-named block",
    "p11_fruit_clutter": "Fruit in clutter",
    "p12_bottle_clutter": "Bottle in clutter",
    "p13_lego_clutter": "Block in clutter",
    "l01_bottle_in_bin_tight": "Bottle → bin",
    "l03_fruit_in_bin_spread": "Fruit → bin",
    "l05_fruit_on_plate": "Fruit → plate",
    "l07_lego_on_plate": "Block → plate",
    "l08_fruit_on_tray": "Fruit → tray",
    "l09_bottle_on_tray": "Bottle → tray",
    "l10_fruit_on_plate_clutter": "Fruit → plate, clutter",
    "l11_lego_on_plate_clutter": "Block → plate, clutter",
    "l12_bottle_in_bin_clutter": "Bottle → bin, clutter",
}
GOAL_LABEL = {"pick": "PICK", "place": "PICK & PLACE", "on": "PICK & PLACE", "stack": "STACK"}
GOAL_COLOR = {"pick": "#1f6feb", "place": "#c2410c", "on": "#c2410c", "stack": "#6d28d9"}
INK = "#111827"
MUTED = "#6b7280"
RULE = "#d1d5db"


def _font():
    for name in ("Helvetica", "Arial", "Inter", "DejaVu Sans"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


def crop_top(img: np.ndarray) -> np.ndarray:
    """Trim the featureless wall above the table and the base gantry below it."""
    h = img.shape[0]
    return img[int(0.21 * h): int(0.90 * h)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", required=True)
    ap.add_argument("--frames", default="outputs/ladder_figure")
    ap.add_argument("--evalres", default="outputs/ladder_figure_evalres/l10_fruit_on_plate_clutter")
    ap.add_argument("--out", default="outputs/ladder_figure/ladder_figure")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    plt.rcParams.update({"font.family": _font(), "pdf.fonttype": 42, "ps.fonttype": 42})
    rungs = args.rungs.split(",")
    cols, rows = args.cols, int(np.ceil(len(rungs) / args.cols))

    # Layout in inches: a header strip (the policy's three observations), then the grid.
    panel_w, panel_h = 1.72, 1.72 * (0.69 * 672) / 896
    gap_x, gap_y = 0.10, 0.40
    header_h = 1.15
    fig_w = cols * panel_w + (cols - 1) * gap_x
    fig_h = header_h + 0.10 + rows * (panel_h + gap_y) - 0.06
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    # ---- header: what the policy sees ----
    hy = fig_h - header_h
    fig.text(0, (hy + header_h - 0.02) / fig_h, "Policy observation (224×168, 3 cameras)",
             fontsize=6.6, fontweight="bold", color=INK, va="top")
    cam_names = [("top", "top"), ("left", "left wrist"), ("right", "right wrist")]
    cam_h = header_h - 0.36
    cam_w = cam_h * 224 / 168
    for i, (cam, label) in enumerate(cam_names):
        x = i * (cam_w + 0.06)
        ax = fig.add_axes([x / fig_w, (hy + 0.14) / fig_h, cam_w / fig_w, cam_h / fig_h])
        ax.imshow(Image.open(Path(args.evalres) / f"{cam}.png"), interpolation="nearest")
        ax.set_axis_off()
        fig.text((x + cam_w / 2) / fig_w, (hy + 0.10) / fig_h, label, fontsize=5.4, color=MUTED, ha="center", va="top")
    meta0 = json.loads((Path(args.frames) / rungs[0] / "meta.json").read_text())
    tx = 3 * (cam_w + 0.06) + 0.10
    fig.text(tx / fig_w, (hy + 0.14 + cam_h) / fig_h,
             "Bimanual YAM on the ABC tabletop scene (MuJoCo).\n"
             "Zero-shot: objects, receptacles and clutter never appear in training data;\n"
             "prompts are ABC-130k sub-task strings, e.g. “" + meta0["prompt"] + "”.\n"
             "Success: target lifted ≥ 5 cm (pick), or lifted and resting in/on the receptacle.",
             fontsize=5.3, color=INK, va="top", linespacing=1.45)
    # rule under the header
    fig.add_artist(plt.Line2D([0, 1], [(hy + 0.02) / fig_h] * 2, color=RULE, lw=0.6, transform=fig.transFigure))

    # ---- grid ----
    for k, rung in enumerate(rungs):
        r, c = divmod(k, cols)
        x0 = c * (panel_w + gap_x)
        y0 = fig_h - header_h - 0.10 - r * (panel_h + gap_y) - panel_h
        meta = json.loads((Path(args.frames) / rung / "meta.json").read_text())
        img = crop_top(np.asarray(Image.open(Path(args.frames) / rung / "top.png")))
        ax = fig.add_axes([x0 / fig_w, y0 / fig_h, panel_w / fig_w, panel_h / fig_h])
        ax.imshow(img, interpolation="lanczos")
        ax.set_axis_off()
        for s in ax.spines.values():
            s.set_visible(False)
        # rounded frame
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes, boxstyle="round,pad=0,rounding_size=0.02",
                                    fill=False, ec=RULE, lw=0.5, clip_on=False))
        # goal tag (top-left, inside the panel)
        goal = meta["goal"]
        ax.text(0.03, 0.95, GOAL_LABEL[goal], transform=ax.transAxes, fontsize=4.4, fontweight="bold", color="white",
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.5", fc=GOAL_COLOR[goal], ec="none"))
        # title + prompt under the panel
        title = TITLES.get(rung, rung)
        fig.text((x0 + 0.02) / fig_w, (y0 - 0.05) / fig_h, title, fontsize=6.2, fontweight="bold", color=INK, va="top")
        prompt = "“" + meta["prompt"] + "”"
        if len(prompt) > 40:  # wrap at the last space before the panel's width runs out
            cut = prompt.rfind(" ", 0, 40)
            prompt = prompt[:cut] + "\n" + prompt[cut + 1:]
        fig.text((x0 + 0.02) / fig_w, (y0 - 0.17) / fig_h, prompt, fontsize=4.9,
                 style="italic", color=MUTED, va="top", linespacing=1.25)

    out = Path(args.out)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04, dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))


if __name__ == "__main__":
    main()
