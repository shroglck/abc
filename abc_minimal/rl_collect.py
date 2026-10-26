"""Rollout collection for flow-head SFT / RL on one ladder rung (2026-10-25).

Runs the LAP flow head on a ladder rung and dumps, for every executed decision, exactly
what a trainer needs: the request the server saw (images by camera role, 20-D state,
prompt) and the chunk the flow head sampled in normalized model space (`actions_model`,
returned by a server started with LAP_RETURN_MODEL_ACTIONS=1 from rl_trees/r101k_flowrl).

  --selector sample   plain on-policy sampling (RL arms); K is ignored
  --selector oracle   K candidates per decision from one snapshot, commit the highest
                      privileged progress score (SFT arm's data, as branch_select)

--rollouts-per-world R repeats each world R times with distinct sampling seeds
(SeedSequence([world, rollout_id, decision, k]), rollout_id = --rollout-id-base + r), so
RL groups share a start state but not their noise.

Every decision also records privileged progress features before and after it, and a
subgoal label for the wipe rungs (`wipe_subgoal`): milestone m = cumulative wipe progress
(0 none, 1 stroke >= half the required total, 2 >= 30% of stains cleaned) and phase c =
current interaction (0 away, 1 touching the sponge, 2 grasped, 3 lifted, 4 over the
plate, 5 sponge in contact with the plate). Raw features are saved too, so labels can be
redefined offline without recollecting.

Output: <dump_dir>/ep_<world>_<rollout>.npz (compressed) + one line per episode in
<dump_dir>/index.jsonl; dump_dir defaults to --output-dir, so the parallel envs of one
launcher each write their own index (no shared append over NFS). Needs --policy-backend lap, --task ladder, --vanilla-physics,
--lap-head flow.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from abc_minimal.branch_select import BranchSelectConfig, _build_policy, _priv_summary
from abc_minimal.branching import privileged_state, progress_score, restore, snapshot
from abc_minimal.eval_policy import SimEnv, active_scene, jsonable, require_mjwarp
from abc_minimal.ladder import get_task

FEATURES = ("touching", "grasped", "lift_m", "xy_to_dirt_m", "dist_to_dirt_m", "in_contact", "stroke_m",
            "cleaned_frac", "success")


@dataclass
class RLCollectConfig(BranchSelectConfig):
    dump_dir: str = ""
    rollouts_per_world: int = 1
    rollout_id_base: int = 0
    save_candidates: bool = True


def rollout_seed(world_seed: int, rollout_id: int, decision: int, k: int) -> int:
    return int(np.random.SeedSequence([world_seed, rollout_id, decision, k]).generate_state(1)[0] & 0x7FFFFFFF)


def features(priv: dict[str, Any], ev: dict[str, Any]) -> np.ndarray:
    dirt = priv.get("dirt_center")
    xy = float(np.linalg.norm(priv["target_pos"][:2] - dirt[:2])) if dirt is not None else np.nan
    d3 = float(priv["target_to_dirt_m"]) if priv.get("target_to_dirt_m") is not None else np.nan
    return np.asarray([
        float(any(priv["touching"].values())), float(any(priv["grasped"].values())), float(priv["lift_m"]),
        xy, d3, float(bool(ev.get("wipe_in_contact", False))), float(ev.get("wipe_total_stroke_m", 0.0)),
        float(ev.get("wipe_cleaned_frac", 0.0)), float(bool(ev.get("ever_success", False))),
    ], np.float32)


def wipe_subgoal(f: np.ndarray, lift_m: float, min_stroke: float) -> tuple[int, int]:
    v = dict(zip(FEATURES, f))
    m = 2 if v["cleaned_frac"] >= 0.3 else 1 if v["stroke_m"] >= 0.5 * min_stroke else 0
    if v["in_contact"]:
        c = 5
    elif v["grasped"] and v["xy_to_dirt_m"] < 0.10:
        c = 4
    elif v["grasped"] and v["lift_m"] >= 0.5 * lift_m:
        c = 3
    elif v["grasped"]:
        c = 2
    elif v["touching"]:
        c = 1
    else:
        c = 0
    return m, c


class _Capture:
    """Wraps the websocket client so the last request/response pair is kept."""

    def __init__(self, client: Any):
        self._client = client
        self.request: dict[str, Any] | None = None
        self.response: dict[str, Any] | None = None

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        self.request, self.response = request, self._client.infer(request)
        if "actions_model" not in self.response:
            raise RuntimeError("server response has no `actions_model`: start it from rl_trees/r101k_flowrl "
                               "with LAP_RETURN_MODEL_ACTIONS=1")
        return self.response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def run(config: RLCollectConfig) -> dict[str, Any]:
    if config.selector not in ("sample", "oracle"):
        raise ValueError("rl_collect supports --selector sample or oracle")
    if config.policy_backend != "lap" or config.task != "ladder" or not config.vanilla_physics:
        raise ValueError("rl_collect needs --policy-backend lap, --task ladder and --vanilla-physics")
    if config.lap_head != "flow" or config.rtc:
        raise ValueError("rl_collect supports the flow head without RTC only")
    require_mjwarp()
    policy = _build_policy(config)
    cap = _Capture(policy.client)
    policy.client = cap
    env = SimEnv(height=config.camera_height, width=config.camera_width, camera_keys=config.model.camera_keys,
                 prompt=config.prompt, scene=active_scene(config), task=config.task,
                 ladder_task=config.ladder_task, gpu_id=config.gpu_id)
    lift_m = float(active_scene(config).lift_m)
    min_stroke = float(getattr(get_task(config.ladder_task), "wipe_min_total_stroke", 0.0) or 0.10)
    n_cand = 1 if config.selector == "sample" else int(config.branch_k)
    out = Path(config.dump_dir or config.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    if (out / "index.jsonl").exists():
        done = {(r["world_seed"], r["rollout_id"]) for r in map(json.loads, open(out / "index.jsonl")) if r}
    print(f"[rl_collect] rung={config.ladder_task} selector={config.selector} K={n_cand} "
          f"R={config.rollouts_per_world} worlds={config.num_worlds} from seed {config.seed} -> {out}", flush=True)
    n_success = n_eps = 0
    try:
        for world_index in range(config.num_worlds):
            seed = int(config.seed + world_index)
            for r in range(config.rollouts_per_world):
                rid = int(config.rollout_id_base + r)
                if (seed, rid) in done:
                    continue
                t0 = time.perf_counter()
                env.reset(seed=seed)
                policy.reset_episode()
                ev = env.evaluate_vanilla()
                f_now = features(privileged_state(env), ev)
                rec: dict[str, list] = {k: [] for k in ("state", "actions_model", "steps", "feat_before", "feat_after",
                                                        "cand_actions", "cand_scores", "pick")}
                imgs: dict[str, list] = {}
                req0 = None
                steps = 0
                for chunk in range(config.num_chunks):
                    obs = env.obs_vanilla_state()
                    root = snapshot(env, policy) if n_cand > 1 else None
                    cands = []
                    for k in range(n_cand):
                        if k:
                            restore(env, policy, root)
                        policy.rng_seed = rollout_seed(seed, rid, chunk, k)
                        actions = policy.infer(obs)
                        request, a_model = cap.request, np.asarray(cap.response["actions_model"], np.float32)
                        e, n = ev, 0
                        for action in actions[: config.execute_chunk_dim]:
                            env.step_one_vanilla(action)
                            e = env.evaluate_vanilla()
                            n += 1
                            if e["ever_success"]:
                                break
                        priv = privileged_state(env)
                        cands.append({"request": request, "a_model": a_model, "n": n, "eval": e,
                                      "score": progress_score(priv, e, lift_m), "feat": features(priv, e),
                                      "end": snapshot(env, policy) if n_cand > 1 else None})
                    pick = int(np.argmax([c["score"] for c in cands])) if n_cand > 1 else 0
                    if n_cand > 1 and pick != n_cand - 1:
                        restore(env, policy, cands[pick]["end"])
                    c = cands[pick]
                    req = c["request"]
                    req0 = req0 or req
                    for role, img in req["observation"].items():
                        if role != "state":
                            imgs.setdefault(role, []).append(np.asarray(img, np.uint8))
                    rec["state"].append(np.asarray(req["observation"]["state"], np.float32))
                    rec["actions_model"].append(c["a_model"])
                    rec["steps"].append(c["n"])
                    rec["feat_before"].append(f_now)
                    rec["feat_after"].append(c["feat"])
                    rec["pick"].append(pick)
                    if n_cand > 1 and config.save_candidates:
                        rec["cand_actions"].append(np.stack([x["a_model"] for x in cands]).astype(np.float16))
                        rec["cand_scores"].append(np.asarray([x["score"] for x in cands], np.float32))
                    f_now, ev = c["feat"], c["eval"]
                    steps += c["n"]
                    if config.log_every_chunk:
                        m, ph = wipe_subgoal(f_now, lift_m, min_stroke)
                        print(f"seed={seed} r={rid} chunk={chunk:03d} pick={pick} subgoal=({m},{ph}) "
                              f"success={ev['ever_success']}", flush=True)
                    if ev["ever_success"]:
                        break
                fb, fa = np.stack(rec["feat_before"]), np.stack(rec["feat_after"])
                sub_b = np.asarray([wipe_subgoal(x, lift_m, min_stroke) for x in fb], np.int8)
                sub_a = np.asarray([wipe_subgoal(x, lift_m, min_stroke) for x in fa], np.int8)
                arrays = {"state": np.stack(rec["state"]), "actions_model": np.stack(rec["actions_model"]),
                          "steps": np.asarray(rec["steps"], np.int16), "feat_before": fb, "feat_after": fa,
                          "subgoal_before": sub_b, "subgoal_after": sub_a, "pick": np.asarray(rec["pick"], np.int16),
                          **{f"img_{role}": np.stack(v) for role, v in imgs.items()}}
                if rec["cand_actions"]:
                    arrays["cand_actions"] = np.stack(rec["cand_actions"])
                    arrays["cand_scores"] = np.stack(rec["cand_scores"])
                path = out / f"ep_{seed}_{rid}.npz"
                np.savez_compressed(path, **arrays)
                row = {"world_seed": seed, "rollout_id": rid, "file": path.name, "success": bool(ev["ever_success"]),
                       "decisions": len(rec["steps"]), "steps": steps, "selector": config.selector, "branch_k": n_cand,
                       "prompt": req0["prompt"], "frame_description": req0["frame_description"],
                       "is_bimanual": bool(req0["is_bimanual"]), "feature_names": list(FEATURES),
                       "final_eval": _priv_summary(privileged_state(env), ev) | {"success": bool(ev["ever_success"])},
                       "wall_s": time.perf_counter() - t0}
                with open(out / "index.jsonl", "a") as fh:
                    fh.write(json.dumps(jsonable(row)) + "\n")
                n_eps += 1
                n_success += int(row["success"])
                print(f"[rl_collect] seed={seed} r={rid} success={row['success']} decisions={row['decisions']} "
                      f"steps={steps} wall={row['wall_s']:.0f}s | running {n_success}/{n_eps}", flush=True)
    finally:
        env.close()
    return {"episodes": n_eps, "successes": n_success}


if __name__ == "__main__":
    import tyro

    run(tyro.cli(RLCollectConfig))
