"""Frozen-VLA verified selection on ladder rungs (branch-and-restore, study steps 1-2).

At every decision the LAP server proposes K chunks from the SAME observation, each with
its own request seed. Every candidate is executed for `execute_chunk_dim` steps from a
snapshot, scored on privileged state, and the selector commits one branch:

  sample   candidate 0 only -- ordinary sampling, the baseline (K is ignored)
  random   a uniformly random candidate among K -- K samples with no selection
  oracle   the candidate with the highest privileged progress score after its prefix
  gemini   Gemini Robotics-ER 2 picks from BEFORE + each candidate's rendered AFTER frame
           (abc_minimal.gemini_selector; --gemini-context adds privileged text)

Every candidate's privileged score is logged under every selector, so a gemini run
reports its regret (best privileged score minus the picked one's) and its agreement
with the privileged argmax at each decision.

Candidate k at (world, decision) always gets the same seed, whatever K is, so the K=1
oracle reproduces `sample` exactly and larger K strictly adds candidates.

`any_candidate_success` per decision (did ANY of the K branches reach task success
within its prefix) is the coverage signal: selection can only help where it is true.

Same CLI as abc_minimal.eval_policy plus --branch-k / --selector / --selector-seed /
--gemini-model / --gemini-context / --gemini-cameras;
needs --policy-backend lap, --task ladder and --vanilla-physics. Writes summary.json in
eval_policy's layout (success_rate, num_success, worlds[...]) plus per-decision logs.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from abc_minimal.branching import privileged_state, progress_score, restore, snapshot
from abc_minimal.config import SimEvalConfig
from abc_minimal.eval_policy import SimEnv, active_scene, build_scene_xml, jsonable, require_mjwarp, video_frame
from abc_minimal.ladder import get_task

SELECTORS = ("sample", "random", "oracle", "gemini", "beam")


@dataclass
class BranchSelectConfig(SimEvalConfig):
    branch_k: int = 1
    selector: str = "oracle"
    selector_seed: int = 0
    gemini_model: str = "gemini-robotics-er-2-preview"
    gemini_context: str = "outcomes"
    gemini_cameras: tuple[str, ...] = ("top",)
    # Record the executed (chosen-path) physics state every N sim steps to
    # states_<seed>.npz for exact offline rendering (scripts/render_branch_states.py).
    # 0 = off. --replay-log re-queries a fresh server, which is not bit-reproducible.
    record_states_every: int = 0
    # Save every decision's BEFORE frames and each candidate's AFTER frames (JPEG, all
    # camera_keys) plus the candidates' action chunks to frames_<seed>.npz, for training
    # and auditing image-only critics against the selector. Off by default.
    save_candidate_frames: bool = False
    # Replay: re-execute ONE logged world (worlds.jsonl / summary.json line with this
    # --seed) using only each decision's CHOSEN candidate seed, and record a video.
    # beam: bounded lookahead (study step 3). Expand each of beam_width states by branch_k
    # sampled chunks for beam_depth levels, keep the top beam_width by privileged score,
    # then COMMIT the best leaf's whole path (its states are real sim states, so nothing
    # is re-executed). Per committed chunk this costs ~beam_width*branch_k policy calls,
    # i.e. B=4,K=4 is matched-compute with the one-step oracle at K=16.
    beam_width: int = 4
    beam_depth: int = 3
    replay_log: str = ""
    replay_video_every: int = 2


def _jpg(img: np.ndarray) -> bytes:
    import io

    from PIL import Image

    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = img.transpose(1, 2, 0)  # renderer returns CHW
    if img.dtype != np.uint8:
        img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _save_frames(path: Path, cams: tuple[str, ...], rec: list) -> None:
    """rec: per decision (chunk, before_images, [after_images per k], [actions per k], scores, pick,
    prop_before, eef_before, [prop_after per k], [eef_after per k]). JPEG bytes go into one flat buffer;
    off[d, j, c] = (start, end), j = 0 BEFORE, j = 1 + k AFTER k. prop = joint state with the measured
    gripper opening (get_state_vanilla), eef = left/right grasp-site positions (forward kinematics)."""
    chunks, picks, scores, acts, parts, off, pos = [], [], [], [], [], [], 0
    prop_b, eef_b, prop_a, eef_a = [], [], [], []
    for chunk, before, afters, actions, sc, pick, pb, eb, pa, ea in rec:
        prop_b.append(pb); eef_b.append(eb); prop_a.append(np.stack(pa)); eef_a.append(np.stack(ea))
        o = np.zeros((1 + len(afters), len(cams), 2), np.int64)
        for j, imgs in enumerate([before, *afters]):
            for c, cam in enumerate(cams):
                b = _jpg(imgs[cam])
                o[j, c] = (pos, pos + len(b))
                parts.append(b)
                pos += len(b)
        chunks.append(chunk); picks.append(pick); scores.append(sc); acts.append(np.stack(actions)); off.append(o)
    np.savez(path, chunk=np.asarray(chunks, np.int32), pick=np.asarray(picks, np.int32),
             scores=np.asarray(scores, np.float32), actions=np.stack(acts).astype(np.float16),
             jpg_off=np.stack(off), jpg_buf=np.frombuffer(b"".join(parts), np.uint8),
             cams=np.asarray(cams), prop_before=np.stack(prop_b).astype(np.float32),
             eef_before=np.stack(eef_b).astype(np.float32), prop_after=np.stack(prop_a).astype(np.float32),
             eef_after=np.stack(eef_a).astype(np.float32))


def candidate_seed(world_seed: int, decision: int, k: int) -> int:
    return int(np.random.SeedSequence([world_seed, decision, k]).generate_state(1)[0] & 0x7FFFFFFF)


def _build_policy(config: BranchSelectConfig) -> Any:
    from abc_minimal.lap_client import LAPClientPolicy

    bridge = None
    if config.lap_bridge:
        from abc_minimal.lap_bridge import LAPSimBridge

        bridge = LAPSimBridge(
            mujoco.MjModel.from_xml_string(build_scene_xml(config)),
            gripper_ctrl_max=active_scene(config).gripper_ctrl_max,
            rotation_delta_in_base=config.lap_rotation_delta_in_base,
            align_eef_rot=config.lap_align_eef_rot,
            anchor_on_command=config.lap_anchor_on_command,
            binarize_gripper=config.lap_binarize_gripper,
            eef_link=config.lap_eef_link,
        )
    return LAPClientPolicy(config, bridge=bridge)


def _priv_summary(priv: dict[str, Any], ev: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "lift_m": priv["lift_m"],
        "reach_m": min(priv["reach_m"].values()),
        "grasped": any(priv["grasped"].values()),
        "touching": any(priv["touching"].values()),
        "arm_world_contacts": priv["arm_world_contacts"],
    }
    # Goal-specific ground truth (gemini --gemini-context outcomes+privileged prints all of it).
    if priv.get("hold_lift"):
        out["container_lift_m"] = priv["container_lift_m"]
        out["container_held"] = priv["container_lift_m"] >= priv["hold_lift"]
        out["container_arms"] = sorted(a for a, t in priv.get("container_touching", {}).items() if t)
        out["target_arms"] = sorted(a for a, g in priv["grasped"].items() if g)
    if priv.get("target_to_dirt_m") is not None:
        out["target_to_dirt_m"] = priv["target_to_dirt_m"]
    if priv.get("support_pos") is not None:
        out["target_to_support_xy_m"] = float(np.linalg.norm(priv["target_pos"][:2] - priv["support_pos"][:2]))
    for k in ("wipe_in_contact", "wipe_total_stroke_m", "wipe_cleaned_frac", "on_top_now", "inside_container"):
        if ev is not None and k in ev:
            out[k] = ev[k]
    return out


def _mean(xs: list) -> float | None:
    return float(np.mean(xs)) if xs else None


def _beam_world(config: "BranchSelectConfig", env: Any, policy: Any, seed: int, final_eval: dict[str, Any],
                lift_m: float) -> tuple[list, int, int, float, dict[str, Any]]:
    B, K, H, E = int(config.beam_width), int(config.branch_k), int(config.beam_depth), int(config.execute_chunk_dim)
    decisions: list = []
    steps = sim_steps = committed = 0
    infer_s = 0.0
    root = {"snap": snapshot(env, policy), "eval": final_eval, "path": []}
    while committed < config.num_chunks and not root["eval"]["ever_success"]:
        beam, stats = [root], []
        for depth in range(min(H, config.num_chunks - committed)):
            children = []
            for bi, node in enumerate(beam):
                restore(env, policy, node["snap"])
                obs = env.obs_vanilla_state()  # real rendered successor observation
                for k in range(K):
                    if k:
                        restore(env, policy, node["snap"])
                    policy.rng_seed = candidate_seed(seed, committed + depth, bi * K + k)
                    t_inf = time.perf_counter()
                    actions = policy.infer(obs)
                    infer_s += time.perf_counter() - t_inf
                    ev, n = node["eval"], 0
                    for action in actions[:E]:
                        env.step_one_vanilla(action)
                        ev = env.evaluate_vanilla()
                        n += 1
                        if ev["ever_success"]:
                            break
                    sim_steps += n
                    priv = privileged_state(env)
                    sc = progress_score(priv, ev, lift_m)
                    entry = {"k": bi * K + k, "parent": bi, "seed": policy.rng_seed, "score": sc, "steps": n,
                             "success": bool(ev["ever_success"]), "picked": bool(ev.get("picked", False)),
                             "priv": _priv_summary(priv, ev)}
                    children.append({"snap": snapshot(env, policy), "eval": ev, "score": sc, "path": node["path"] + [entry]})
            children.sort(key=lambda c: -c["score"])
            last = [c["path"][-1]["priv"] for c in children]
            stats.append({"depth": depth, "children": len(children), "best": children[0]["score"],
                          "plate_held": sum(bool(p.get("container_held")) for p in last),
                          "held_and_gripped": sum(bool(p.get("container_held") and p.get("grasped")) for p in last),
                          "bimanual": sum(bool(p.get("container_held") and p.get("target_arms") and
                                               set(p.get("container_arms") or []) - set(p.get("target_arms") or []))
                                          for p in last)})
            wins = [c for c in children if c["eval"]["ever_success"]]
            beam = wins[:1] if wins else children[:B]
            if wins:
                break
        best = beam[0]
        restore(env, policy, best["snap"])
        for i, entry in enumerate(best["path"]):
            decisions.append({"chunk": committed + i, "pick": 0, "privileged_argmax": 0, "regret": 0.0,
                              "any_candidate_success": entry["success"], "num_candidate_success": int(entry["success"]),
                              "any_candidate_grasp": bool(entry["priv"]["grasped"]), "gemini": None,
                              "beam": stats if i == 0 else None, "candidates": [entry]})
        steps += sum(e["steps"] for e in best["path"])
        committed += len(best["path"])
        root = {"snap": best["snap"], "eval": best["eval"], "path": []}
        if config.log_every_chunk:
            print(f"[beam] seed={seed} committed={committed:03d} best={best['score']:+.2f} "
                  f"plate_held={[s_['plate_held'] for s_ in stats]} held+gripped={[s_['held_and_gripped'] for s_ in stats]} "
                  f"bimanual={[s_['bimanual'] for s_ in stats]} success={root['eval']['ever_success']}", flush=True)
    return decisions, steps, sim_steps, infer_s, root["eval"]


def _physics_state(env: Any) -> np.ndarray:
    """Full mjSTATE_INTEGRATION vector (qpos, qvel, act, mocap incl. stain markers, ...)."""
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    x = np.empty(mujoco.mj_stateSize(env.model, spec), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, x, spec)
    return x


def _load_world(path: str, seed: int) -> dict[str, Any]:
    p = Path(path)
    records = (json.load(open(p))["worlds"] if p.suffix == ".json"
               else [json.loads(line) for line in open(p) if line.strip()])
    for w in records:
        if int(w["world_seed"]) == seed:
            return w
    raise ValueError(f"world_seed {seed} not in {path}")


def replay(config: BranchSelectConfig, env: Any, policy: Any) -> dict[str, Any]:
    """Replay of a logged branch-select world, checked decision-by-decision against the
    logged chosen candidate's privileged state. NOT reliably faithful: physics and the
    request seed are deterministic, but a fresh policy server process is not bit-identical
    to the one that ran the search (two fresh-server replays of one world already differ at
    chunk 0 and split by chunk 10-14), so check `first_divergent_chunk` before trusting it."""
    import imageio.v2 as imageio

    world = _load_world(config.replay_log, int(config.seed))
    out_dir = Path(config.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lift_m = float(active_scene(config).lift_m)
    video_path = out_dir / f"replay_{world['world_seed']}.mp4"
    video = imageio.get_writer(str(video_path), fps=config.video_fps // config.replay_video_every, macro_block_size=1)
    env.reset(seed=int(world["world_seed"]))
    policy.reset_episode()
    final_eval = env.evaluate_vanilla()
    video.append_data(video_frame(env.render_cameras_vanilla_state(), config.model.camera_keys))
    trace, steps = [], 0
    try:
        for d in world["decisions"]:
            chosen = d["candidates"][d["pick"]]
            obs = env.obs_vanilla_state()
            policy.rng_seed = int(chosen["seed"])
            actions = policy.infer(obs)
            ev = final_eval
            for action in actions[: int(chosen["steps"])]:
                env.step_one_vanilla(action)
                ev = env.evaluate_vanilla()
                steps += 1
                if steps % config.replay_video_every == 0:
                    video.append_data(video_frame(env.render_cameras_vanilla_state(), config.model.camera_keys))
            final_eval = ev
            priv = privileged_state(env)
            got = _priv_summary(priv, ev)
            want = chosen["priv"]
            drift = abs(progress_score(priv, ev, lift_m) - float(chosen["score"]))
            trace.append({"chunk": d["chunk"], "score_logged": chosen["score"], "score_replay": progress_score(priv, ev, lift_m),
                          "drift": drift, "plate_lift_m": got.get("container_lift_m"), "plate_arms": got.get("container_arms"),
                          "sponge_arms": got.get("target_arms"), "sponge_to_stains_m": got.get("target_to_dirt_m"),
                          "plate_lift_logged": want.get("container_lift_m")})
            if config.log_every_chunk:
                print(f"[replay] chunk={d['chunk']:03d} score logged={chosen['score']:+.2f} replay={trace[-1]['score_replay']:+.2f} "
                      f"plate={100 * (got.get('container_lift_m') or 0):.1f}cm arms={got.get('container_arms')} "
                      f"sponge_arms={got.get('target_arms')}", flush=True)
    finally:
        video.close()
        env.close()
    max_drift = max((t["drift"] for t in trace), default=0.0)
    first_div = next((t["chunk"] for t in trace if t["drift"] > 0.05), None)
    summary = {"format": "abc_minimal_branch_replay/v1", "world_seed": world["world_seed"], "video": str(video_path),
               "decisions": len(trace), "steps": steps, "success": bool(final_eval["ever_success"]),
               "max_score_drift": max_drift, "first_divergent_chunk": first_div, "trace": trace}
    (out_dir / f"replay_{world['world_seed']}.json").write_text(json.dumps(jsonable(summary), indent=2))
    print(f"[replay] wrote {video_path} | decisions={len(trace)} max_drift={max_drift:.3f} "
          f"first_divergent_chunk={first_div}", flush=True)
    return summary


def run(config: BranchSelectConfig) -> dict[str, Any]:
    if config.selector not in SELECTORS:
        raise ValueError(f"--selector must be one of {SELECTORS}, got {config.selector!r}")
    if config.policy_backend != "lap" or config.task != "ladder":
        raise ValueError("branch_select needs --policy-backend lap and --task ladder")
    if not config.vanilla_physics:
        raise ValueError("branch-and-restore needs --vanilla-physics (MJWarp state lives on the GPU)")
    if config.rtc or config.lap_head not in ("flow", "ar"):
        raise ValueError("branch_select supports the flow or AR head without RTC only")
    if config.lap_head == "ar" and config.selector == "beam":
        raise ValueError("beam is not wired for the AR head yet")
    require_mjwarp()
    gemini = None
    if config.selector == "gemini":
        from abc_minimal.gemini_selector import GeminiSelector

        gemini = GeminiSelector(model=config.gemini_model, context=config.gemini_context,
                                cameras=config.gemini_cameras)

    policy = _build_policy(config)
    if config.replay_log:
        gemini = None  # a replay never selects
    env = SimEnv(
        height=config.camera_height,
        width=config.camera_width,
        camera_keys=config.model.camera_keys,
        prompt=config.prompt,
        scene=active_scene(config),
        task=config.task,
        ladder_task=config.ladder_task,
        gpu_id=config.gpu_id,
    )
    if config.replay_log:
        return replay(config, env, policy)
    lift_m = float(active_scene(config).lift_m)
    n_cand = 1 if config.selector == "sample" else int(config.branch_k)
    if config.selector == "beam":
        print(f"[branch] beam B={config.beam_width} K={config.branch_k} H={config.beam_depth}", flush=True)
    record_every = int(config.record_states_every)
    if record_every and config.selector == "beam":
        raise ValueError("--record-states-every is not wired into the beam path yet")
    pick_rng = np.random.default_rng(config.selector_seed)
    out_dir = Path(config.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[branch] rung={config.ladder_task} selector={config.selector} K={n_cand} "
          f"prefix={config.execute_chunk_dim} chunks={config.num_chunks}", flush=True)

    worlds = []
    try:
        for world_index in range(config.num_worlds):
            t0 = time.perf_counter()
            seed = int(config.seed + world_index)
            env.reset(seed=seed)
            policy.reset_episode()
            final_eval = env.evaluate_vanilla()
            decisions, steps, sim_steps, infer_s = [], 0, 0, 0.0
            world_states: list[tuple[int, int, np.ndarray]] = []
            frame_rec: list = []
            save_frames = bool(config.save_candidate_frames)
            if record_every:
                world_states.append((0, -1, _physics_state(env)))
            if config.selector == "beam":
                decisions, steps, sim_steps, infer_s, final_eval = _beam_world(config, env, policy, seed, final_eval, lift_m)
            for chunk in range(0 if config.selector == "beam" else config.num_chunks):
                gemini_out = None
                obs = env.obs_vanilla_state()
                root = snapshot(env, policy)
                if save_frames:
                    gs0 = privileged_state(env)["grasp_site"]
                    eef_root = np.stack([gs0["left"], gs0["right"]])
                cands = []
                for k in range(n_cand):
                    if k:
                        restore(env, policy, root)
                    policy.rng_seed = candidate_seed(seed, chunk, k)
                    t_inf = time.perf_counter()
                    actions = policy.infer(obs)
                    infer_s += time.perf_counter() - t_inf
                    # AR head: this candidate's decoded langact IS its action (parsed client-side).
                    trace = getattr(policy, "reasoning_trace", None) or []
                    langact = trace[-1].get("reasoning", "") if trace else ""
                    ev, n, cand_states = final_eval, 0, []
                    for action in actions[: config.execute_chunk_dim]:
                        env.step_one_vanilla(action)
                        ev = env.evaluate_vanilla()
                        n += 1
                        if record_every and (steps + n) % record_every == 0:
                            cand_states.append((steps + n, _physics_state(env)))
                        if ev["ever_success"]:
                            break
                    sim_steps += n
                    priv = privileged_state(env)
                    cands.append({
                        "k": k,
                        "seed": policy.rng_seed,
                        "langact": langact,
                        "score": progress_score(priv, ev, lift_m),
                        "steps": n,
                        "success": bool(ev["ever_success"]),
                        "picked": bool(ev.get("picked", False)),
                        "priv": _priv_summary(priv, ev),
                        "_eval": ev,
                        "_priv": _priv_summary(priv, ev),
                        "_after": env.render_cameras_vanilla_state() if (gemini is not None or save_frames) else None,
                        "_actions": np.asarray(actions[: config.execute_chunk_dim], np.float32) if save_frames else None,
                        "_prop": env.get_state_vanilla() if save_frames else None,
                        "_eef": np.stack([priv["grasp_site"]["left"], priv["grasp_site"]["right"]]) if save_frames else None,
                        "_end": snapshot(env, policy) if n_cand > 1 else None,
                        "_states": cand_states,
                    })
                if config.selector == "oracle":
                    pick = int(np.argmax([c["score"] for c in cands]))  # ties -> lowest k
                elif config.selector == "gemini":
                    pick, gemini_out = gemini.choose(
                        env.prompt, obs["images"], [c["_after"] for c in cands], [c["_priv"] for c in cands])
                elif config.selector == "random":
                    pick = int(pick_rng.integers(n_cand))
                else:
                    pick = 0
                scores = [c["score"] for c in cands]
                if save_frames:
                    frame_rec.append((chunk, obs["images"], [c["_after"] for c in cands],
                                      [c["_actions"] for c in cands], scores, pick, np.asarray(obs["state"]),
                                      eef_root, [c["_prop"] for c in cands], [c["_eef"] for c in cands]))
                if pick != n_cand - 1:
                    restore(env, policy, cands[pick]["_end"])
                chosen = cands[pick]
                world_states.extend((t, chunk, x) for t, x in chosen["_states"])
                steps += chosen["steps"]
                final_eval = chosen["_eval"]
                decisions.append({
                    "chunk": chunk,
                    "pick": pick,
                    "any_candidate_success": any(c["success"] for c in cands),
                    "num_candidate_success": sum(c["success"] for c in cands),
                    "any_candidate_grasp": any(c["priv"]["grasped"] for c in cands),
                    "privileged_argmax": int(np.argmax(scores)),
                    "regret": float(max(scores) - scores[pick]),
                    "gemini": gemini_out if config.selector == "gemini" else None,
                    "candidates": [{k: v for k, v in c.items() if not k.startswith("_")} for c in cands],
                })
                if config.log_every_chunk:
                    scores = " ".join(f"{c['score']:+.2f}" for c in cands)
                    gem = ""
                    if gemini_out is not None:
                        gem = (f" gemini=FALLBACK({gemini_out.get('error', '')[:60]})" if gemini_out.get("fallback")
                               else f" gemini={gemini_out['latency_s']:.1f}s/a{gemini_out['attempts']}"
                                    f"/think{gemini_out.get('tokens_thinking')}")
                    print(f"world={world_index:03d} chunk={chunk:03d} pick={pick} "
                          f"success={final_eval['ever_success']} scores=[{scores}]{gem}", flush=True)
                if final_eval["ever_success"]:
                    break
            world = {
                "world_index": world_index,
                "world_seed": seed,
                "success": bool(final_eval["ever_success"]),
                "final_success": bool(final_eval["success"]),
                "reward": float(final_eval["reward"]),
                "steps": steps,
                "sim_steps_total": sim_steps,
                "infer_s_total": infer_s,
                "wall_s": time.perf_counter() - t0,
                "coverage_any_decision": any(d["any_candidate_success"] for d in decisions),
                "randomization": env.randomization,
                "final_task_eval": final_eval,
                "decisions": decisions,
                "reasoning_trace": list(getattr(policy, "reasoning_trace", []) or []),
            }
            if record_every:
                np.savez_compressed(out_dir / f"states_{seed}.npz",
                                    step=np.asarray([t for t, _, _ in world_states], dtype=np.int32),
                                    chunk=np.asarray([c for _, c, _ in world_states], dtype=np.int32),
                                    physics=np.stack([x for _, _, x in world_states]))
                world["states_file"] = str(out_dir / f"states_{seed}.npz")
            if save_frames and frame_rec:
                _save_frames(out_dir / f"frames_{seed}.npz", tuple(config.model.camera_keys), frame_rec)
                world["frames_file"] = str(out_dir / f"frames_{seed}.npz")
            worlds.append(world)
            # Append as we go: summary.json is only written after the last world, so a
            # time-limit kill would otherwise lose every finished world.
            with open(out_dir / "worlds.jsonl", "a") as fh:
                fh.write(json.dumps(jsonable(world)) + "\n")
            print(f"world={world_index:03d} done success={world['success']} "
                  f"picked={final_eval.get('picked')} max_lift={final_eval.get('max_lift_m', float('nan')):.3f} "
                  f"steps={steps} sim_steps={sim_steps} wall={world['wall_s']:.0f}s", flush=True)
    finally:
        env.close()

    success = np.asarray([w["success"] for w in worlds], dtype=bool)
    picked = np.asarray([bool(w["final_task_eval"].get("picked", False)) for w in worlds], dtype=bool)
    summary = {
        "format": "abc_minimal_branch_select/v1",
        "task": config.task,
        "ladder_task": config.ladder_task,
        "goal": get_task(config.ladder_task).goal,
        "selector": config.selector,
        "branch_k": n_cand,
        "checkpoint": f"lap://{config.lap_host}:{config.lap_port}",
        "config": asdict(config),
        "success_rate": float(success.mean()) if success.size else None,
        "num_success": int(success.sum()),
        "num_worlds": len(worlds),
        "pick_rate": float(picked.mean()) if picked.size else None,
        "num_picked": int(picked.sum()),
        "coverage_rate": float(np.mean([w["coverage_any_decision"] for w in worlds])) if worlds else None,
        "mean_sim_steps": float(np.mean([w["sim_steps_total"] for w in worlds])) if worlds else None,
        "mean_regret": _mean([d["regret"] for w in worlds for d in w["decisions"]]),
        "agree_with_privileged": _mean([d["pick"] == d["privileged_argmax"] for w in worlds for d in w["decisions"]]),
        "gemini_calls": gemini.calls if gemini else 0,
        "gemini_failures": gemini.failures if gemini else 0,
        "gemini_latency_s": gemini.latency_s if gemini else 0.0,
        "worlds": worlds,
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    print(f"[branch] success {summary['num_success']}/{summary['num_worlds']} "
          f"picked {summary['num_picked']} coverage={summary['coverage_rate']} -> {out_dir / 'summary.json'}",
          flush=True)
    return summary


if __name__ == "__main__":
    import tyro

    run(tyro.cli(BranchSelectConfig))
