"""Nested rollouts: how much of the outcome variance is the choice of chunk?

At probe states h along a path, sample K chunks a_k ~ pi_ref(.|h) (the frozen VLA), and
from the state after each chunk run M independent pi_ref continuations of up to H chunks:

    G[k, m] = 1[success within the prefix + H chunks]   (also final/max progress_score)

One-way ANOVA per probe (k = group, m = replicate):
    sigma_w^2 = mean_k Var_m G[k, :]                      within-chunk: future-policy noise
                                                          (sim physics is deterministic, so this
                                                          is all from later action sampling)
    s^2       = Var_k mean_m G[k, :] - sigma_w^2 / M      between-chunk: action signal at h
s^2 is Var_a Q^{pi_ref}(h, a), the variance an oracle selector can exploit at this step;
sigma_w^2 is what a single Monte Carlo label for chunk k is buried under.
Also logged per chunk: progress_score right after it, to check the heuristic label
against the Monte Carlo Q.

The path between probes follows the privileged oracle (argmax progress_score over
--branch-k candidates), so probes reach states near success; the world stops after the
last probe. Probes every --probe-every chunks until the path first touches/grasps, then
right after contact onset and every --dense-every chunks (s^2 lives after contact).
With --save-frames each probe also writes frames/seed{S}_c{CCC}.jpg (all cameras at the
probe state) and .mp4 (the last --clip-seconds of the path, every --clip-every steps),
for annotators (Gemini) to judge against s^2. Same CLI as branch_select plus --anova-k/
--anova-m/--anova-horizon/--probe-every/--dense-every/--max-probes. Writes probes.jsonl.
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
from abc_minimal.eval_policy import SimEnv, active_scene, jsonable, require_mjwarp, video_frame


@dataclass
class AnovaConfig(BranchSelectConfig):
    anova_k: int = 16
    anova_m: int = 4
    anova_horizon: int = 30
    probe_every: int = 10
    dense_every: int = 3
    max_probes: int = 6
    save_frames: bool = False
    clip_seconds: float = 2.0
    clip_every: int = 3
    # E3 (Doob identity, run with --branch-k 1 so the path is pi_ref itself):
    deadline: int = 0  # >0: G = success by absolute chunk `deadline` (rollout horizon deadline-chunk-1), not anova_horizon
    probe_offset: str = "fixed"  # "random": chunk 0, then a world-seeded offset in 1..probe_every, then every probe_every
    contact_probe: bool = True  # probe immediately at contact onset (off for E3's regular grid)
    stop_after_dead: int = 0  # >0: end a world after this many consecutive probes with zero successes in all rollouts
    # E4 (two-level branching): per probe, gamma_k chunks x anova_m continuations under pi_ref AND under an
    # improved continuation (privileged best-of-gamma_kimp at every chunk). Replaces the plain ANOVA probe.
    gamma_k: int = 0
    gamma_kimp: int = 4


def _seed(*parts: int) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0] & 0x7FFFFFFF)


def _run_chunk(env: Any, policy: Any, obs: dict, seed: int, ev: dict, E: int) -> tuple[dict, int, list]:
    policy.rng_seed = seed
    actions = policy.infer(obs)
    n, done = 0, []
    for action in actions[:E]:
        env.step_one_vanilla(action)
        done.append(action)
        ev = env.evaluate_vanilla()
        n += 1
        if ev["ever_success"]:
            break
    return ev, n, done


def _in_contact(priv: dict) -> bool:
    return any(bool(v) for v in priv["touching"].values()) or any(bool(v) for v in priv["grasped"].values())


def _horizon(cfg: AnovaConfig, chunk: int) -> int:
    return max(0, int(cfg.deadline) - chunk - 1) if cfg.deadline else int(cfg.anova_horizon)


def _continue(env: Any, policy: Any, e: dict, seed_of: Any, H: int, lift_m: float, E: int, kimp: int) -> tuple[dict, int]:
    """Roll on for up to H chunks: pi_ref if kimp <= 1, else privileged best-of-kimp per chunk."""
    d = 0
    while not e["ever_success"] and d < H:
        if kimp <= 1:
            e, _, _ = _run_chunk(env, policy, env.obs_vanilla_state(), seed_of(d, 0), e, E)
        else:
            root, obs, best = snapshot(env, policy), env.obs_vanilla_state(), None
            for j in range(kimp):
                restore(env, policy, root)
                ej, _, _ = _run_chunk(env, policy, obs, seed_of(d, j), e, E)
                sc = progress_score(privileged_state(env), ej, lift_m)  # success scores 100
                if best is None or sc > best[0]:
                    best = (sc, snapshot(env, policy), ej)
            restore(env, policy, best[1])
            e = best[2]
        d += 1
    return e, d


def _anova(X: np.ndarray) -> dict:
    M = X.shape[1]
    w = float(X.var(axis=1, ddof=1).mean()) if M > 1 else float("nan")
    b = float(X.mean(axis=1).var(ddof=1) - w / M)
    return {"mean": float(X.mean()), "s2_between": b, "sigma2_within": w, "best_chunk_mean": float(X.mean(axis=1).max())}


def probe_gamma(cfg: AnovaConfig, env: Any, policy: Any, world_seed: int, chunk: int, ev0: dict, lift_m: float) -> dict:
    """E4: Q_ref(h, a) and Q_imp(h, a) for the same upstream chunks a; Gamma = Q_imp - Q_ref."""
    E, K, M, H, Ki = int(cfg.execute_chunk_dim), int(cfg.gamma_k), int(cfg.anova_m), _horizon(cfg, chunk), int(cfg.gamma_kimp)
    root = snapshot(env, policy)
    obs = env.obs_vanilla_state()
    t0 = time.perf_counter()
    cands = []
    for k in range(K):
        restore(env, policy, root)
        ev, _, _ = _run_chunk(env, policy, obs, _seed(world_seed, chunk, 7, k), ev0, E)  # = probe()'s chunk k
        priv = privileged_state(env)
        after = {"score_after": progress_score(priv, ev, lift_m), "success_after": bool(ev["ever_success"]),
                 "priv_after": _priv_summary(priv, ev)}
        s_k = snapshot(env, policy)
        reps = {}
        for arm, kimp, tag in (("ref", 1, 8), ("imp", Ki, 9)):
            reps[arm] = []
            for m in range(M):
                restore(env, policy, s_k)
                e, d = _continue(env, policy, ev, lambda d, j: _seed(world_seed, chunk, tag, k, m, d, j), H, lift_m, E, kimp)
                reps[arm].append({"success": bool(e["ever_success"]), "chunks": d,
                                  "final_score": progress_score(privileged_state(env), e, lift_m)})
        cands.append({"k": k, **after, "reps_ref": reps["ref"], "reps_imp": reps["imp"]})
    restore(env, policy, root)
    Gr = np.array([[r["success"] for r in c["reps_ref"]] for c in cands], dtype=float)
    Gi = np.array([[r["success"] for r in c["reps_imp"]] for c in cands], dtype=float)
    return {"world_seed": world_seed, "chunk": chunk, "K": K, "M": M, "H": H, "kimp": Ki,
            "wall_s": time.perf_counter() - t0, "success": _anova(Gr), "success_imp": _anova(Gi),
            "candidates": cands}


def probe(cfg: AnovaConfig, env: Any, policy: Any, world_seed: int, chunk: int, ev0: dict, lift_m: float) -> dict:
    E, K, M, H = int(cfg.execute_chunk_dim), int(cfg.anova_k), int(cfg.anova_m), _horizon(cfg, chunk)
    root = snapshot(env, policy)
    obs = env.obs_vanilla_state()
    t0 = time.perf_counter()
    cands = []
    for k in range(K):
        restore(env, policy, root)
        ev, _, _ = _run_chunk(env, policy, obs, _seed(world_seed, chunk, 7, k), ev0, E)
        priv = privileged_state(env)
        after = {"score_after": progress_score(priv, ev, lift_m), "success_after": bool(ev["ever_success"]),
                 "priv_after": _priv_summary(priv, ev)}
        s_k = snapshot(env, policy)
        reps = []
        for m in range(M):
            restore(env, policy, s_k)
            e, d, best = ev, 0, after["score_after"]
            while not e["ever_success"] and d < H:
                e, _, _ = _run_chunk(env, policy, env.obs_vanilla_state(), _seed(world_seed, chunk, 8, k, m, d), e, E)
                d += 1
                best = max(best, progress_score(privileged_state(env), e, lift_m))
            reps.append({"success": bool(e["ever_success"]), "chunks": d,
                         "final_score": progress_score(privileged_state(env), e, lift_m), "max_score": best})
        cands.append({"k": k, **after, "reps": reps})
    restore(env, policy, root)
    G = np.array([[r["success"] for r in c["reps"]] for c in cands], dtype=float)
    S = np.array([[r["final_score"] for r in c["reps"]] for c in cands], dtype=float)
    return {"world_seed": world_seed, "chunk": chunk, "K": K, "M": M, "H": H,
            "wall_s": time.perf_counter() - t0, "success": _anova(G), "final_score": _anova(S),
            "candidates": cands}


def run(cfg: AnovaConfig) -> None:
    if cfg.policy_backend != "lap" or cfg.task != "ladder" or not cfg.vanilla_physics:
        raise ValueError("variance_anova needs --policy-backend lap --task ladder --vanilla-physics")
    require_mjwarp()
    policy = _build_policy(cfg)
    env = SimEnv(height=cfg.camera_height, width=cfg.camera_width, camera_keys=cfg.model.camera_keys,
                 prompt=cfg.prompt, scene=active_scene(cfg), task=cfg.task, ladder_task=cfg.ladder_task,
                 gpu_id=cfg.gpu_id)
    lift_m = float(active_scene(cfg).lift_m)
    out = Path(cfg.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    E, Kp = int(cfg.execute_chunk_dim), int(cfg.branch_k)
    cams = tuple(cfg.model.camera_keys)
    clip_len = max(1, int(round(cfg.clip_seconds * cfg.video_fps / cfg.clip_every)))
    if cfg.save_frames:
        import imageio.v2 as imageio
        (out / "frames").mkdir(exist_ok=True)
    print(f"[anova] rung={cfg.ladder_task} K={cfg.anova_k} M={cfg.anova_m} H={cfg.anova_horizon} "
          f"probe_every={cfg.probe_every} dense_every={cfg.dense_every} max_probes={cfg.max_probes} "
          f"path=oracle K{Kp} save_frames={cfg.save_frames} deadline={cfg.deadline} probe_offset={cfg.probe_offset} "
          f"contact_probe={cfg.contact_probe} stop_after_dead={cfg.stop_after_dead} gamma_k={cfg.gamma_k} "
          f"gamma_kimp={cfg.gamma_kimp}", flush=True)
    try:
        for wi in range(cfg.num_worlds):
            seed = int(cfg.seed + wi)
            env.reset(seed=seed)
            policy.reset_episode()
            ev = env.evaluate_vanilla()
            n_probes, next_probe, contact_chunk, dead = 0, 0, None, 0
            offset = 1 + _seed(seed, 99) % int(cfg.probe_every)  # E3: random grid phase, chunk 0 always probed
            clip = [video_frame(env.obs_vanilla_state()["images"], cams)] if cfg.save_frames else []
            for chunk in range(cfg.num_chunks):
                if ev["ever_success"] or n_probes >= cfg.max_probes or (cfg.deadline and chunk >= cfg.deadline):
                    break
                if cfg.stop_after_dead and dead >= cfg.stop_after_dead:
                    print(f"[anova] seed={seed} chunk={chunk:03d} stop: {dead} probes with no success", flush=True)
                    break
                if chunk == next_probe:
                    priv_path = privileged_state(env)
                    tag = f"seed{seed}_c{chunk:03d}"
                    if cfg.save_frames:  # before probe(): it restores the root, so the frame is the probe state
                        imageio.imwrite(out / "frames" / f"{tag}.jpg", video_frame(env.obs_vanilla_state()["images"], cams),
                                        quality=92)
                        with imageio.get_writer(str(out / "frames" / f"{tag}.mp4"),
                                                fps=max(1, cfg.video_fps // cfg.clip_every), macro_block_size=1) as w:
                            for f in clip[-clip_len:]:
                                w.append_data(f)
                    p = (probe_gamma if cfg.gamma_k > 0 else probe)(cfg, env, policy, seed, chunk, ev, lift_m)
                    p["path_score"] = progress_score(priv_path, ev, lift_m)
                    p["path_priv"] = _priv_summary(priv_path, ev)
                    p["path_contact"] = _in_contact(priv_path)
                    p["prompt"] = env.prompt
                    p["contact_chunk"] = contact_chunk
                    if cfg.save_frames:
                        p["frame"], p["clip"] = f"frames/{tag}.jpg", f"frames/{tag}.mp4"
                    p["probe_weight"] = 1 if chunk == 0 or cfg.probe_offset != "random" else int(cfg.probe_every)
                    n_probes += 1
                    any_succ = p["success"]["mean"] > 0 or p.get("success_imp", {}).get("mean", 0) > 0
                    dead = 0 if any_succ else dead + 1
                    if cfg.probe_offset == "random":
                        next_probe = offset if chunk == 0 else chunk + int(cfg.probe_every)
                    else:
                        next_probe = chunk + (cfg.dense_every if contact_chunk is not None else cfg.probe_every)
                    with open(out / "probes.jsonl", "a") as fh:
                        fh.write(json.dumps(jsonable(p)) + "\n")
                    gi = p.get("success_imp")
                    print(f"[anova] seed={seed} chunk={chunk:03d} path_score={p['path_score']:+.2f} "
                          f"P(succ)={p['success']['mean']:.2f} s2={p['success']['s2_between']:.4f} "
                          + (f"P_imp={gi['mean']:.2f} s2_imp={gi['s2_between']:.4f} " if gi else "") +
                          f"sw2={p['success']['sigma2_within']:.4f} best={p['success']['best_chunk_mean']:.2f} "
                          f"wall={p['wall_s']:.0f}s", flush=True)
                # path step: privileged oracle over Kp candidates (same seeds as branch_select)
                root, obs, best = snapshot(env, policy), env.obs_vanilla_state(), None
                for k in range(Kp):
                    restore(env, policy, root)
                    e, _, acts = _run_chunk(env, policy, obs, _seed(seed, chunk, k), ev, E)
                    sc = progress_score(privileged_state(env), e, lift_m)
                    if best is None or sc > best[0]:
                        best = (sc, snapshot(env, policy), e, acts)
                if cfg.save_frames:  # replay the chosen chunk's actions (physics is deterministic) to film it
                    restore(env, policy, root)
                    for i, a in enumerate(best[3]):
                        env.step_one_vanilla(a)
                        if (i + 1) % cfg.clip_every == 0 or i + 1 == len(best[3]):
                            clip.append(video_frame(env.render_cameras_vanilla_state(), cams))
                    clip = clip[-clip_len:]
                restore(env, policy, best[1])
                ev = best[2]
                if contact_chunk is None and _in_contact(privileged_state(env)):
                    contact_chunk = chunk + 1  # probe right at contact onset, then densely
                    if cfg.contact_probe:
                        next_probe = min(next_probe, chunk + 1)
            print(f"[anova] world seed={seed} done probes={n_probes} path_success={ev['ever_success']}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    import tyro

    run(tyro.cli(AnovaConfig))
