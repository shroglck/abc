"""SimEvalConfig exactly as ego-lap's eval_rhb_abc_bottles.sh launches a run.

Offline tools (replays, scripted rollouts, audits) must not build the config from the
dataclass defaults: the launcher overrides bridge settings that change the joint
commands (`--no-lap-align-eef-rot`, `--no-lap-binarize-gripper`, `--lap-eef-link`) and
the scene, so a defaults-built config replays a different trajectory. Keep this list in
sync with that script.
"""

from __future__ import annotations

from typing import Any

LAUNCHER_FLAGS = [
    "--policy-backend", "lap",
    "--scene.bottle-count", "6",
    "--scene.table-bounds", "0.3025", "0.8975", "-0.65", "0.65",
    "--scene.bin-xy-range", "0.57", "0.73", "-0.25", "0.25",
    "--scene.bin-scale-range", "0.95", "1.05",
    "--scene.bin-z-scale", "0.83",
    "--lap-eef-link", "grasp_site",
    "--no-lap-align-eef-rot",
    "--no-lap-binarize-gripper",
    "--vanilla-physics",
    "--task", "ladder",
]


def launcher_config(ladder_task: str, prompt: str = "", extra: list[str] | None = None, cls: Any = None) -> Any:
    import tyro

    from abc_minimal.config import SimEvalConfig

    args = LAUNCHER_FLAGS + ["--ladder-task", ladder_task] + (["--prompt", prompt] if prompt else []) + list(extra or [])
    return tyro.cli(cls or SimEvalConfig, args=args)
