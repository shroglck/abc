"""Gemini Robotics-ER 2 as the branch selector for branch_select.

One request per decision: the BEFORE frame plus each candidate's AFTER frame (every
candidate executed from the same snapshot), and the model returns the index of the
candidate that best advances the task, plus a progress estimate per candidate. The
simulator, not Gemini, decides task success; the privileged progress score is logged
next to Gemini's pick so its selection regret is measurable.

Context levels (study step 7 -- how much information the selector needs):
  outcomes             images only: BEFORE + each candidate's AFTER (simulated lookahead)
  outcomes+privileged  same, plus ground-truth text per candidate (lift, grasp, reach,
                       arm-scenery contacts) that the cameras may not show

REST via `requests` (no google-genai in the sim venv). Key from $GOOGLE_API_KEY.
Network failures retry, then fall back to candidate 0 and are counted, never hidden.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any

import numpy as np
import requests
from PIL import Image

DEFAULT_MODEL = "gemini-robotics-er-2-preview"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
CONTEXTS = ("outcomes", "outcomes+privileged")

_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "best": {"type": "INTEGER"},
        "progress": {"type": "ARRAY", "items": {"type": "NUMBER"}},
        "reason": {"type": "STRING"},
    },
    "required": ["best", "progress", "reason"],
}


def _jpeg_part(image_chw: np.ndarray) -> dict[str, Any]:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(np.asarray(image_chw).transpose(1, 2, 0))).save(buf, format="JPEG", quality=90)
    return {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}}


def _frame(images: dict[str, np.ndarray], cameras: tuple[str, ...]) -> np.ndarray:
    """Side-by-side cameras as one CHW image (fewer image parts, same pixels)."""
    return np.concatenate([np.asarray(images[c]) for c in cameras], axis=2)


class GeminiSelector:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        context: str = "outcomes",
        cameras: tuple[str, ...] = ("top",),
        timeout_s: float = 45.0,
        retries: int = 4,
        temperature: float = 0.0,
        retry_temperature: float = 0.7,
        thinking_budget: int | None = 1024,
    ):
        if context not in CONTEXTS:
            raise ValueError(f"gemini context must be one of {CONTEXTS}, got {context!r}")
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is unset (non-interactive shells skip ~/.bashrc; source it before sbatch)")
        self.model, self.context, self.cameras = model, context, tuple(cameras)
        self.timeout_s, self.retries, self.temperature = timeout_s, retries, temperature
        self.retry_temperature, self.thinking_budget = retry_temperature, thinking_budget
        self._key = key
        self._session = self._new_session()
        self.calls = self.failures = 0
        self.latency_s = 0.0

    def _new_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({"x-goog-api-key": self._key, "Content-Type": "application/json"})
        return session

    def _prompt(self, task: str, n: int, priv: list[dict[str, Any]] | None) -> str:
        lines = [
            f"Task for the bimanual robot: \"{task}\".",
            f"The first image is the scene NOW. Each of the next {n} images is the scene after executing "
            f"one candidate action chunk (~0.5 s) from that same starting state; they are candidates 0..{n - 1} "
            "in order. Cameras are placed side by side: " + ", ".join(self.cameras) + ".",
            "Pick the candidate whose outcome best advances THIS task as worded (e.g. reaching for and grasping the "
            "object the task names, then doing with it what the task asks: placing, stacking on the named object, "
            "or pressing and rubbing it across a surface to clean it). Penalize knocking objects away, dropping a "
            "held object, using the wrong object, collisions, and doing nothing.",
            f"Return `best` (an index 0..{n - 1}), `progress` (exactly {n} numbers in [0,1], the task progress "
            "after each candidate), and a one-sentence `reason`.",
        ]
        if priv is not None:
            lines.append("Ground-truth simulator facts after each candidate (trust these over the images if they conflict):")
            for k, p in enumerate(priv):
                line = (f"  candidate {k}: target lift {100 * p['lift_m']:.1f} cm, nearest gripper {100 * p['reach_m']:.1f} cm "
                        f"from target, grasped={p['grasped']}, touching={p['touching']}, "
                        f"arm-scenery contacts={p['arm_world_contacts']}")
                if "container_lift_m" in p:
                    line += (f", container lift {100 * p['container_lift_m']:.1f} cm (held up={p['container_held']}, "
                             f"arms touching it={p.get('container_arms') or 'none'}, arms gripping the target={p.get('target_arms') or 'none'})")
                if "target_to_dirt_m" in p:
                    line += f", target is {100 * p['target_to_dirt_m']:.1f} cm (3D) from the stains"
                if "wipe_cleaned_frac" in p:
                    line += (f", sponge on stains={p.get('wipe_in_contact')}, scrubbed {100 * p.get('wipe_total_stroke_m', 0):.1f} cm "
                             f"so far, stains cleaned {100 * p['wipe_cleaned_frac']:.0f}%")
                if "on_top_now" in p:
                    line += f", resting on the named support={p['on_top_now']}"
                lines.append(line)
        return "\n".join(lines)

    def choose(
        self,
        task: str,
        before: dict[str, np.ndarray],
        afters: list[dict[str, np.ndarray]],
        priv: list[dict[str, Any]] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        n = len(afters)
        parts: list[dict[str, Any]] = [{"text": self._prompt(task, n, priv if self.context == "outcomes+privileged" else None)}]
        parts.append(_jpeg_part(_frame(before, self.cameras)))
        for k, after in enumerate(afters):
            parts.append({"text": f"candidate {k}:"})
            parts.append(_jpeg_part(_frame(after, self.cameras)))
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
                "responseSchema": _SCHEMA,
            },
        }
        if self.thinking_budget is not None:
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": int(self.thinking_budget)}
        url = ENDPOINT.format(model=self.model)
        err = None
        for attempt in range(self.retries):
            if attempt:
                # Temperature 0 makes a retry the same computation as the attempt that
                # just stalled; resample instead.
                body["generationConfig"]["temperature"] = self.retry_temperature
            t0 = time.perf_counter()
            try:
                self.calls += 1
                r = self._session.post(url, data=json.dumps(body), timeout=self.timeout_s)
                self.latency_s += time.perf_counter() - t0
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                payload = r.json()
                text = "".join(p.get("text", "") for p in payload["candidates"][0]["content"]["parts"])
                out = json.loads(text)
                best = int(out["best"])
                if not 0 <= best < n:
                    raise ValueError(f"best={best} outside 0..{n - 1}")
                usage = payload.get("usageMetadata", {})
                return best, {
                    "best": best,
                    "progress": [float(v) for v in out.get("progress", [])],
                    "reason": out.get("reason", ""),
                    "latency_s": time.perf_counter() - t0,
                    "tokens_in": usage.get("promptTokenCount"),
                    "tokens_out": usage.get("candidatesTokenCount"),
                    "tokens_thinking": usage.get("thoughtsTokenCount"),
                    "attempts": attempt + 1,
                }
            except Exception as e:  # network, quota, malformed JSON: retry, then fall back
                err = f"{type(e).__name__}: {e}"
                # Stalls are runaway hidden thinking, not the network: visible output is
                # ~75 tokens and uncorrelated with latency (r=0.10), a fresh connection
                # still stalled (job 4076297), and at temperature 0 every retry repeated
                # the same long computation. The budget above bounds it; fresh session kept
                # as cheap insurance.
                self._session.close()
                self._session = self._new_session()
                time.sleep(min(4.0, 0.5 * 2**attempt))
        self.failures += 1
        return 0, {"best": 0, "error": err, "fallback": True}
