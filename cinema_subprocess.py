from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
from typing import Optional, Sequence, Tuple

import numpy as np

CINEMA_PY = r"C:\Users\show2\miniconda3\envs\cinema\python.exe"
CINEMA_RUNNER = r"C:\Users\show2\CineMA\cinema_infer_mv.py"

DEFAULT_SETTINGS = {
    "python": CINEMA_PY,
    "runner": CINEMA_RUNNER,
    "debug_dir": "",
    "debug_keep": False,
}

SETTINGS_ENV_MAP = {
    "python": "CINEMA_PY",
    "runner": "CINEMA_RUNNER",
}

DEFAULT_SETTINGS_PATH = Path(__file__).with_name("cinema_settings.json")


def _load_settings_json(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_cinema_settings(settings_path: Optional[Path] = None) -> dict:
    env_path = os.environ.get("CINEMA_SETTINGS_PATH")
    if env_path:
        settings_path = Path(env_path)
    if settings_path is None and DEFAULT_SETTINGS_PATH.exists():
        settings_path = DEFAULT_SETTINGS_PATH

    settings = dict(DEFAULT_SETTINGS)
    settings.update(_load_settings_json(settings_path))

    for key, env_name in SETTINGS_ENV_MAP.items():
        env_val = os.environ.get(env_name)
        if env_val:
            settings[key] = env_val
    return settings


def _log_run_info(
    *,
    cine: np.ndarray,
    view: str,
    cmd: Sequence[str],
    stdout: str,
    stderr: str,
) -> None:
    stats = {
        "shape": cine.shape,
        "dtype": str(cine.dtype),
        "min": float(np.min(cine)) if cine.size else 0.0,
        "max": float(np.max(cine)) if cine.size else 0.0,
    }
    cmd_str = subprocess.list2cmdline(list(cmd))
    print(
        "CineMA input stats: "
        f"shape={stats['shape']}, dtype={stats['dtype']}, "
        f"min={stats['min']:.3f}, max={stats['max']:.3f}, view={view}",
        file=sys.stderr,
    )
    print(f"CineMA command: {cmd_str}", file=sys.stderr)
    print("CineMA stdout:\n" + (stdout or ""), file=sys.stderr)
    print("CineMA stderr:\n" + (stderr or ""), file=sys.stderr)


def run_cinema_subprocess(
    cine_stack: np.ndarray,
    view: str,
    tmpdir: Optional[Path] = None,
    settings: Optional[dict] = None,
) -> Tuple[dict, str, str]:
    settings = settings or get_cinema_settings()
    cine_stack = np.asarray(cine_stack, dtype=np.float32)
    if cine_stack.ndim != 3:
        raise ValueError(f"cine_stack must have shape (T,H,W), got {cine_stack.shape}")
    if not view:
        raise ValueError("view is required")

    runner_cwd = Path(settings["runner"]).parent
    debug_dir = settings.get("debug_dir") or ""
    debug_keep = bool(settings.get("debug_keep"))

    def _run_in_dir(td: Path) -> Tuple[dict, str, str]:
        td.mkdir(parents=True, exist_ok=True)
        input_npz = td / "input.npz"
        output_json = td / "output.json"
        np.savez(input_npz, cine=cine_stack, view=view)

        cmd = [
            settings["python"],
            settings["runner"],
            "--input",
            str(input_npz),
            "--output",
            str(output_json),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=runner_cwd)
        _log_run_info(
            cine=cine_stack,
            view=view,
            cmd=cmd,
            stdout=p.stdout or "",
            stderr=p.stderr or "",
        )
        if p.returncode != 0:
            raise RuntimeError(
                "CineMA failed\nSTDOUT:\n" + (p.stdout or "") + "\nSTDERR:\n" + (p.stderr or "")
            )
        if not output_json.exists():
            raise RuntimeError("CineMA failed to produce output.json")

        output = json.loads(output_json.read_text(encoding="utf-8"))
        return output, (p.stdout or ""), (p.stderr or "")

    if tmpdir is not None:
        return _run_in_dir(Path(tmpdir))

    if debug_keep and debug_dir:
        return _run_in_dir(Path(debug_dir))

    with tempfile.TemporaryDirectory(prefix="vt_cinema_") as td:
        return _run_in_dir(Path(td))
