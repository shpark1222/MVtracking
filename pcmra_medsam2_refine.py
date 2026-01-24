from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import tempfile
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

MEDSAM2_PY = r"C:\Users\show2\miniconda3\envs\medsam2\python.exe"
MEDSAM2_RUNNER = r"C:\Users\show2\MedSAM2\medsam2_infer.py"
MEDSAM2_CKPT = r"C:\Users\show2\MedSAM2\checkpoints\MedSAM2_latest.pt"
MEDSAM2_CONFIG = "configs/sam2.1_hiera_t512.yaml"
MEDSAM2_DEVICE = "cpu"

DEFAULT_SETTINGS = {
    "python": MEDSAM2_PY,
    "runner": MEDSAM2_RUNNER,
    "checkpoint": MEDSAM2_CKPT,
    "config": MEDSAM2_CONFIG,
    "device": MEDSAM2_DEVICE,
}

SETTINGS_ENV_MAP = {
    "python": "MEDSAM2_PY",
    "runner": "MEDSAM2_RUNNER",
    "checkpoint": "MEDSAM2_CKPT",
    "config": "MEDSAM2_CONFIG",
    "device": "MEDSAM2_DEVICE",
}

DEFAULT_SETTINGS_PATH = Path(__file__).with_name("medsam2_settings.json")


def _load_settings_json(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_medsam2_settings(settings_path: Optional[Path] = None) -> dict:
    env_path = os.environ.get("MEDSAM2_SETTINGS_PATH")
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


def run_medsam2_subprocess(
    img_u8: np.ndarray,
    box_xyxy: Sequence[float],
    point_xy: Optional[Sequence[float]] = None,
    tmpdir: Optional[Path] = None,
    settings: Optional[dict] = None,
) -> np.ndarray:
    settings = settings or get_medsam2_settings()

    def _run_in_dir(td: Path) -> np.ndarray:
        td.mkdir(parents=True, exist_ok=True)
        in_npy = td / "input.npy"
        pr_json = td / "prompt.json"
        out_npy = td / "mask.npy"

        np.save(in_npy, img_u8)

        prompt = {"box_xyxy": [float(v) for v in box_xyxy]}
        if point_xy is not None:
            prompt["point_xy"] = [float(point_xy[0]), float(point_xy[1])]
            prompt["point_label"] = 1

        pr_json.write_text(json.dumps(prompt), encoding="utf-8")

        cmd = [
            settings["python"],
            settings["runner"],
            "--input",
            str(in_npy),
            "--prompt",
            str(pr_json),
            "--output",
            str(out_npy),
            "--ckpt",
            settings["checkpoint"],
            "--config",
            settings["config"],
            "--device",
            settings.get("device", MEDSAM2_DEVICE),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(
                "MedSAM2 failed\nSTDOUT:\n" + (p.stdout or "") + "\nSTDERR:\n" + (p.stderr or "")
            )

        if not out_npy.exists():
            raise RuntimeError("MedSAM2 failed to produce mask.npy")
        mask = np.load(out_npy)
        return (mask > 0).astype(np.uint8)

    if tmpdir is not None:
        return _run_in_dir(Path(tmpdir))

    with tempfile.TemporaryDirectory(prefix="vt_medsam2_") as td:
        return _run_in_dir(Path(td))


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    try:
        from scipy.ndimage import label

        labels, n_labels = label(mask)
        if n_labels <= 1:
            return mask
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        largest = np.argmax(counts)
        return (labels == largest).astype(np.uint8)
    except Exception:
        return mask


def _extract_contour_points(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.empty((0, 2), dtype=np.float64)
        contour = max(contours, key=cv2.contourArea)
        contour = contour.reshape(-1, 2).astype(np.float64)
        return contour
    except Exception:
        pass

    try:
        from skimage import measure

        contours = measure.find_contours(mask.astype(np.float32), 0.5)
        if not contours:
            return np.empty((0, 2), dtype=np.float64)
        contour = max(contours, key=len)
        contour = np.asarray(contour, dtype=np.float64)
        if contour.shape[1] == 2:
            contour = contour[:, ::-1]
        return contour
    except Exception:
        return np.empty((0, 2), dtype=np.float64)


def _resample_closed_curve(pts_xy: np.ndarray, target_n: int) -> np.ndarray:
    pts = np.asarray(pts_xy, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return pts
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[0]])
    d = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return pts[:-1]
    s_new = np.linspace(0, s[-1], target_n + 1, endpoint=True)[:-1]
    x_new = np.interp(s_new, s, pts[:, 0])
    y_new = np.interp(s_new, s, pts[:, 1])
    return np.column_stack([x_new, y_new]).astype(np.float64)


def mask_to_polygon_points(mask: np.ndarray) -> List[List[float]]:
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.ndim != 2:
        raise ValueError("mask must be 2D")
    if mask_u8.sum() == 0:
        return []

    mask_largest = _largest_connected_component(mask_u8)
    contour = _extract_contour_points(mask_largest)
    if contour.size == 0:
        return []

    target_n = int(np.clip(contour.shape[0], 64, 128))
    contour = _resample_closed_curve(contour, target_n)
    return contour.astype(np.float64).tolist()


def polygon_points_to_roi_state(
    pts_xy: Iterable[Sequence[float]],
    current_state: Optional[dict] = None,
    shape_hw: Optional[Tuple[int, int]] = None,
) -> dict:
    pts = np.asarray(list(pts_xy), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        raise ValueError("Need at least 3 points for polygon ROI")

    if current_state is None:
        return {"pos": (0.0, 0.0), "points": pts.tolist(), "closed": True}

    pos = np.array(current_state.get("pos", (0.0, 0.0)), dtype=np.float64)
    ref_pts = np.array(current_state.get("points", []), dtype=np.float64)

    use_local = True
    if shape_hw is not None and ref_pts.ndim == 2 and ref_pts.shape[0] >= 3:
        H, W = shape_hw

        cand_local = ref_pts + pos[None, :]
        cand_abs = ref_pts

        def score(P: np.ndarray) -> float:
            x, y = P[:, 0], P[:, 1]
            return float(np.mean((x >= 0) & (x < W) & (y >= 0) & (y < H)))

        use_local = score(cand_local) >= score(cand_abs)

    if use_local:
        points_local = pts - pos[None, :]
        return {"pos": (float(pos[0]), float(pos[1])), "points": points_local.tolist(), "closed": True}

    return {"pos": (0.0, 0.0), "points": pts.tolist(), "closed": True}
