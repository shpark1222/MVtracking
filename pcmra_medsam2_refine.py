from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
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
    "debug_dir": "",
    "debug_keep": False,
    "force_512": True,
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


def _get_cv2():
    if importlib.util.find_spec("cv2") is not None:
        import cv2

        return cv2
    return None


def _get_pil_image():
    if importlib.util.find_spec("PIL.Image") is not None:
        from PIL import Image

        return Image
    return None


def to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    p1, p99 = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmax <= vmin:
            return np.zeros(arr.shape, dtype=np.uint8)
    else:
        vmin = float(p1)
        vmax = float(p99)
    scaled = (arr - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def _resize_u8_nn(img_u8: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    src_h, src_w = img_u8.shape[:2]
    if (src_h, src_w) == (target_h, target_w):
        return img_u8
    y_idx = np.clip(
        np.round(np.linspace(0, src_h - 1, target_h)).astype(np.int64), 0, src_h - 1
    )
    x_idx = np.clip(
        np.round(np.linspace(0, src_w - 1, target_w)).astype(np.int64), 0, src_w - 1
    )
    if img_u8.ndim == 2:
        return img_u8[np.ix_(y_idx, x_idx)]
    return img_u8[np.ix_(y_idx, x_idx, np.arange(img_u8.shape[2]))]


def _resize_to_512(
    img_u8: np.ndarray,
    box_xyxy: Sequence[float],
    point_xy: Optional[Sequence[float]],
    target: int = 512,
) -> Tuple[np.ndarray, List[float], Optional[List[float]], Tuple[int, int]]:
    h, w = img_u8.shape[:2]
    if (h, w) == (target, target):
        return (
            img_u8,
            [float(v) for v in box_xyxy],
            None if point_xy is None else [float(point_xy[0]), float(point_xy[1])],
            (h, w),
        )

    cv2 = _get_cv2()
    if cv2 is not None:
        img_rs = cv2.resize(img_u8, (target, target), interpolation=cv2.INTER_LINEAR)
    else:
        pil_image = _get_pil_image()
        if pil_image is not None:
            img_rs = np.array(pil_image.fromarray(img_u8).resize((target, target)))
        else:
            img_rs = _resize_u8_nn(img_u8, (target, target))

    scale_x = float(target) / float(w)
    scale_y = float(target) / float(h)

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    box_rs = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

    pt_rs = None
    if point_xy is not None:
        pt_rs = [float(point_xy[0]) * scale_x, float(point_xy[1]) * scale_y]

    return img_rs.astype(np.uint8), box_rs, pt_rs, (h, w)


def _resize_mask_back(mask: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    if mask.shape[:2] == (out_h, out_w):
        return mask
    cv2 = _get_cv2()
    if cv2 is not None:
        return cv2.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    pil_image = _get_pil_image()
    if pil_image is not None:
        return np.array(
            pil_image.fromarray(mask.astype(np.uint8)).resize(
                (out_w, out_h), resample=pil_image.NEAREST
            )
        )
    return _resize_u8_nn(mask.astype(np.uint8), (out_h, out_w))


def _pad_box(box: Sequence[float], target_hw: Tuple[int, int], frac: float = 0.2) -> List[float]:
    h, w = target_hw
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = x2 - x1
    bh = y2 - y1
    px = bw * frac
    py = bh * frac
    x1 = max(0.0, x1 - px)
    y1 = max(0.0, y1 - py)
    x2 = min(float(w - 1), x2 + px)
    y2 = min(float(h - 1), y2 + py)
    return [x1, y1, x2, y2]


def _log_empty_mask(
    *,
    stage: str,
    img_shape: Tuple[int, int],
    box_xyxy: Sequence[float],
    point_xy: Optional[Sequence[float]],
    target: int,
    resized: bool,
    stdout: str,
    stderr: str,
) -> None:
    msg = (
        "MedSAM2 empty mask: "
        f"stage={stage}, input_shape={img_shape}, box={list(box_xyxy)}, "
        f"point={None if point_xy is None else list(point_xy)}, "
        f"target={target}, resized={resized}"
    )
    print(msg, file=sys.stderr)
    if stdout:
        print("MedSAM2 stdout:\n" + stdout, file=sys.stderr)
    if stderr:
        print("MedSAM2 stderr:\n" + stderr, file=sys.stderr)


def _log_run_info(
    *,
    img: np.ndarray,
    box_xyxy: Sequence[float],
    point_xy: Optional[Sequence[float]],
    cmd: Sequence[str],
    stdout: str,
    stderr: str,
) -> None:
    stats = {
        "shape": img.shape,
        "dtype": str(img.dtype),
        "min": float(np.min(img)) if img.size else 0.0,
        "max": float(np.max(img)) if img.size else 0.0,
        "p1": float(np.percentile(img, 1.0)) if img.size else 0.0,
        "p99": float(np.percentile(img, 99.0)) if img.size else 0.0,
    }
    cmd_str = subprocess.list2cmdline(list(cmd))
    print(
        "MedSAM2 input stats: "
        f"shape={stats['shape']}, dtype={stats['dtype']}, "
        f"min={stats['min']:.3f}, max={stats['max']:.3f}, "
        f"p1={stats['p1']:.3f}, p99={stats['p99']:.3f}",
        file=sys.stderr,
    )
    print(f"MedSAM2 prompt: box={list(box_xyxy)}, point={None if point_xy is None else list(point_xy)}", file=sys.stderr)
    print(f"MedSAM2 command: {cmd_str}", file=sys.stderr)
    print("MedSAM2 stdout:\n" + (stdout or ""), file=sys.stderr)
    print("MedSAM2 stderr:\n" + (stderr or ""), file=sys.stderr)


def run_medsam2_subprocess(
    img_u8: np.ndarray,
    box_xyxy: Sequence[float],
    point_xy: Optional[Sequence[float]] = None,
    tmpdir: Optional[Path] = None,
    settings: Optional[dict] = None,
) -> np.ndarray:
    settings = settings or get_medsam2_settings()
    img_u8 = to_uint8(img_u8)
    debug_dir = settings.get("debug_dir") or ""
    debug_keep = bool(settings.get("debug_keep"))
    force_512 = bool(settings.get("force_512", True))
    runner_cwd = Path(settings["runner"]).parent

    def _run_once(td: Path, img_in: np.ndarray, box_in: Sequence[float], pt_in: Optional[Sequence[float]]):
        td.mkdir(parents=True, exist_ok=True)
        in_npy = td / "input.npy"
        pr_json = td / "prompt.json"
        out_npy = td / "mask.npy"

        np.save(in_npy, img_in)

        prompt = {"box_xyxy": [float(v) for v in box_in]}
        if pt_in is not None:
            prompt["point_xy"] = [float(pt_in[0]), float(pt_in[1])]
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
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=runner_cwd)
        _log_run_info(
            img=img_in,
            box_xyxy=box_in,
            point_xy=pt_in,
            cmd=cmd,
            stdout=p.stdout or "",
            stderr=p.stderr or "",
        )
        if p.returncode != 0:
            raise RuntimeError(
                "MedSAM2 failed\nSTDOUT:\n" + (p.stdout or "") + "\nSTDERR:\n" + (p.stderr or "")
            )

        if not out_npy.exists():
            raise RuntimeError("MedSAM2 failed to produce mask.npy")
        mask = np.load(out_npy)
        return (mask > 0).astype(np.uint8), (p.stdout or ""), (p.stderr or "")

    def _run_in_dir(td: Path) -> np.ndarray:
        if force_512:
            img_rs, box_rs, pt_rs, orig_hw = _resize_to_512(
                img_u8, box_xyxy, point_xy, target=512
            )
        else:
            img_rs = img_u8
            box_rs = [float(v) for v in box_xyxy]
            pt_rs = None if point_xy is None else [float(point_xy[0]), float(point_xy[1])]
            orig_hw = img_u8.shape[:2]
        resized = img_rs.shape[:2] != img_u8.shape[:2]

        mask_512, stdout, stderr = _run_once(td, img_rs, box_rs, pt_rs)
        if mask_512.sum() == 0:
            _log_empty_mask(
                stage="initial",
                img_shape=img_u8.shape[:2],
                box_xyxy=box_xyxy,
                point_xy=point_xy,
                target=512,
                resized=resized,
                stdout=stdout,
                stderr=stderr,
            )
            box_pad = _pad_box(box_rs, img_rs.shape[:2], frac=0.2)
            mask_512, stdout, stderr = _run_once(td, img_rs, box_pad, pt_rs)

        if mask_512.sum() == 0 and pt_rs is not None:
            box_pad = _pad_box(box_rs, img_rs.shape[:2], frac=0.2)
            mask_512, stdout, stderr = _run_once(td, img_rs, box_pad, None)

        if mask_512.sum() == 0:
            _log_empty_mask(
                stage="final",
                img_shape=img_u8.shape[:2],
                box_xyxy=box_xyxy,
                point_xy=point_xy,
                target=512,
                resized=resized,
                stdout=stdout,
                stderr=stderr,
            )

        mask = _resize_mask_back(mask_512, orig_hw)
        return (mask > 0).astype(np.uint8)

    if tmpdir is not None:
        return _run_in_dir(Path(tmpdir))

    if debug_keep and debug_dir:
        return _run_in_dir(Path(debug_dir))

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


def _sample_contour_points(pts_xy: np.ndarray, n_points: int) -> np.ndarray:
    pts = np.asarray(pts_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return pts
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    idx = np.linspace(0, pts.shape[0] - 1, n_points, dtype=int)
    return pts[idx].astype(np.float64)


def mask_to_polygon_points(mask: np.ndarray) -> List[List[float]]:
    import cv2

    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.ndim != 2:
        raise ValueError("mask must be 2D")

    m = (mask_u8 > 0).astype(np.uint8) * 255
    nnz = int(np.count_nonzero(m))
    if nnz == 0:
        print(
            "[mask_to_polygon_points] failed: empty mask, nnz=0, components=0, "
            "contours_a=0, contours_b=0, path=A->B->C"
        )
        return []

    _, labels = cv2.connectedComponents((m > 0).astype(np.uint8))
    components = int(labels.max())

    contour = None
    contours_a, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_a_count = len(contours_a)
    if contours_a:
        contour = max(contours_a, key=lambda c: (cv2.contourArea(c), c.shape[0]))
        path_taken = "A"
    else:
        kernel = np.ones((5, 5), np.uint8)
        m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        h, w = m2.shape
        ff = m2.copy()
        tmp = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, tmp, (0, 0), 255)
        ff_inv = cv2.bitwise_not(ff)
        filled = cv2.bitwise_or(m2, ff_inv)
        contours_b, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_b_count = len(contours_b)
        if contours_b:
            contour = max(contours_b, key=lambda c: (cv2.contourArea(c), c.shape[0]))
            path_taken = "A->B"
        else:
            ys, xs = np.nonzero(m)
            if ys.size:
                pts = np.stack([xs, ys], axis=1).astype(np.int32)
                contour = cv2.convexHull(pts)
                path_taken = "A->B->C"
            else:
                path_taken = "A->B->C (no points)"

    if contour is None or len(contour) == 0:
        print(
            "[mask_to_polygon_points] failed: "
            f"nnz={nnz}, components={components}, contours_a={contours_a_count}, "
            f"contours_b={contours_b_count}, path={path_taken}"
        )
        return []

    contour = contour.reshape(-1, 2).astype(np.float64)
    contour = _sample_contour_points(contour, n_points=10)
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
