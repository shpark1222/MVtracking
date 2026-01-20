from typing import Tuple

import numpy as np


def remember_eps(x: float) -> float:
    x = float(x)
    return x if x != 0.0 else 1e-12


def closed_spline_xy(pts_xy: np.ndarray, n_out: int = 400) -> np.ndarray:
    pts = np.asarray(pts_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return pts

    if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[0]])

    try:
        from scipy.interpolate import splprep, splev

        x = pts[:, 0]
        y = pts[:, 1]
        k = min(3, len(x) - 1)
        tck, _ = splprep([x, y], s=0.0, per=True, k=k)
        u_new = np.linspace(0, 1, n_out, endpoint=False)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack([x_new, y_new]).astype(np.float64)
    except Exception:
        d = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
        s = np.hstack([[0.0], np.cumsum(d)])
        if s[-1] <= 0:
            return pts[:-1]
        s_new = np.linspace(0, s[-1], n_out, endpoint=False)
        x_new = np.interp(s_new, s, pts[:, 0])
        y_new = np.interp(s_new, s, pts[:, 1])
        return np.column_stack([x_new, y_new]).astype(np.float64)


def polygon_mask(shape: Tuple[int, int], pts_xy: np.ndarray) -> np.ndarray:
    H, W = shape
    if pts_xy is None or len(pts_xy) < 3:
        return np.zeros((H, W), dtype=bool)

    xs = pts_xy[:, 0]
    ys = pts_xy[:, 1]
    xmin = int(np.clip(np.floor(xs.min()), 0, W - 1))
    xmax = int(np.clip(np.ceil(xs.max()), 0, W - 1))
    ymin = int(np.clip(np.floor(ys.min()), 0, H - 1))
    ymax = int(np.clip(np.ceil(ys.max()), 0, H - 1))

    if xmax <= xmin or ymax <= ymin:
        return np.zeros((H, W), dtype=bool)

    X, Y = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    xq = X.astype(np.float64)
    yq = Y.astype(np.float64)

    inside = np.zeros_like(xq, dtype=bool)
    x = xs
    y = ys
    n = len(x)

    j = n - 1
    for i in range(n):
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]
        intersect = ((yi > yq) != (yj > yq)) & (xq < (xj - xi) * (yq - yi) / remember_eps(yj - yi) + xi)
        inside ^= intersect
        j = i

    mask = np.zeros((H, W), dtype=bool)
    mask[ymin : ymax + 1, xmin : xmax + 1] = inside
    return mask
