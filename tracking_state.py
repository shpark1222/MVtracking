import os
import glob
import json
from typing import Optional

import h5py
import numpy as np


def mvtrack_path_for_folder(folder: str) -> str:
    return os.path.join(folder, "MVtrack.h5")


def find_mvtrack_files(folder: str) -> list[str]:
    hits = glob.glob(os.path.join(folder, "MVtrack*.h5"))
    return sorted(hits)


def save_tracking_state_h5(
    path: str,
    line_norm: list,
    roi_state: list,
    roi_locked: list,
    metrics_Q: np.ndarray,
    metrics_Vpk: np.ndarray,
    metrics_Vmn: np.ndarray,
    metrics_KE: np.ndarray,
    metrics_VortPk: np.ndarray,
    metrics_VortMn: np.ndarray,
    active_cine_key: str,
    display_levels: dict,
    vel_auto_once: dict,
    cine_levels: tuple,
    cine_auto_once: bool,
    pcmra_levels: tuple,
    pcmra_auto_once: bool,
    cine_flip: tuple,
    cine_swap: str,
):
    Nt = len(line_norm)

    ln = np.full((Nt, 2, 2), np.nan, dtype=np.float64)
    for t in range(Nt):
        if line_norm[t] is None:
            continue
        a = np.array(line_norm[t], dtype=np.float64)
        if a.shape == (2, 2):
            ln[t] = a

    rs = []
    for t in range(Nt):
        st = roi_state[t]
        if st is None:
            rs.append("")
        else:
            rs.append(json.dumps(st))

    rs = np.array(rs, dtype=h5py.string_dtype(encoding="utf-8"))
    locked = np.array([1 if v else 0 for v in roi_locked], dtype=np.uint8)

    with h5py.File(path, "w") as f:
        g = f.create_group("state")
        g.create_dataset("line_norm", data=ln, compression="gzip")
        g.create_dataset("roi_state_json", data=rs, compression="gzip")
        g.create_dataset("roi_locked", data=locked, compression="gzip")
        g.create_dataset("metrics_Q", data=np.asarray(metrics_Q, dtype=np.float64), compression="gzip")
        g.create_dataset("metrics_Vpk", data=np.asarray(metrics_Vpk, dtype=np.float64), compression="gzip")
        g.create_dataset("metrics_Vmn", data=np.asarray(metrics_Vmn, dtype=np.float64), compression="gzip")
        g.create_dataset("metrics_KE", data=np.asarray(metrics_KE, dtype=np.float64), compression="gzip")
        g.create_dataset("metrics_VortPk", data=np.asarray(metrics_VortPk, dtype=np.float64), compression="gzip")
        g.create_dataset("metrics_VortMn", data=np.asarray(metrics_VortMn, dtype=np.float64), compression="gzip")
        g.attrs["active_cine_key"] = str(active_cine_key)
        g.attrs["display_levels_json"] = json.dumps(display_levels)
        g.attrs["vel_auto_once_json"] = json.dumps(vel_auto_once)
        g.attrs["cine_levels_json"] = json.dumps(cine_levels)
        g.attrs["cine_auto_once"] = int(bool(cine_auto_once))
        g.attrs["pcmra_levels_json"] = json.dumps(pcmra_levels)
        g.attrs["pcmra_auto_once"] = int(bool(pcmra_auto_once))
        g.attrs["cine_flip_json"] = json.dumps(cine_flip)
        g.attrs["cine_swap"] = str(cine_swap)


def load_tracking_state_h5(path: str, expected_Nt: int) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        display_levels = "{}"
        vel_auto_once = "{}"
        cine_levels = "[null, null]"
        cine_auto_once = 1
        pcmra_levels = "[null, null]"
        pcmra_auto_once = 1
        cine_flip = "[false, false, false]"
        cine_swap = "X Y Z"
        with h5py.File(path, "r") as f:
            if "/state/line_norm" not in f:
                return None
            ln = np.array(f["/state/line_norm"][()], dtype=np.float64)
            rs = np.array(f["/state/roi_state_json"][()], dtype=object)
            Q = np.array(f["/state/metrics_Q"][()], dtype=np.float64)
            Vpk = np.array(f["/state/metrics_Vpk"][()], dtype=np.float64)
            Vmn = np.array(f["/state/metrics_Vmn"][()], dtype=np.float64)
            KE = np.array(f["/state/metrics_KE"][()], dtype=np.float64) if "/state/metrics_KE" in f else None
            VortPk = (
                np.array(f["/state/metrics_VortPk"][()], dtype=np.float64)
                if "/state/metrics_VortPk" in f
                else None
            )
            VortMn = (
                np.array(f["/state/metrics_VortMn"][()], dtype=np.float64)
                if "/state/metrics_VortMn" in f
                else None
            )
            active = f["/state"].attrs.get("active_cine_key", "")
            locked = None
            if "/state/roi_locked" in f:
                locked = np.array(f["/state/roi_locked"][()], dtype=np.uint8)
            display_levels = f["/state"].attrs.get("display_levels_json", display_levels)
            vel_auto_once = f["/state"].attrs.get("vel_auto_once_json", vel_auto_once)
            cine_levels = f["/state"].attrs.get("cine_levels_json", cine_levels)
            cine_auto_once = f["/state"].attrs.get("cine_auto_once", cine_auto_once)
            pcmra_levels = f["/state"].attrs.get("pcmra_levels_json", pcmra_levels)
            pcmra_auto_once = f["/state"].attrs.get("pcmra_auto_once", pcmra_auto_once)
            cine_flip = f["/state"].attrs.get("cine_flip_json", cine_flip)
            cine_swap = f["/state"].attrs.get("cine_swap", cine_swap)

        Nt = min(expected_Nt, ln.shape[0], Q.shape[0], Vpk.shape[0], Vmn.shape[0], rs.shape[0])
        if locked is not None:
            Nt = min(Nt, locked.shape[0])
        if KE is not None:
            Nt = min(Nt, KE.shape[0])
        if VortPk is not None:
            Nt = min(Nt, VortPk.shape[0])
        if VortMn is not None:
            Nt = min(Nt, VortMn.shape[0])

        out_line = [None] * expected_Nt
        for t in range(Nt):
            if np.all(np.isnan(ln[t])):
                continue
            out_line[t] = ln[t].copy()

        out_roi = [None] * expected_Nt
        for t in range(Nt):
            s = rs[t]
            if s is None:
                continue
            if isinstance(s, (bytes, np.bytes_)):
                try:
                    s = s.decode("utf-8")
                except Exception:
                    s = s.decode("utf-8", errors="ignore")
            s = str(s)
            if not s:
                continue
            try:
                out_roi[t] = json.loads(s)
            except Exception:
                out_roi[t] = None

        out_locked = [False] * expected_Nt
        if locked is not None:
            for t in range(Nt):
                out_locked[t] = bool(locked[t])

        return {
            "line_norm": out_line,
            "roi_state": out_roi,
            "roi_locked": out_locked,
            "metrics_Q": Q,
            "metrics_Vpk": Vpk,
            "metrics_Vmn": Vmn,
            "metrics_KE": KE,
            "metrics_VortPk": VortPk,
            "metrics_VortMn": VortMn,
            "active_cine_key": str(active),
            "display_levels_json": display_levels,
            "vel_auto_once_json": vel_auto_once,
            "cine_levels_json": cine_levels,
            "cine_auto_once": int(cine_auto_once),
            "pcmra_levels_json": pcmra_levels,
            "pcmra_auto_once": int(pcmra_auto_once),
            "cine_flip_json": cine_flip,
            "cine_swap": str(cine_swap),
        }
    except Exception:
        return None
