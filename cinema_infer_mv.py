from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def _resolve_convvit_class():
    import importlib

    try:
        cinema_pkg = importlib.import_module("cinema")
        if hasattr(cinema_pkg, "ConvViT"):
            return cinema_pkg.ConvViT
    except Exception:
        pass

    module_candidates = [
        "cinema.convvit",
        "cinema.conv_vit",
        "cinema.models.convvit",
        "cinema.models.conv_vit",
        "cinema.model.convvit",
        "cinema.model.conv_vit",
    ]

    for module_name in module_candidates:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(mod, "ConvViT"):
            return mod.ConvViT

    raise ImportError(
        "ConvViT class not found. "
        "Check that CineMA is installed in the selected conda env and that import cinema works."
    )


def _load_input(path: Path) -> Tuple[np.ndarray, str]:
    data = np.load(path, allow_pickle=True)
    if "cine" not in data or "view" not in data:
        raise ValueError("input.npz must contain 'cine' and 'view'")
    cine = np.asarray(data["cine"], dtype=np.float32)
    view = data["view"]
    if isinstance(view, np.ndarray):
        view = view.item()
    view = str(view)
    if cine.ndim != 3:
        raise ValueError(f"cine must have shape (T,H,W), got {cine.shape}")
    return cine, view


def _normalize_view(view: str) -> str:
    view = view.strip()
    if view in ("lax_2c", "lax_4c"):
        return view
    if view in ("lax_3c", "lax_3ch", "3ch"):
        return "lax_4c"
    raise ValueError(f"Unsupported view: {view}")


def run_inference(cine: np.ndarray, view: str) -> dict:
    view = _normalize_view(view)
    convvit_cls = _resolve_convvit_class()
    model = convvit_cls.from_finetuned(
        repo_id="mathpluscode/CineMA",
        model_filename=f"finetuned/landmark_coordinate/{view}/{view}_0.safetensors",
        config_filename=f"finetuned/landmark_coordinate/{view}/config.yaml",
    )
    model.to("cpu")
    model.eval()

    coords_list = []
    _, H, W = cine.shape
    pad_h = (2 - (H % 2)) % 2
    pad_w = (2 - (W % 2)) % 2
    Hp = H + pad_h
    Wp = W + pad_w
    scale = np.array([Wp, Hp, Wp, Hp, Wp, Hp], dtype=np.float32)
    with torch.no_grad():
        for t in range(cine.shape[0]):
            frame = cine[t]
            if pad_h or pad_w:
                frame = np.pad(frame, ((0, pad_h), (0, pad_w)), mode="constant")
            tensor = torch.from_numpy(frame)[None, None, ...]
            coords = model({view: tensor})[0]
            if isinstance(coords, torch.Tensor):
                coords = coords.detach().cpu().numpy()
            coords = np.asarray(coords, dtype=np.float32).reshape(-1)
            if coords.shape[0] < 6:
                raise RuntimeError(f"Invalid landmark output shape: {coords.shape}")
            coords = coords[:6] * scale
            coords[0::2] = np.clip(coords[0::2], 0, W - 1)
            coords[1::2] = np.clip(coords[1::2], 0, H - 1)
            coords_list.append(coords)

    coords_all = np.stack(coords_list, axis=1)
    mv_coords = coords_all[2:6, :]
    return {
        "view": view,
        "coords": coords_all.tolist(),
        "mv": mv_coords.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cine, view = _load_input(Path(args.input))
    output = run_inference(cine, view)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
