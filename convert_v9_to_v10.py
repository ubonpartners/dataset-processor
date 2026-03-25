#!/usr/bin/env python3
"""
Patch a multilabel checkpoint (nc=main_nc+attr_nc, attr_nc=0 on head)
into an attribute-head checkpoint (nc=main_nc, attr_nc=attr_nc on head).

No weight changes are needed: cv3's final 1x1 already outputs nc+attr_nc channels
in both the old multilabel layout and the new ubon layout.  We only patch metadata.

Usage:
    python tools/convert_multilabel_pt_to_attributes_pt.py \\
        --src old_model.pt --dst new_model.pt --main-nc 4 --attr-nc 6
"""
import argparse
from pathlib import Path

import torch


def _infer_split(names) -> tuple[int, int]:
    """Find where attribute classes start by looking for the first person_xx / person:xx name."""
    name_list = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    for i, n in enumerate(name_list):
        if n.startswith("person_") or n.startswith("person:"):
            return i, len(name_list) - i
    raise ValueError(
        f"Could not infer split: no class name starts with 'person_' or 'person:' in {name_list}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",      required=True, type=Path)
    ap.add_argument("--dst",      required=True, type=Path)
    ap.add_argument("--main-nc",  default=None, type=int, help="Number of main detection classes (inferred if omitted)")
    ap.add_argument("--attr-nc",  default=None, type=int, help="Number of attribute outputs (inferred if omitted)")
    ap.add_argument("--attr-names", nargs="+", default=None,
                    help="Attribute names (optional; defaults to trailing entries in model.names)")
    args = ap.parse_args()

    try:
        from ultralytics.nn.tasks import torch_safe_load
        ckpt, _ = torch_safe_load(str(args.src))
    except Exception:
        ckpt = torch.load(args.src, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise TypeError("Unsupported checkpoint format: expected dict with 'model' key")

    # Infer split from model.names if not supplied.
    if args.main_nc is None or args.attr_nc is None:
        nc_main, attr_nc = _infer_split(ckpt["model"].names)
        print(f"Inferred: main_nc={nc_main}, attr_nc={attr_nc}")
    else:
        nc_main, attr_nc = args.main_nc, args.attr_nc
    nc_total = nc_main + attr_nc

    for key in ("model", "ema"):
        model = ckpt.get(key)
        if model is None:
            continue

        head = model.model[-1]

        assert head.nc == nc_total, (
            f"Expected head.nc={nc_total} (main_nc={nc_main}+attr_nc={attr_nc}), got {head.nc}"
        )
        assert int(getattr(head, "attr_nc", 0) or 0) == 0, (
            "Checkpoint already has attr_nc > 0; no conversion needed"
        )

        # Patch head metadata only — weights are already in the right layout.
        head.nc = nc_main
        head.attr_nc = attr_nc
        # head.no stays the same: nc_main + attr_nc + reg_max*4 == nc_total + reg_max*4

        # Split model.names into main names and attr names.
        old_names = model.names
        if isinstance(old_names, dict):
            model.names  = {i: old_names[i] for i in range(nc_main)}
            trailing     = [old_names.get(nc_main + i, f"attr_{i}") for i in range(attr_nc)]
        else:
            model.names  = list(old_names[:nc_main])
            trailing     = list(old_names[nc_main:nc_total])

        if args.attr_names:
            assert len(args.attr_names) == attr_nc, \
                f"--attr-names has {len(args.attr_names)} entries, expected {attr_nc}"
            model.attr_names = args.attr_names
        else:
            # Convert person_male → person:male convention if applicable.
            model.attr_names = [
                n.replace("person_", "person:", 1) if n.startswith("person_") else n
                for n in trailing
            ]

        # Patch stored task so YOLO._load uses the correct predictor.
        # For .pt files, YOLO uses model.task directly (guess_model_task is never called).
        from ultralytics.nn.modules.head import PoseReID as _PoseReID
        model.task = "posereid" if isinstance(head, _PoseReID) else "pose"

        if hasattr(model, "yaml") and isinstance(model.yaml, dict):
            model.yaml["nc"]         = nc_main
            model.yaml["attr_nc"]    = attr_nc
            model.yaml["attr_names"] = model.attr_names
            model.yaml["task"]       = model.task

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.dst)

    m = ckpt["model"]
    print(f"Saved → {args.dst}")
    print(f"  nc={m.model[-1].nc}, attr_nc={m.model[-1].attr_nc}")
    print(f"  names:      {m.names}")
    print(f"  attr_names: {m.attr_names}")


if __name__ == "__main__":
    main()
