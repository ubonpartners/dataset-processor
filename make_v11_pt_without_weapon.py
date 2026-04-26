#!/usr/bin/env python3
"""Create a v11 checkpoint by removing the standalone weapon detection class.

This patches the classifier output channels for common Ultralytics YOLO heads:
Detect, Pose, Pose26, PoseReID/Pose26ReID, and their end-to-end one2one variants.
Box, keypoint, ReID, backbone, and neck weights are left unchanged.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_REMOVE_CLASS_NAME = "weapon"
DEFAULT_ULTRALYTICS_REPO = Path(__file__).resolve().parent.parent / "ultralytics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="Source v10 .pt checkpoint")
    parser.add_argument("--dst", required=True, type=Path, help="Destination v11 .pt checkpoint")
    parser.add_argument("--remove-class", default=DEFAULT_REMOVE_CLASS_NAME, help="Class name to remove")
    parser.add_argument("--remove-index", type=int, default=None, help="Class index to remove if name is unavailable")
    parser.add_argument(
        "--ultralytics-repo",
        type=Path,
        default=Path(os.environ.get("ULTRALYTICS_REPO", DEFAULT_ULTRALYTICS_REPO)),
        help="Ultralytics repo to prepend to PYTHONPATH for custom checkpoint classes",
    )
    parser.add_argument("--dry-run", action="store_true", help="Patch in memory and report without saving")
    return parser.parse_args()


def add_ultralytics_repo(path: Path) -> None:
    if path.is_dir():
        sys.path.insert(0, str(path))


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        from ultralytics.nn.tasks import torch_safe_load

        ckpt, _ = torch_safe_load(str(path))
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise TypeError("Unsupported checkpoint format: expected a dict with a 'model' key")
    return ckpt


def names_as_list(names: Any, nc: int) -> list[str]:
    if isinstance(names, dict):
        return [names.get(i, str(i)) for i in range(nc)]
    if names is None:
        return [str(i) for i in range(nc)]
    return list(names)[:nc]


def infer_remove_index(model: nn.Module, class_name: str, fallback: int | None) -> int:
    head = model.model[-1]
    nc = int(head.nc)
    names = names_as_list(getattr(model, "names", None), nc)
    if class_name in names:
        return names.index(class_name)
    if fallback is not None:
        return fallback
    raise ValueError(f"Class {class_name!r} not found in model.names={names}; pass --remove-index")


def find_and_replace_last_conv(module: nn.Module, keep_rows: torch.Tensor) -> bool:
    """Replace the last Conv2d in module, slicing output rows by keep_rows."""
    if isinstance(module, nn.Conv2d):
        raise ValueError("Cannot replace a bare Conv2d without its parent module")

    for name, child in reversed(list(module._modules.items())):
        if isinstance(child, nn.Conv2d):
            old = child
            if old.out_channels < int(keep_rows.max()) + 1:
                raise ValueError(
                    f"Classifier conv has only {old.out_channels} outputs, cannot keep row {int(keep_rows.max())}"
                )
            new = nn.Conv2d(
                in_channels=old.in_channels,
                out_channels=len(keep_rows),
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                dilation=old.dilation,
                groups=old.groups,
                bias=old.bias is not None,
                padding_mode=old.padding_mode,
            )
            new.to(device=old.weight.device, dtype=old.weight.dtype)
            rows = keep_rows.to(device=old.weight.device)
            new.weight.data.copy_(old.weight.data.index_select(0, rows))
            new.weight.requires_grad = old.weight.requires_grad
            if old.bias is not None:
                new.bias.data.copy_(old.bias.data.index_select(0, rows))
                new.bias.requires_grad = old.bias.requires_grad
            module._modules[name] = new
            return True
        if find_and_replace_last_conv(child, keep_rows):
            return True
    return False


def patch_classifier_branch(branch: nn.Module, keep_rows: torch.Tensor, branch_name: str) -> list[tuple[int, int]]:
    if branch is None:
        return []
    if not isinstance(branch, nn.ModuleList):
        raise TypeError(f"Expected {branch_name} to be nn.ModuleList, got {type(branch).__name__}")

    shapes: list[tuple[int, int]] = []
    for i, module in enumerate(branch):
        before = None
        for child in module.modules():
            if isinstance(child, nn.Conv2d):
                before = child.out_channels
        if before is None:
            raise ValueError(f"No Conv2d found in {branch_name}[{i}]")
        if not find_and_replace_last_conv(module, keep_rows):
            raise ValueError(f"Could not replace final Conv2d in {branch_name}[{i}]")
        after = None
        for child in module.modules():
            if isinstance(child, nn.Conv2d):
                after = child.out_channels
        shapes.append((before, int(after)))
    return shapes


def set_yaml_metadata(model: nn.Module, names: list[str], attr_nc: int) -> None:
    if hasattr(model, "yaml") and isinstance(model.yaml, dict):
        model.yaml["nc"] = len(names)
        model.yaml["names"] = names
        if attr_nc:
            model.yaml["attr_nc"] = attr_nc


def rewrite_v10_strings(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("/v10/", "/v11/").replace("-v10", "-v11")
    if isinstance(value, list):
        return [rewrite_v10_strings(v) for v in value]
    if isinstance(value, tuple):
        return tuple(rewrite_v10_strings(v) for v in value)
    if isinstance(value, dict):
        return {k: rewrite_v10_strings(v) for k, v in value.items()}
    return value


def patch_model(model: nn.Module, class_name: str, remove_index_arg: int | None) -> dict[str, Any]:
    head = model.model[-1]
    old_nc = int(head.nc)
    attr_nc = int(getattr(head, "attr_nc", 0) or 0)
    remove_index = infer_remove_index(model, class_name, remove_index_arg)
    if remove_index < 0 or remove_index >= old_nc:
        raise ValueError(f"remove_index={remove_index} is outside class range 0..{old_nc - 1}")

    old_names = names_as_list(getattr(model, "names", None), old_nc)
    new_names = [name for i, name in enumerate(old_names) if i != remove_index]
    keep_classes = [i for i in range(old_nc) if i != remove_index]
    keep_rows = torch.tensor(keep_classes + list(range(old_nc, old_nc + attr_nc)), dtype=torch.long)

    patched: dict[str, list[tuple[int, int]]] = {}
    for branch_name in ("cv3", "one2one_cv3"):
        branch = getattr(head, branch_name, None)
        if branch is not None:
            patched[branch_name] = patch_classifier_branch(branch, keep_rows, branch_name)

    if not patched:
        raise ValueError("No classifier branches found; expected head.cv3 and/or head.one2one_cv3")

    head.nc = old_nc - 1
    if hasattr(head, "no"):
        reg_max = int(getattr(head, "reg_max", 0) or 0)
        head.no = head.nc + attr_nc + (4 * reg_max)

    model.names = {i: name for i, name in enumerate(new_names)}
    set_yaml_metadata(model, new_names, attr_nc)

    return {
        "head": type(head).__name__,
        "old_nc": old_nc,
        "new_nc": head.nc,
        "attr_nc": attr_nc,
        "remove_index": remove_index,
        "names": model.names,
        "patched": patched,
    }


def main() -> None:
    args = parse_args()
    add_ultralytics_repo(args.ultralytics_repo.resolve())
    ckpt = load_checkpoint(args.src)

    reports: dict[str, Any] = {}
    seen: set[int] = set()
    for key in ("model", "ema"):
        model = ckpt.get(key)
        if model is None or id(model) in seen:
            continue
        seen.add(id(model))
        reports[key] = patch_model(model, args.remove_class, args.remove_index)

    for metadata_key in ("train_args", "train_metrics", "train_results"):
        if metadata_key in ckpt:
            ckpt[metadata_key] = rewrite_v10_strings(ckpt[metadata_key])

    for key, report in reports.items():
        print(f"{key}: {report['head']} nc {report['old_nc']} -> {report['new_nc']}, attr_nc={report['attr_nc']}")
        print(f"  removed index {report['remove_index']}; names={report['names']}")
        for branch_name, shapes in report["patched"].items():
            print(f"  {branch_name}: {shapes}")

    if args.dry_run:
        print("Dry run only; not saving")
        return

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.dst)
    print(f"Saved -> {args.dst}")


if __name__ == "__main__":
    main()
