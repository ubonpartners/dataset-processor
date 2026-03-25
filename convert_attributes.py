#!/usr/bin/env python3
"""
Convert v9 dataset to v10 (split attribute format) at /mldata/v10.
- Copies all image files (no symlinks).
- Converts label files from multi-row-per-object (55 "classes") to one-row-per-object
  with main class (0-4) + 50-dim attribute vector + keypoints.
- Writes updated dataset.yaml per subdataset (path, nc, names, attributes, attr_nc, attr_label_format).
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

# Default paths
DEFAULT_V9 = Path("/mldata/v9")
DEFAULT_V10 = Path("/mldata/v10")

# Image extensions to copy
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Main vs attribute: first 5 are main classes, rest are attributes
MAIN_NC = 5
ATTR_NC = 50
ROUND_DECIMALS = 6
KPT_ROUND_DECIMALS = 4


def _flow_style_list(items: list) -> CommentedSeq:
    """Return a CommentedSeq that will be dumped as compact [a, b, c]."""
    seq = CommentedSeq(items)
    seq.fa.set_flow_style()
    return seq


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML (plain dict, for reading names/kpt_shape etc)."""
    yaml_loader = YAML(typ="safe")
    with open(path, encoding="utf-8") as f:
        return yaml_loader.load(f) or {}


def load_yaml_round_trip(path: Path):
    """Load YAML with round-trip so comments are preserved. Returns ruamel data."""
    yaml_rt = YAML(typ="rt")
    yaml_rt.preserve_quotes = True
    with open(path, encoding="utf-8") as f:
        return yaml_rt.load(f)


def save_yaml_v10(
    path: Path,
    data,
    main_names: list[str],
    attr_names: list[str],
    v10_path_str: str,
    dataset_name: str | None,
    config_name: str | None,
    kpt_shape: list[int],
    flip_idx: list[int] | None,
) -> None:
    """
    Write v10 dataset.yaml. Uses round-trip style so existing comments are preserved.
    - kpt_shape and flip_idx written as compact [a, b, c].
    - names and attr_names stay block (multi-line).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build/update the structure in place on the loaded data so comments are kept
    data["path"] = v10_path_str
    data["train"] = "train/images"
    data["val"] = "val/images"
    data["nc"] = MAIN_NC
    data["names"] = main_names
    data["attributes"] = True
    data["attr_nc"] = ATTR_NC
    data["attr_label_format"] = "split"
    data["attr_names"] = attr_names
    data["kpt_shape"] = _flow_style_list(kpt_shape)
    if flip_idx:
        data["flip_idx"] = _flow_style_list(flip_idx)
    if dataset_name is not None:
        data["dataset_name"] = dataset_name.replace("v9", "v10")
    if config_name is not None:
        data["config_name"] = config_name

    yaml_rt = YAML(typ="rt")
    yaml_rt.preserve_quotes = True
    yaml_rt.default_flow_style = None
    yaml_rt.width = 4096  # keep kpt_shape/flip_idx on one line
    with open(path, "w", encoding="utf-8") as f:
        yaml_rt.dump(data, f)


def parse_v9_row(line: str, num_cols: int) -> tuple[int, list[float], list[float]] | None:
    """Parse one v9 label line. Returns (cls, bbox_xywh, keypoints) or None if invalid."""
    parts = line.split()
    if len(parts) != num_cols:
        return None
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        return None
    cls = int(vals[0])
    bbox = vals[1:5]
    kpts = vals[5:] if len(vals) > 5 else []
    return cls, bbox, kpts


def group_key(bbox: list[float], kpts: list[float]) -> tuple:
    """Stable key for grouping rows that represent the same object."""
    b = tuple(round(x, ROUND_DECIMALS) for x in bbox)
    k = tuple(round(x, KPT_ROUND_DECIMALS) for x in kpts) if kpts else ()
    return (b, k)


def convert_label_file(
    v9_path: Path,
    v10_path: Path,
    main_nc: int,
    attr_nc: int,
    has_keypoints: bool,
    nkpt: int,
    ndim: int,
) -> tuple[int, int]:
    """
    Convert one v9 label file to v10 split format.
    Returns (num_objects_in, num_objects_out).
    """
    v9_path = Path(v9_path)
    v10_path = Path(v10_path)
    v10_path.parent.mkdir(parents=True, exist_ok=True)

    if not v9_path.exists():
        v10_path.write_text("", encoding="utf-8")
        return 0, 0

    text = v9_path.read_text(encoding="utf-8").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if not lines:
        v10_path.write_text("", encoding="utf-8")
        return 0, 0

    # v9: 1 + 4 + nkpt*ndim
    nkpt_dim = nkpt * ndim if has_keypoints else 0
    num_cols = 5 + nkpt_dim

    # group rows by (bbox, keypoints)
    groups: dict[tuple, list[tuple[int, list[float], list[float]]]] = defaultdict(list)
    for line in lines:
        row = parse_v9_row(line, num_cols)
        if row is None:
            continue
        cls, bbox, kpts = row
        key = group_key(bbox, kpts)
        groups[key].append((cls, bbox, kpts))

    out_rows: list[str] = []
    no_main_count = 0

    for (_bbox_key, _kpt_key), rows in groups.items():
        main_classes = sorted({r[0] for r in rows if 0 <= r[0] < main_nc})
        attr_indices = {r[0] - main_nc for r in rows if main_nc <= r[0] < main_nc + attr_nc}

        if not main_classes:
            no_main_count += 1
            main_classes = [0]

        # One output row per main class (e.g. person and face on same box -> two rows)
        x, y, w, h = rows[0][1]
        kpts = rows[0][2]
        attr_vec = [1.0 if i in attr_indices else 0.0 for i in range(attr_nc)]

        for cls_main in main_classes:
            parts = [str(cls_main), f"{x:.6g}", f"{y:.6g}", f"{w:.6g}", f"{h:.6g}"]
            parts.extend(f"{a:.6g}" for a in attr_vec)
            if kpts:
                parts.extend(f"{k:.6g}" for k in kpts)
            out_rows.append(" ".join(parts))

    if no_main_count:
        print(f"  [warn] {v9_path.name}: {no_main_count} groups had no main class, defaulted to 0")

    v10_path.write_text("\n".join(out_rows) + ("\n" if out_rows else ""), encoding="utf-8")
    return len(lines), len(out_rows)


def copy_images(v9_images_dir: Path, v10_images_dir: Path) -> int:
    """Copy all image files. Returns number of files copied."""
    v10_images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    if not v9_images_dir.exists():
        return 0
    for f in v9_images_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            shutil.copy2(f, v10_images_dir / f.name)
            count += 1
    return count


def discover_splits(v9_sub_path: Path) -> list[str]:
    """Return list of split names (e.g. ['train', 'val'])."""
    if not v9_sub_path.is_dir():
        return []
    return [d.name for d in v9_sub_path.iterdir() if d.is_dir() and (d / "images").exists()]


def _get_names_list(data) -> list[str]:
    """Extract names as a list from yaml data (list or dict)."""
    names = data.get("names") or []
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    return list(names)


def process_subdataset(
    v9_base: Path,
    v10_base: Path,
    sub_name: str,
    main_nc: int,
    attr_nc: int,
) -> None:
    v9_sub = v9_base / sub_name
    v10_sub = v10_base / sub_name
    v9_yaml = v9_sub / "dataset.yaml"
    if not v9_yaml.exists():
        print(f"Skip {sub_name}: no dataset.yaml")
        return

    # Load with round-trip so we can preserve comments when writing
    data_rt = load_yaml_round_trip(v9_yaml)
    if data_rt is None:
        data_rt = {}
    names = _get_names_list(data_rt)
    kpt_shape = data_rt.get("kpt_shape") or [0, 0]
    kpt_shape = list(kpt_shape)[:2]
    nkpt = int(kpt_shape[0]) if len(kpt_shape) >= 1 else 0
    ndim = int(kpt_shape[1]) if len(kpt_shape) >= 2 else 0
    has_kpt = nkpt > 0 and ndim in (2, 3)
    flip_idx = list(data_rt.get("flip_idx") or [])

    splits = discover_splits(v9_sub)
    for split in splits:
        v9_images = v9_sub / split / "images"
        v10_images = v10_sub / split / "images"
        n_img = copy_images(v9_images, v10_images)
        print(f"  {sub_name}/{split}: copied {n_img} images")

        v9_labels = v9_sub / split / "labels"
        v10_labels = v10_sub / split / "labels"
        if not v9_labels.exists():
            v10_labels.mkdir(parents=True, exist_ok=True)
            continue

        total_in = 0
        total_out = 0
        for lbl in v9_labels.glob("*.txt"):
            v10_lbl = v10_labels / lbl.name
            nin, nout = convert_label_file(
                lbl, v10_lbl, main_nc, attr_nc, has_kpt, nkpt, ndim
            )
            total_in += nin
            total_out += nout
        print(f"  {sub_name}/{split}: labels {total_in} v9 rows -> {total_out} v10 rows")

    # Write dataset.yaml for v10 (compact kpt_shape/flip_idx, block names/attr_names, preserve comments)
    main_names = names[:MAIN_NC]
    if len(main_names) < MAIN_NC:
        main_names = main_names + ["unknown"] * (MAIN_NC - len(main_names))
    attr_names = names[MAIN_NC : MAIN_NC + ATTR_NC]
    if len(attr_names) < ATTR_NC:
        attr_names = attr_names + [f"attr_{i}" for i in range(len(attr_names), ATTR_NC)]
    save_yaml_v10(
        path=v10_sub / "dataset.yaml",
        data=data_rt,
        main_names=main_names,
        attr_names=attr_names,
        v10_path_str=str(v10_base / sub_name),
        dataset_name=data_rt.get("dataset_name"),
        config_name=data_rt.get("config_name"),
        kpt_shape=[int(kpt_shape[0]), int(kpt_shape[1])] if len(kpt_shape) >= 2 else [0, 0],
        flip_idx=flip_idx if flip_idx else None,
    )
    print(f"  {sub_name}: wrote dataset.yaml (path={v10_base / sub_name})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert v9 dataset to v10 (split attribute format). Copies images (no symlinks)."
    )
    parser.add_argument(
        "--v9",
        type=Path,
        default=DEFAULT_V9,
        help="v9 dataset root (default: /mldata/v9)",
    )
    parser.add_argument(
        "--v10",
        type=Path,
        default=DEFAULT_V10,
        help="v10 output root (default: /mldata/v10)",
    )
    parser.add_argument(
        "--subdataset",
        type=str,
        default=None,
        help="Process only this subdataset (e.g. coco). Default: all.",
    )
    args = parser.parse_args()

    v9_base = args.v9.resolve()
    v10_base = args.v10.resolve()
    if not v9_base.exists():
        raise SystemExit(f"v9 path does not exist: {v9_base}")

    v10_base.mkdir(parents=True, exist_ok=True)

    subdatasets = [d.name for d in v9_base.iterdir() if d.is_dir() and (d / "dataset.yaml").exists()]
    if args.subdataset:
        if args.subdataset not in subdatasets:
            raise SystemExit(f"Subdataset '{args.subdataset}' not found. Available: {subdatasets}")
        subdatasets = [args.subdataset]

    print(f"Converting v9 -> v10 (copy files, split attribute format)")
    print(f"  v9: {v9_base}")
    print(f"  v10: {v10_base}")
    print(f"  subdatasets: {subdatasets}")

    for sub in subdatasets:
        print(f"\n{sub}:")
        process_subdataset(v9_base, v10_base, sub, main_nc=MAIN_NC, attr_nc=ATTR_NC)


if __name__ == "__main__":
    main()
