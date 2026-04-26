#!/usr/bin/env python3
"""Create a v11 dataset copy from v10 with the standalone weapon class removed.

The converter physically copies files from /mldata/v10 to /mldata/v11, then edits
the copied label/YAML files in place. Symlinks are dereferenced during the copy so
the destination contains regular files/directories, not links.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from tqdm import tqdm


DEFAULT_SRC = Path("/mldata/v10")
DEFAULT_DST = Path("/mldata/v11")
DEFAULT_REMOVE_CLASS_NAME = "weapon"
DEFAULT_REMOVE_CLASS_INDEX = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help=f"Source dataset root (default: {DEFAULT_SRC})")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help=f"Destination dataset root (default: {DEFAULT_DST})")
    parser.add_argument("--remove-class", default=DEFAULT_REMOVE_CLASS_NAME, help="Class name to remove from YAML names")
    parser.add_argument(
        "--remove-index",
        type=int,
        default=DEFAULT_REMOVE_CLASS_INDEX,
        help=f"Class index to remove from labels if it cannot be inferred (default: {DEFAULT_REMOVE_CLASS_INDEX})",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing destination before copying")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report what would change without copying")
    return parser.parse_args()


def class_names_as_list(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def infer_remove_index(src: Path, class_name: str, fallback: int) -> int:
    yaml = YAML(typ="safe")
    for yaml_path in sorted(src.glob("*/dataset.yaml")) + sorted(src.glob("*.yaml")):
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.load(f) or {}
        names = data.get("names")
        if not names:
            continue
        names_list = class_names_as_list(names)
        if class_name in names_list:
            return names_list.index(class_name)
    return fallback


def rewrite_path_string(value: str, src: Path, dst: Path) -> str:
    src_s = str(src)
    dst_s = str(dst)
    value = value.replace(src_s, dst_s)
    return value.replace("/v10/", "/v11/").replace("-v10", "-v11")


def rewrite_strings(value: Any, src: Path, dst: Path) -> Any:
    if isinstance(value, str):
        return rewrite_path_string(value, src, dst)
    if isinstance(value, list):
        for i, item in enumerate(value):
            value[i] = rewrite_strings(item, src, dst)
    elif isinstance(value, dict):
        for key in list(value.keys()):
            value[key] = rewrite_strings(value[key], src, dst)
    return value


def remove_name(names: Any, remove_index: int) -> Any:
    names_list = class_names_as_list(names)
    if remove_index >= len(names_list):
        raise ValueError(f"remove_index={remove_index} is outside YAML names list of length {len(names_list)}")
    names_list.pop(remove_index)
    return names_list


def rewrite_dataset_yaml(yaml_path: Path, src: Path, dst: Path, remove_index: int) -> None:
    yaml = YAML()
    yaml.preserve_quotes = True
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.load(f) or {}

    data = rewrite_strings(data, src, dst)
    if "path" in data:
        data["path"] = str(yaml_path.parent)
    if "dataset_name" in data and isinstance(data["dataset_name"], str):
        data["dataset_name"] = data["dataset_name"].replace("-v10", "-v11").replace("v10", "v11")
    if "names" in data:
        data["names"] = remove_name(data["names"], remove_index)
    if "nc" in data:
        data["nc"] = int(data["nc"]) - 1

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)


def rewrite_label_file(label_path: Path, remove_index: int) -> tuple[int, int]:
    kept: list[str] = []
    removed = 0
    changed = 0

    text = label_path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        if cls == remove_index:
            removed += 1
            continue
        if cls > remove_index:
            parts[0] = str(cls - 1)
            changed += 1
        else:
            parts[0] = str(cls)
        kept.append(" ".join(parts))

    label_path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    return removed, changed


def ignore_caches(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name.endswith(".cache")}


def count_labels(src: Path, remove_index: int) -> tuple[int, int]:
    files = 0
    weapon_rows = 0
    for label_path in src.rglob("labels/*.txt"):
        files += 1
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and int(float(parts[0])) == remove_index:
                weapon_rows += 1
    return files, weapon_rows


def count_symlinks(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_symlink())


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Source dataset root does not exist: {src}")
    if src == dst:
        raise ValueError("--src and --dst must be different directories")

    remove_index = infer_remove_index(src, args.remove_class, args.remove_index)
    print(f"Removing class index {remove_index} ({args.remove_class!r})")

    if args.dry_run:
        files, weapon_rows = count_labels(src, remove_index)
        yamls = len(list(src.rglob("dataset.yaml")))
        print(f"Would copy {src} -> {dst} with symlinks dereferenced")
        print(f"Would rewrite {yamls} dataset.yaml files")
        print(f"Would scan {files} label files and remove {weapon_rows} rows")
        return

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination already exists: {dst} (use --overwrite to replace it)")
        shutil.rmtree(dst)

    print(f"Copying {src} -> {dst} (regular files, no symlinks)")
    shutil.copytree(src, dst, symlinks=False, ignore=ignore_caches, ignore_dangling_symlinks=True)

    yaml_paths = sorted(dst.rglob("dataset.yaml"))
    for yaml_path in yaml_paths:
        rewrite_dataset_yaml(yaml_path, src, dst, remove_index)

    label_paths = sorted(dst.rglob("labels/*.txt"))
    removed_rows = 0
    remapped_rows = 0
    for label_path in tqdm(label_paths, desc="Rewriting labels", colour="#008000", smoothing=0.01):
        removed, remapped = rewrite_label_file(label_path, remove_index)
        removed_rows += removed
        remapped_rows += remapped

    symlinks = count_symlinks(dst)
    if symlinks:
        raise RuntimeError(f"Destination contains {symlinks} symlinks; expected none")

    print(f"Saved v11 dataset: {dst}")
    print(f"Rewrote {len(yaml_paths)} YAML files and {len(label_paths)} label files")
    print(f"Removed {removed_rows} weapon rows; remapped {remapped_rows} rows above removed class")


if __name__ == "__main__":
    main()
