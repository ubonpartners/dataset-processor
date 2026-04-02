#!/usr/bin/env python3
"""Mirror a dataset tree while transcoding JPEG images with jpegli.

Example:
    python transcode_jpegli_dataset.py \
        --src /mldata/v10 \
        --dst /mldata/v10-opt \
        --workers 8 \
        --quality 90 \
        --max-side 1280
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import piexif
from tqdm import tqdm

JPEG_EXTS = {".jpg", ".jpeg"}


@dataclass
class Job:
    src: Path
    dst: Path
    is_jpeg: bool


@dataclass
class JobResult:
    ok: bool
    action: str
    src_bytes: int
    dst_bytes: int
    resized: bool
    skipped: bool
    error: str = ""


def is_jpeg_path(path: Path) -> bool:
    return path.suffix.lower() in JPEG_EXTS


def iter_jobs(src_root: Path, dst_root: Path) -> list[Job]:
    jobs: list[Job] = []
    for root, _dirs, files in os.walk(src_root):
        root_path = Path(root)
        rel_root = root_path.relative_to(src_root)
        for name in files:
            src = root_path / name
            dst = dst_root / rel_root / name
            jobs.append(Job(src=src, dst=dst, is_jpeg=is_jpeg_path(src)))
    return jobs


def copy_non_jpeg(src: Path, dst: Path, overwrite: bool) -> JobResult:
    src_bytes = src.stat().st_size
    if dst.exists() and not overwrite:
        return JobResult(True, "copy", src_bytes, dst.stat().st_size, False, True)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    dst_bytes = dst.stat().st_size
    return JobResult(True, "copy", src_bytes, dst_bytes, False, False)


def _run_cjpegli(input_path: Path, output_path: Path, quality: int) -> None:
    subprocess.run(
        [
            "cjpegli",
            str(input_path),
            str(output_path),
            f"--quality={quality}",
            "--chroma_subsampling=420",
            "--progressive_level=2",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def transcode_jpeg(
    src: Path,
    dst: Path,
    quality: int,
    max_side: int,
    min_reduction_pct: float,
    preserve_exif: bool,
    overwrite: bool,
) -> JobResult:
    src_bytes = src.stat().st_size
    if dst.exists() and not overwrite:
        return JobResult(True, "jpeg", src_bytes, dst.stat().st_size, False, True)

    dst.parent.mkdir(parents=True, exist_ok=True)
    exif_bytes: bytes | None = None
    resized = False
    tmp_ppm_path: Path | None = None
    tmp_jpeg_path: Path | None = None

    try:
        with Image.open(src) as im:
            w, h = im.size
            exif_bytes = im.info.get("exif") if preserve_exif else None
            ratio = min(1.0, float(max_side) / float(max(w, h)))
            if ratio < 0.999999:
                resized = True
                nw = max(1, int(round(w * ratio)))
                nh = max(1, int(round(h * ratio)))
                resized_im = im.convert("RGB").resize((nw, nh), Image.Resampling.LANCZOS)
                with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as tmp:
                    tmp_ppm_path = Path(tmp.name)
                resized_im.save(tmp_ppm_path, format="PPM")

        # Fast path for non-resized images: encode directly from source JPEG.
        input_for_encoder = tmp_ppm_path if tmp_ppm_path is not None else src
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_jpg:
            tmp_jpeg_path = Path(tmp_jpg.name)
        _run_cjpegli(input_for_encoder, tmp_jpeg_path, quality)

        if preserve_exif and exif_bytes:
            # Re-insert original EXIF into transcoded JPEG when present.
            try:
                piexif.insert(exif_bytes, str(tmp_jpeg_path))
            except Exception:
                pass

        encoded_bytes = tmp_jpeg_path.stat().st_size
        reduction_ratio = (src_bytes - encoded_bytes) / max(src_bytes, 1)
        min_reduction_ratio = min_reduction_pct / 100.0

        if reduction_ratio < min_reduction_ratio:
            # Keep source image if compression gain is too small (or negative).
            shutil.copy2(src, dst)
            dst_bytes = dst.stat().st_size
            return JobResult(
                True,
                "jpeg_kept_original",
                src_bytes,
                dst_bytes,
                False,
                False,
                f"reduction={reduction_ratio*100:.2f}% below threshold={min_reduction_pct:.2f}%",
            )

        shutil.move(str(tmp_jpeg_path), str(dst))
        shutil.copystat(src, dst, follow_symlinks=True)
        dst_bytes = dst.stat().st_size
        return JobResult(True, "jpeg", src_bytes, dst_bytes, resized, False)
    except Exception as e:
        # Fallback behavior: if transcode fails, copy original JPEG unchanged.
        try:
            shutil.copy2(src, dst)
            dst_bytes = dst.stat().st_size
            return JobResult(True, "jpeg_fallback_copy", src_bytes, dst_bytes, False, False, str(e))
        except Exception as copy_e:
            return JobResult(False, "jpeg", src_bytes, 0, resized, False, f"{e}; fallback copy failed: {copy_e}")
    finally:
        if tmp_ppm_path and tmp_ppm_path.exists():
            try:
                tmp_ppm_path.unlink()
            except OSError:
                pass
        if tmp_jpeg_path and tmp_jpeg_path.exists():
            try:
                tmp_jpeg_path.unlink()
            except OSError:
                pass


def run_job(
    job: Job,
    quality: int,
    max_side: int,
    min_reduction_pct: float,
    preserve_exif: bool,
    overwrite: bool,
) -> JobResult:
    if job.is_jpeg:
        return transcode_jpeg(
            src=job.src,
            dst=job.dst,
            quality=quality,
            max_side=max_side,
            min_reduction_pct=min_reduction_pct,
            preserve_exif=preserve_exif,
            overwrite=overwrite,
        )
    return copy_non_jpeg(job.src, job.dst, overwrite=overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror dataset tree while transcoding JPEGs with jpegli.")
    parser.add_argument("--src", type=Path, required=True, help="Source dataset root (e.g. /mldata/v10)")
    parser.add_argument("--dst", type=Path, required=True, help="Destination root (e.g. /mldata/v10-opt)")
    parser.add_argument("--quality", type=int, default=90, help="jpegli quality (default: 90)")
    parser.add_argument("--max-side", type=int, default=1280, help="Max long-edge for JPEG output (default: 1280)")
    parser.add_argument(
        "--min-reduction-pct",
        type=float,
        default=10.0,
        help="Keep original JPEG if size reduction is below this percent (default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 8))),
        help="Parallel worker count (default: min(8, cpu_count))",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination")
    parser.add_argument("--no-preserve-exif", action="store_true", help="Disable EXIF metadata preservation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()

    if shutil.which("cjpegli") is None:
        raise RuntimeError("cjpegli not found in PATH. Install jpegli tools first.")
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Source path is not a directory: {src}")
    if src == dst:
        raise ValueError("Source and destination must be different directories.")
    if dst.exists() and any(dst.iterdir()) and not args.overwrite:
        print(f"Destination exists and is non-empty: {dst}")
        print("Use --overwrite to replace existing files.")
        return 2

    jobs = iter_jobs(src, dst)
    jpeg_jobs = sum(1 for j in jobs if j.is_jpeg)
    other_jobs = len(jobs) - jpeg_jobs
    preserve_exif = not args.no_preserve_exif

    print(f"Source: {src}")
    print(f"Destination: {dst}")
    print(f"Files found: {len(jobs)} (JPEG: {jpeg_jobs}, other: {other_jobs})")
    print(
        f"Settings: quality={args.quality} max_side={args.max_side} min_reduction_pct={args.min_reduction_pct} "
        f"workers={args.workers} preserve_exif={preserve_exif} overwrite={args.overwrite}"
    )

    ok = 0
    failed = 0
    skipped = 0
    resized = 0
    fallback_copied = 0
    kept_original_low_gain = 0
    src_jpeg_total = 0
    dst_jpeg_total = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                run_job,
                job,
                args.quality,
                args.max_side,
                args.min_reduction_pct,
                preserve_exif,
                args.overwrite,
            ): job
            for job in jobs
        }
        with tqdm(total=len(futures), desc="Processing", unit="file") as pbar:
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    failed += 1
                    print(f"[ERROR] {job.src}: {e}")
                    pbar.update(1)
                    continue

                if not result.ok:
                    failed += 1
                    print(f"[ERROR] {job.src}: {result.error}")
                else:
                    ok += 1
                    if result.skipped:
                        skipped += 1
                    if result.action == "jpeg_fallback_copy":
                        fallback_copied += 1
                        print(f"[WARN] jpegli failed, copied original: {job.src} ({result.error})")
                    if result.action == "jpeg_kept_original":
                        kept_original_low_gain += 1
                    if result.action in {"jpeg", "jpeg_fallback_copy", "jpeg_kept_original"}:
                        src_jpeg_total += result.src_bytes
                        dst_jpeg_total += result.dst_bytes
                        if result.resized:
                            resized += 1
                pbar.update(1)

    print("\nDone.")
    print(f"Success: {ok}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"JPEG resized: {resized}")
    print(f"JPEG fallback-copied: {fallback_copied}")
    print(f"JPEG kept-original (low gain): {kept_original_low_gain}")
    if src_jpeg_total > 0:
        saved = src_jpeg_total - dst_jpeg_total
        pct = (saved / src_jpeg_total) * 100.0
        print(f"JPEG bytes: {src_jpeg_total} -> {dst_jpeg_total}")
        print(f"JPEG savings: {saved} bytes ({pct:.2f}%)")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
