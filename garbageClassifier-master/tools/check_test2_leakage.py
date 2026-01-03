#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check whether images in test2 also appear in train/test.

Default behavior is exact-duplicate detection via SHA-256 over file bytes.
Optionally supports a simple perceptual hash (aHash) implemented with Pillow
(no extra deps) to catch re-encoded duplicates.

Examples (run from garbageClassifier-master):
  python tools/check_test2_leakage.py
  python tools/check_test2_leakage.py --mode sha256 --write-csv leakage.csv
  python tools/check_test2_leakage.py --mode ahash --ahash-threshold 0

Notes:
- This script only answers "does an image appear" by content similarity,
  not by filename.
- aHash can produce false positives; treat results as candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTS_DEFAULT = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


@dataclass(frozen=True)
class MatchResult:
    query_path: Path
    query_key: str
    matched_paths: Tuple[Path, ...]


def iter_image_files(root: Path, exts: Sequence[str]) -> Iterable[Path]:
    if not root.exists():
        return
    lower_exts = tuple(e.lower() for e in exts)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in lower_exts:
            yield p


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ahash_file(path: Path, size: int = 8) -> Optional[int]:
    """Average-hash (aHash) using Pillow only.

    Returns a 64-bit integer, or None if the image can't be opened.
    """

    try:
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((size, size))
            pixels = list(img.getdata())
    except Exception:
        return None

    mean_val = sum(pixels) / float(len(pixels))
    bits = 0
    for px in pixels:
        bits = (bits << 1) | (1 if px >= mean_val else 0)
    return bits


def hamming_distance_u64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def build_index(
    files: Iterable[Path],
    mode: str,
    ahash_size: int,
) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for p in files:
        if mode == "sha256":
            key = sha256_file(p)
        elif mode == "ahash":
            v = ahash_file(p, size=ahash_size)
            if v is None:
                continue
            key = f"{v:016x}"
        else:
            raise ValueError(f"Unknown mode: {mode}")
        index.setdefault(key, []).append(p)
    return index


def find_matches_exact(
    query_files: Iterable[Path],
    index: Dict[str, List[Path]],
    mode: str,
    ahash_size: int,
) -> List[MatchResult]:
    results: List[MatchResult] = []
    for q in query_files:
        if mode == "sha256":
            key = sha256_file(q)
        else:
            v = ahash_file(q, size=ahash_size)
            if v is None:
                continue
            key = f"{v:016x}"
        matched = tuple(index.get(key, []))
        if matched:
            results.append(MatchResult(query_path=q, query_key=key, matched_paths=matched))
    return results


def find_matches_ahash_threshold(
    query_files: Iterable[Path],
    ref_index: Dict[str, List[Path]],
    ahash_threshold: int,
    max_matches_per_query: int,
    ahash_size: int,
) -> List[MatchResult]:
    """Threshold aHash match by scanning all unique reference hashes.

    This is O(|Q|*|H|). Use only if dataset is not huge.
    """

    ref_items: List[Tuple[int, str, List[Path]]] = []
    for key, paths in ref_index.items():
        try:
            ref_items.append((int(key, 16), key, paths))
        except Exception:
            continue

    results: List[MatchResult] = []
    for q in query_files:
        qv = ahash_file(q, size=ahash_size)
        if qv is None:
            continue

        matched_paths: List[Path] = []
        for ref_v, _key, paths in ref_items:
            if hamming_distance_u64(qv, ref_v) <= ahash_threshold:
                matched_paths.extend(paths)
                if len(matched_paths) >= max_matches_per_query:
                    matched_paths = matched_paths[:max_matches_per_query]
                    break

        if matched_paths:
            results.append(
                MatchResult(query_path=q, query_key=f"{qv:016x}", matched_paths=tuple(matched_paths))
            )

    return results


def rel(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base))
    except Exception:
        return str(p)


def write_csv_report(
    out_path: Path,
    matches: List[MatchResult],
    base: Path,
    ref_base: Path,
    mode: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_path", "mode", "query_key", "matched_count", "matched_paths"])
        for m in matches:
            w.writerow(
                [
                    rel(m.query_path, base),
                    mode,
                    m.query_key,
                    len(m.matched_paths),
                    "|".join(rel(p, ref_base) for p in m.matched_paths),
                ]
            )


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]  # garbageClassifier-master
    default_data = default_root / "Garbage data"

    p = argparse.ArgumentParser(description="Check if test2 images appear in train/test (by content).")
    p.add_argument("--data-root", type=Path, default=default_data, help="Path to 'Garbage data' folder")
    p.add_argument("--train", type=str, default="train", help="Subfolder name under data-root")
    p.add_argument("--test", type=str, default="test", help="Subfolder name under data-root")
    p.add_argument("--test2", type=str, default="test2", help="Subfolder name under data-root")

    p.add_argument(
        "--mode",
        choices=("sha256", "ahash"),
        default="sha256",
        help="Hash mode: sha256 for exact duplicates; ahash for perceptual candidates",
    )
    p.add_argument("--ext", nargs="*", default=list(IMAGE_EXTS_DEFAULT), help="Image extensions to include")

    p.add_argument("--ahash-size", type=int, default=8, help="aHash grid size (default 8 => 64-bit)")
    p.add_argument(
        "--ahash-threshold",
        type=int,
        default=0,
        help="If >0 and mode=ahash: allow Hamming distance <= threshold (slow scan)",
    )
    p.add_argument(
        "--max-matches-per-query",
        type=int,
        default=20,
        help="When using threshold scan, cap matched paths per query",
    )

    p.add_argument("--write-csv", type=Path, default=None, help="Write matched pairs to CSV")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    data_root: Path = args.data_root
    train_root = data_root / args.train
    test_root = data_root / args.test
    test2_root = data_root / args.test2

    exts = tuple(args.ext)

    train_files = list(iter_image_files(train_root, exts))
    test_files = list(iter_image_files(test_root, exts))
    test2_files = list(iter_image_files(test2_root, exts))

    if not test2_root.exists():
        print(f"[ERROR] test2 folder not found: {test2_root}")
        return 2

    ref_files = train_files + test_files
    print(f"[INFO] data_root: {data_root}")
    print(f"[INFO] mode: {args.mode}")
    print(f"[INFO] train images: {len(train_files)}")
    print(f"[INFO] test  images: {len(test_files)}")
    print(f"[INFO] test2 images: {len(test2_files)}")

    if not ref_files:
        print("[ERROR] train+test contains 0 images; nothing to compare.")
        return 2

    print("[INFO] building reference index...")
    ref_index = build_index(ref_files, mode=args.mode, ahash_size=args.ahash_size)
    print(f"[INFO] unique reference keys: {len(ref_index)}")

    print("[INFO] matching test2...")
    if args.mode == "ahash" and int(args.ahash_threshold) > 0:
        matches = find_matches_ahash_threshold(
            test2_files,
            ref_index,
            ahash_threshold=int(args.ahash_threshold),
            max_matches_per_query=int(args.max_matches_per_query),
            ahash_size=int(args.ahash_size),
        )
    else:
        matches = find_matches_exact(test2_files, ref_index, mode=args.mode, ahash_size=int(args.ahash_size))

    matched_queries = len(matches)
    print(f"[RESULT] matched test2 images: {matched_queries}/{len(test2_files)}")

    if matched_queries:
        # Print a small sample for quick inspection
        preview_n = min(20, matched_queries)
        print(f"[RESULT] showing first {preview_n} matches:")
        for m in matches[:preview_n]:
            print(f"  - {rel(m.query_path, data_root)} -> {len(m.matched_paths)} hits")
            for hit in m.matched_paths[:3]:
                print(f"      * {rel(hit, data_root)}")
            if len(m.matched_paths) > 3:
                print("      * ...")

    if args.write_csv is not None:
        out_path: Path = args.write_csv
        print(f"[INFO] writing CSV: {out_path}")
        write_csv_report(out_path, matches, base=data_root, ref_base=data_root, mode=args.mode)

    # Exit code: 0 if no leakage, 1 if found matches
    return 1 if matched_queries else 0


if __name__ == "__main__":
    raise SystemExit(main())
