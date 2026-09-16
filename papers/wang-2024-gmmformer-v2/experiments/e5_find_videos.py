#!/usr/bin/env python3
"""Locate Charades videos by id. Do not download."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_5090_1 = "/data/zhaopu/wang-2024-gmmformer-v2"
EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov"}
SKIP_DIR_PARTS = {
    ".git",
    "anaconda3",
    "miniconda3",
    "conda",
    ".cache",
    "site-packages",
    "node_modules",
    ".vscode-server",
}


def sample_ids() -> list[str]:
    ids = []
    candidates = [
        Path(ROOT_5090_1) / "data/prvr/charades/TextData/charadestrain.caption.txt",
        Path("/home/zhaopu/wang-2024-gmmformer-v2/data/prvr/charades/TextData/charadestrain.caption.txt"),
    ]
    h5s = [
        Path(ROOT_5090_1) / "data/prvr/charades/FeatureData/charades_clip_L14.h5",
        Path("/home/zhaopu/wang-2024-gmmformer-v2/data/prvr/charades/FeatureData/charades_clip_L14.h5"),
    ]
    for cap in candidates:
        if not cap.is_file():
            continue
        with cap.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vid = line.split(" ", 1)[0].split("#")[0]
                if vid not in ids:
                    ids.append(vid)
                if len(ids) >= 8:
                    return ids
    for h5 in h5s:
        if not h5.is_file():
            continue
        import h5py

        with h5py.File(h5, "r") as h:
            return list(h.keys())[:8]
    return ids


def skip_dir(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return bool(parts & SKIP_DIR_PARTS)


def walk_roots(roots: list[str], ids: list[str]) -> None:
    print("HOST", os.uname().nodename if hasattr(os, "uname") else "unknown")
    print("CWD", os.getcwd())
    print("IDS", ids)
    idset = set(ids)
    found_by_id = {k: [] for k in ids}
    many_mp4_dirs = []
    archives = []
    charades_dirs = []
    n_mp4 = 0
    n_id_files = 0
    for root in roots:
        p = Path(root)
        print("ROOT", root, "exists", p.exists(), "is_dir", p.is_dir())
        if not p.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(p, followlinks=False):
            dp = Path(dirpath)
            if skip_dir(dp):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIR_PARTS]
            low = dirpath.lower()
            if "charade" in low:
                charades_dirs.append(dirpath)
            n_vid = 0
            for name in filenames:
                suf = Path(name).suffix.lower()
                stem = Path(name).stem
                if suf in EXTS:
                    n_mp4 += 1
                    n_vid += 1
                    if stem in idset:
                        found_by_id[stem].append(str(dp / name))
                        n_id_files += 1
                low_name = name.lower()
                if "charade" in low_name and suf in {".tar", ".gz", ".tgz", ".zip", ".7z"}:
                    archives.append(str(dp / name))
                if stem in idset and suf not in EXTS:
                    # frames/npz/etc
                    if n_id_files < 20:
                        print("ID_NONVIDEO", stem, dp / name)
            if n_vid >= 50:
                many_mp4_dirs.append((n_vid, dirpath))
    print("N_VIDEO_FILES", n_mp4)
    print("N_ID_VIDEO_HITS", n_id_files)
    print("CHARADES_DIRS")
    for d in charades_dirs[:40]:
        print(" ", d)
    print("ARCHIVES")
    for a in archives[:20]:
        print(" ", a)
    print("MANY_VIDEO_DIRS")
    for n, d in sorted(many_mp4_dirs, reverse=True)[:20]:
        print(" ", n, d)
    print("FOUND_BY_ID")
    any_hit = False
    for k, vs in found_by_id.items():
        print(" ", k, vs[:3], "n", len(vs))
        if vs:
            any_hit = True
    if any_hit:
        print("VIDEOS_FOUND")
    else:
        print("UNCERTAIN: no charades videos")


def main() -> None:
    roots = sys.argv[1:] or ["/data/zhaopu"]
    walk_roots(roots, sample_ids() or ["001YG", "00HFP", "00T4B", "01O27"])


if __name__ == "__main__":
    main()
