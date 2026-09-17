#!/usr/bin/env python3
"""Unzip MS-SL tvr.zip or activitynet.zip into data/prvr/<name>. Refuse Charades."""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path("/data/zhaopu/wang-2024-gmmformer-v2")
PRVR = ROOT / "data" / "prvr"
CHA = PRVR / "charades"


def refuse_charades(names):
    bad = [n for n in names if "charades" in n.replace("\\", "/").lower()]
    if bad:
        raise SystemExit("REFUSE_CHARADES %s" % bad[:8])


def find_root(staging: Path, feat_dir: str) -> Path:
    hits = []
    for p in [staging] + list(staging.rglob("*")):
        if not p.is_dir():
            continue
        if (p / "FeatureData" / feat_dir).is_dir() or (p / feat_dir).is_dir():
            hits.append(p)
        if (p / "TextData").is_dir() and (p / "FeatureData").is_dir():
            hits.append(p)
    if not hits:
        raise SystemExit("EXTRACT_LAYOUT_UNKNOWN %s" % staging)
    for p in hits:
        if (p / "FeatureData" / feat_dir).is_dir() and (p / "TextData").is_dir():
            return p
    for p in hits:
        if (p / "FeatureData").is_dir() and (p / "TextData").is_dir():
            return p
    return hits[0]


def merge_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                merge_tree(child, target)
            else:
                shutil.move(str(child), str(target))
        elif not target.exists():
            shutil.move(str(child), str(target))


def unzip_one(zip_path: Path, dest: Path, feat_dir: str, need_files) -> None:
    if not CHA.is_dir():
        raise SystemExit("CHARADES_DIR_MISSING")
    dest.mkdir(parents=True, exist_ok=True)
    if all((dest / rel).is_file() and (dest / rel).stat().st_size > 0 for rel in need_files):
        print("UNZIP_SKIP already", dest)
        return
    print("OPEN", zip_path, zip_path.stat().st_size)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        print("ZIP_NFILES", len(names), "head", names[:12])
        refuse_charades(names)
        staging = dest.parent / (dest.name + "_unzip_staging")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        z.extractall(staging)
    src = find_root(staging, feat_dir)
    print("SRC", src)
    if (src / feat_dir).is_dir() and not (src / "FeatureData").is_dir():
        (dest / "FeatureData").mkdir(parents=True, exist_ok=True)
        shutil.move(str(src / feat_dir), str(dest / "FeatureData" / feat_dir))
    if (src / "FeatureData").is_dir():
        merge_tree(src / "FeatureData", dest / "FeatureData")
    if (src / "TextData").is_dir():
        merge_tree(src / "TextData", dest / "TextData")
    shutil.rmtree(staging, ignore_errors=True)
    for rel in need_files:
        p = dest / rel
        print("NEED", p, p.is_file(), p.stat().st_size if p.is_file() else 0)
        if not p.is_file() or p.stat().st_size <= 0:
            raise SystemExit("MISSING %s" % p)
    cha_i3d = CHA / "FeatureData" / "charades_i3d.h5"
    if not cha_i3d.is_file():
        raise SystemExit("CHARADES_I3D_GONE")
    print("CHARADES_KEEP", cha_i3d.stat().st_size)
    print("UNZIP_OK", dest)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=("tvr", "activitynet"))
    args = p.parse_args()
    if args.which == "tvr":
        unzip_one(
            PRVR / "tvr.zip",
            PRVR / "tvr",
            "i3d_resnet",
            [
                "FeatureData/i3d_resnet/feature.bin",
                "TextData/roberta_tvr_query_feat.hdf5",
            ],
        )
    else:
        zips = [
            ROOT / "data" / "ms-sl" / "activitynet.zip",
            PRVR / "activitynet.zip",
        ]
        zp = next((z for z in zips if z.is_file() and z.stat().st_size > 0), None)
        if zp is None:
            raise SystemExit("NO_ACT_ZIP")
        unzip_one(
            zp,
            PRVR / "activitynet",
            "i3d",
            [
                "FeatureData/i3d/feature.bin",
                "TextData/roberta_activitynet_query_feat.hdf5",
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
