#!/usr/bin/env python3
"""Extract ActivityNet tar into data/prvr/activitynet/. Strip netdisk/activitynet/.

Does not touch charades/ or tvr/.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path("/data/zhaopu/wang-2024-gmmformer-v2")
PRVR = ROOT / "data" / "prvr"
TAR = PRVR / "activitynet.tar"
DEST = PRVR / "activitynet"
CHA = PRVR / "charades"
TVR = PRVR / "tvr"
EXPECT_SIZE = 14895718400
NEED = [
    "FeatureData/i3d/feature.bin",
    "FeatureData/i3d/id.txt",
    "FeatureData/i3d/shape.txt",
    "FeatureData/i3d/video2frames.txt",
    "TextData/roberta_activitynet_query_feat.hdf5",
    "TextData/activitynettrain.caption.txt",
    "TextData/activitynetval.caption.txt",
    "TextData/activitynettest.caption.txt",
]


def already() -> bool:
    return all((DEST / rel).is_file() and (DEST / rel).stat().st_size > 0 for rel in NEED)


def main() -> int:
    if not CHA.is_dir():
        raise SystemExit("CHARADES_DIR_MISSING")
    if not TVR.is_dir():
        raise SystemExit("TVR_DIR_MISSING")
    if not TAR.is_file():
        raise SystemExit("TAR_MISSING")
    st = TAR.stat()
    sz = st.st_size
    blocks = getattr(st, "st_blocks", None)
    alloc = blocks * 512 if blocks is not None else None
    print("TAR", TAR, sz, "alloc", alloc, flush=True)
    if sz != EXPECT_SIZE:
        raise SystemExit("TAR_SIZE_FAIL got=%s want=%s" % (sz, EXPECT_SIZE))
    if alloc is not None and alloc < sz * 0.9:
        raise SystemExit("TAR_SPARSE alloc=%s size=%s" % (alloc, sz))
    cha_i3d = CHA / "FeatureData" / "charades_i3d.h5"
    if not cha_i3d.is_file():
        raise SystemExit("CHARADES_I3D_MISSING")
    tvr_bin = TVR / "FeatureData" / "i3d_resnet" / "feature.bin"
    if not tvr_bin.is_file():
        raise SystemExit("TVR_FEAT_MISSING")
    DEST.mkdir(parents=True, exist_ok=True)
    if already():
        print("EXTRACT_SKIP already", DEST, flush=True)
    else:
        print("TAR_X", TAR, "->", DEST, flush=True)
        subprocess.check_call(
            ["tar", "--strip-components=2", "-xf", str(TAR), "-C", str(DEST)]
        )
    for rel in NEED:
        p = DEST / rel
        print("NEED", p, p.is_file(), p.stat().st_size if p.is_file() else 0, flush=True)
        if not p.is_file() or p.stat().st_size <= 0:
            raise SystemExit("MISSING %s" % p)
    if not cha_i3d.is_file():
        raise SystemExit("CHARADES_I3D_GONE")
    if not tvr_bin.is_file():
        raise SystemExit("TVR_FEAT_GONE")
    print("CHARADES_KEEP", cha_i3d.stat().st_size, flush=True)
    print("TVR_KEEP", tvr_bin.stat().st_size, flush=True)
    print("EXTRACT_OK", DEST, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
