#!/bin/bash
set -euo pipefail
ROOT=/data/zhaopu/wang-2024-gmmformer-v2
TD=$ROOT/data/prvr/charades/TextData
FD=$ROOT/data/prvr/charades/FeatureData
PY=$ROOT/conda/bin/python
test -s "$TD/charades_clip_B32_proj.h5"
test -s "$TD/charades_clip_L14.h5"
test -s "$FD/charades_clip_L14.h5"
if [ ! -e "$TD/charades_clip_L14.h5.noproj.bak" ]; then
  cp -a "$TD/charades_clip_L14.h5" "$TD/charades_clip_L14.h5.noproj.bak"
  echo "WROTE_BAK $TD/charades_clip_L14.h5.noproj.bak"
else
  echo "KEEP_BAK $TD/charades_clip_L14.h5.noproj.bak"
fi
cp -a "$TD/charades_clip_B32_proj.h5" "$TD/charades_clip_L14.h5.new"
mv -f "$TD/charades_clip_L14.h5.new" "$TD/charades_clip_L14.h5"
"$PY" - <<'PY'
import os
import hashlib
import h5py

td = "/data/zhaopu/wang-2024-gmmformer-v2/data/prvr/charades/TextData"
fd = "/data/zhaopu/wang-2024-gmmformer-v2/data/prvr/charades/FeatureData"

def sha(p, n=1024 * 1024):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(n))
        f.seek(0, 2)
        return h.hexdigest()[:16], f.tell()

print("TEXT_L14", sha(os.path.join(td, "charades_clip_L14.h5")))
print("TEXT_PROJ", sha(os.path.join(td, "charades_clip_B32_proj.h5")))
print("TEXT_BAK", sha(os.path.join(td, "charades_clip_L14.h5.noproj.bak")))
print("VIS_L14", sha(os.path.join(fd, "charades_clip_L14.h5")))
with h5py.File(os.path.join(td, "charades_clip_L14.h5"), "r") as a, h5py.File(
    os.path.join(td, "charades_clip_B32_proj.h5"), "r"
) as b:
    k = list(a.keys())[0]
    print(
        "TEXT_KEY",
        k,
        "L14",
        a[k].shape,
        "PROJ",
        b[k].shape,
        "EQ",
        bool((a[k][()] == b[k][()]).all()),
    )
    print("NKEYS", len(a.keys()), len(b.keys()))
print("I3D_SEED", os.path.isdir("/data/zhaopu/wang-2024-gmmformer-v2/tmp/i3d_seed9527"))
print("I3D_TC", os.path.isdir("/data/zhaopu/wang-2024-gmmformer-v2/tmp/i3d_tc_seed9527"))
PY
ls -lh "$TD"/charades_clip_L14.h5 "$TD"/charades_clip_L14.h5.noproj.bak "$TD"/charades_clip_B32_proj.h5 "$FD"/charades_clip_L14.h5
echo INSTALL_PROJ_OK
