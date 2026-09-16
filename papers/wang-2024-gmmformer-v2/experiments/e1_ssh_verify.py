#!/usr/bin/env python3
"""SSH helper for E1: verify patch and print GPU occupancy."""
import subprocess
import sys

REMOTE = r"""
set -e
chmod +x /data/zhaopu/wang-2024-gmmformer-v2/run_cha_tc_gpu.sh
echo '==== launcher PRVR_ROOT ===='
grep -n PRVR_ROOT /data/zhaopu/wang-2024-gmmformer-v2/run_cha_tc_gpu.sh
echo '==== grep mean overwrite patched ===='
if grep -n 'out = torch.mean(oo, dim = -1).squeeze()' /data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src/Models/gmmformerV2/model_components.py; then
  echo MEAN_STILL_PRESENT
else
  echo MEAN_OVERWRITE_GONE
fi
echo '==== grep sum remains ===='
grep -n 'out = torch.sum(oo \* weight' /data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src/Models/gmmformerV2/model_components.py
echo '==== backup still has mean ===='
grep -n 'out = torch.mean(oo, dim = -1).squeeze()' /data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src/Models/gmmformerV2/model_components.py.bak_e1_mean_20260916
echo '==== DyGMM tail ===='
sed -n '160,172p' /data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src/Models/gmmformerV2/model_components.py
echo '==== baseline dir untouched ===='
ls -ld /data/zhaopu/wang-2024-gmmformer-v2/tmp/i3d_seed9527
if [ ! -e /data/zhaopu/wang-2024-gmmformer-v2/tmp/i3d_tc_seed9527 ]; then
  echo tc_dir_not_created_yet
fi
echo '==== occupancy ===='
python3 /tmp/e1_gpu_occupancy.py
"""


def main() -> int:
    subprocess.check_call(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            r"F:\论文\papers\wang-2024-gmmformer-v2\experiments\e1_gpu_occupancy.py",
            "5090_1:/tmp/e1_gpu_occupancy.py",
        ]
    )
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "5090_1", REMOTE],
        check=False,
    )
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
