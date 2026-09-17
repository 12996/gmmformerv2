#!/usr/bin/env python3
"""Campus-LAN sequential SCP of complete tvr.zip, then unzip + train launch helpers."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HOST = "5090_1"
LOCAL = r"F:\论文\data\tvr\tvr.zip"
REMOTE_DIR = "/data/zhaopu/wang-2024-gmmformer-v2/data/prvr"
REMOTE = REMOTE_DIR + "/tvr.zip"
REMOTE_BAK = REMOTE_DIR + "/tvr.zip.sparse.bak"
EXPECT = 17787514489
ROOT = "/data/zhaopu/wang-2024-gmmformer-v2"


def ssh(cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=30",
            HOST,
            cmd,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def ssh_ok(cmd: str, timeout: int = 60) -> str:
    r = ssh(cmd, timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError("ssh fail rc=%s cmd=%s\n%s" % (r.returncode, cmd, out[-2000:]))
    return out


def backup_sparse() -> None:
    cmd = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "p = Path(%r)\n"
        "bak = Path(%r)\n"
        "print('exists', p.exists())\n"
        "if p.exists():\n"
        "    st = p.stat()\n"
        "    print('size', st.st_size)\n"
        "    print('phys', st.st_blocks * 512)\n"
        "    print('sparse', st.st_blocks * 512 < st.st_size)\n"
        "    if bak.exists():\n"
        "        bak.unlink()\n"
        "        print('removed_old_bak')\n"
        "    p.rename(bak)\n"
        "    print('moved_to_bak')\n"
        "else:\n"
        "    print('no_zip')\n"
        "parts = Path(%r)\n"
        "if parts.exists():\n"
        "    parts.unlink()\n"
        "    print('removed_parts')\n"
        "print('prvr')\n"
        "for x in sorted(Path(%r).iterdir()):\n"
        "    kind = 'DIR' if x.is_dir() else 'FILE'\n"
        "    print(kind, x.name, x.stat().st_size if x.is_file() else '')\n"
        "PY\n"
        % (REMOTE, REMOTE_BAK, REMOTE + ".parts.json", REMOTE_DIR)
    )
    print(ssh_ok(cmd, timeout=90), flush=True)


def remote_size() -> int:
    out = ssh_ok(
        "python3 -c \"from pathlib import Path; p=Path(%r); print(p.stat().st_size if p.exists() else 0)\""
        % REMOTE,
        timeout=30,
    )
    return int(out.strip().splitlines()[-1])


def scp_full() -> None:
    if not os.path.isfile(LOCAL):
        raise SystemExit("LOCAL_MISSING")
    sz = os.path.getsize(LOCAL)
    if sz != EXPECT:
        raise SystemExit("LOCAL_SIZE %s != %s" % (sz, EXPECT))
    print("SCP_START local=%s expect=%s" % (LOCAL, EXPECT), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=120",
            LOCAL,
            "%s:%s" % (HOST, REMOTE),
        ]
    )
    last = -1
    stall = 0
    while proc.poll() is None:
        time.sleep(10)
        try:
            cur = remote_size()
        except Exception as e:
            print("SIZE_ERR", e, flush=True)
            continue
        dt = max(time.time() - t0, 1e-6)
        mbps = cur / dt / 1024 / 1024
        pct = 100.0 * cur / EXPECT
        print(
            "SCP_PROG bytes=%s/%.0f%% MBps=%.2f elapsed=%.0fs pid=%s"
            % (cur, pct, mbps, dt, proc.pid),
            flush=True,
        )
        if cur == last:
            stall += 10
            if stall >= 300:
                print("SCP_STALL 300s", flush=True)
        else:
            stall = 0
            last = cur
    rc = proc.returncode
    print("SCP_RC", rc, "elapsed", round(time.time() - t0, 1), flush=True)
    if rc != 0:
        raise SystemExit("SCP_FAIL %s" % rc)


def verify_remote_zip() -> None:
    out = ssh_ok(
        "python3 - <<'PY'\n"
        "import zipfile\n"
        "from pathlib import Path\n"
        "p = Path(%r)\n"
        "print('size', p.stat().st_size)\n"
        "print('phys', p.stat().st_blocks * 512)\n"
        "ok_size = p.stat().st_size == %d\n"
        "print('SIZE_OK', ok_size)\n"
        "if not ok_size:\n"
        "    raise SystemExit('SIZE_FAIL')\n"
        "z = zipfile.ZipFile(str(p))\n"
        "print('NMEM', len(z.namelist()))\n"
        "names = z.namelist()\n"
        "need = [\n"
        "    'FeatureData/i3d_resnet/feature.bin',\n"
        "    'TextData/roberta_tvr_query_feat.hdf5',\n"
        "    'TextData/clip_ViT_B_32_tvr_query_feat.hdf5',\n"
        "]\n"
        "for n in names:\n"
        "    print('MEM', n, z.getinfo(n).file_size)\n"
        "for n in need:\n"
        "    hit = any(x.endswith(n) or x == n or x.endswith('/'+n) for x in names)\n"
        "    print('HAS', n, hit)\n"
        "    if not hit:\n"
        "        raise SystemExit('MISSING '+n)\n"
        "print('ZIPFILE_OK')\n"
        "PY\n"
        % (REMOTE, EXPECT),
        timeout=180,
    )
    print(out, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["backup", "scp", "verify", "all"])
    args = ap.parse_args()
    if args.action in ("backup", "all"):
        backup_sparse()
    if args.action in ("scp", "all"):
        scp_full()
        verify_remote_zip()
    if args.action == "verify":
        verify_remote_zip()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
