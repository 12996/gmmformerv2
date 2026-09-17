#!/usr/bin/env python3
"""Wait for local TVR/Act zips, verify, unzip on 5090_1, launch I3D then E4.

Does not start a second TVR download. Never GPU2. Does not touch Charades tmp.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

HOST = "5090_1"
ROOT = "/data/zhaopu/wang-2024-gmmformer-v2"
HERE = Path(__file__).resolve().parent
LOCAL_TVR = Path(r"F:\论文\data\tvr\tvr.zip")
LOCAL_ACT = Path(r"F:\论文\data\act\activitynet.zip")
TVR_EXPECT = 17787514489
ACT_EXPECT = 18527330684
TVR_CHUNKS = (TVR_EXPECT + 8 * 1024 * 1024 - 1) // (8 * 1024 * 1024)
SEED = "9527"
STATE = HERE / "tvr_act_watch_state.json"

SCP_FILES = [
    "act_5090_1.py",
    "tvr_5090_1.py",
    "act_e4_5090_1.py",
    "tvr_e4_5090_1.py",
    "patch_act_i3d_loader.py",
    "patch_tvr_jsonl.py",
    "run_act_gpu.sh",
    "run_tvr_gpu.sh",
    "run_act_e4_gpu.sh",
    "run_tvr_e4_gpu.sh",
    "t1_act_do_setup.sh",
    "t1_act_ntrain_check.py",
    "e2_gpu_occupancy.py",
    "unzip_ms_sl.py",
    "extract_clip_text_proj.py",
    "use_local_7897.sh",
]


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%S "), msg, sep="", flush=True)


def proc_running(needle: str) -> bool:
    r = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*%s*' } | Select-Object -ExpandProperty ProcessId"
            % needle.replace("'", ""),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return bool((r.stdout or "").strip())


def ensure_local_jobs() -> None:
    if parts_done(LOCAL_TVR) < TVR_CHUNKS and not proc_running("tvr_parallel_download.py"):
        log("TVR_DOWNLOAD_DEAD resume one")
        env = os.environ.copy()
        env["HTTPS_PROXY"] = "http://127.0.0.1:7897"
        env["HTTP_PROXY"] = env["HTTPS_PROXY"]
        env["https_proxy"] = env["HTTPS_PROXY"]
        env["http_proxy"] = env["HTTPS_PROXY"]
        subprocess.Popen(
            [os.environ.get("PYTHON", "python"), "-u", str(HERE / "tvr_parallel_download.py"), str(LOCAL_TVR)],
            env=env,
            stdout=open(HERE / "tvr_download_resume.log", "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
    if sent_done(LOCAL_TVR) < TVR_CHUNKS and not proc_running("tvr_pipe_upload.py"):
        log("TVR_UPLOAD_DEAD start one")
        subprocess.Popen(
            [os.environ.get("PYTHON", "python"), "-u", str(HERE / "tvr_pipe_upload.py")],
            stdout=open(HERE / "tvr_upload_resume.log", "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
    act_chunks = (ACT_EXPECT + 8 * 1024 * 1024 - 1) // (8 * 1024 * 1024)
    if parts_done(LOCAL_TVR) >= TVR_CHUNKS and parts_done(LOCAL_ACT) < act_chunks and not proc_running("gdrive_parallel_download.py"):
        log("ACT_DOWNLOAD_START after TVR")
        env = os.environ.copy()
        env["HTTPS_PROXY"] = "http://127.0.0.1:7897"
        env["HTTP_PROXY"] = env["HTTPS_PROXY"]
        env["https_proxy"] = env["HTTPS_PROXY"]
        env["http_proxy"] = env["HTTPS_PROXY"]
        LOCAL_ACT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(
            [
                os.environ.get("PYTHON", "python"),
                "-u",
                str(HERE / "gdrive_parallel_download.py"),
                "--id",
                "1frCLujoWK1Aj0xGWfMO0U1qQLUrLvXgy",
                "--out",
                str(LOCAL_ACT),
                "--expect",
                str(ACT_EXPECT),
                "--nwork",
                "8",
            ],
            env=env,
            stdout=open(HERE / "act_download_resume.log", "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )


def load_state() -> dict:
    if STATE.is_file():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(st: dict) -> None:
    tmp = str(STATE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE)


def json_done(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return len(json.loads(path.read_text(encoding="utf-8")).get("done", []))
    except Exception:
        return 0


def parts_done(zip_path: Path) -> int:
    return json_done(Path(str(zip_path) + ".parts.json"))


def sent_done(zip_path: Path) -> int:
    return json_done(Path(str(zip_path) + ".sent.json"))


def run(cmd, timeout=60):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode, out


def ssh(cmd: str, timeout: int = 120) -> tuple[int, str]:
    return run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, cmd],
        timeout=timeout,
    )


def scp_file(name: str) -> None:
    src = HERE / name
    dst = "%s:%s/experiments/%s" % (HOST, ROOT, name)
    rc, out = run(
        ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", str(src), dst],
        timeout=90,
    )
    if rc != 0:
        raise RuntimeError("scp fail %s %s" % (name, out[-400:]))


def local_zip_ok(path: Path, expect: int) -> bool:
    if not path.is_file() or path.stat().st_size != expect:
        return False
    with open(path, "rb") as f:
        if f.read(2) != b"PK":
            return False
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if not names:
            return False
        log("ZIP_OK %s nfiles=%s head=%s" % (path, len(names), names[:8]))
    return True


def remote_zip_ok(path: str, expect: int) -> bool:
    py = (
        "import os; p=%r; print('SIZE', os.path.getsize(p) if os.path.isfile(p) else -1); "
        "print('HEAD', open(p,'rb').read(2) if os.path.isfile(p) else b'')"
        % path
    )
    rc, out = ssh("python3 -c %s" % json.dumps(py), timeout=40)
    log("REMOTE_ZIP %s" % out.strip().replace("\n", " | "))
    return rc == 0 and ("SIZE %s" % expect) in out and "HEAD b'PK" in out


def pick_gpu() -> str:
    rc, out = ssh("python3 %s/experiments/e2_gpu_occupancy.py" % ROOT, timeout=40)
    log("OCCUPANCY " + out.replace("\n", " || ")[:1500])
    py = r"""
import subprocess
q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'], text=True)
cands=[]
for line in q.splitlines():
    idx,used,total,util=[x.strip() for x in line.split(',')]
    if idx=='2':
        continue
    leftover=float(total)-float(used)
    print('LEFTOVER', idx, leftover)
    if leftover>=8000:
        cands.append((leftover, -float(util), idx))
cands.sort(reverse=True)
print('PICK_GPU', cands[0][2] if cands else 'none')
"""
    rc, out = ssh("python3 -c %s" % json.dumps(py), timeout=40)
    log(out.strip())
    pick = "none"
    for line in out.splitlines():
        if line.startswith("PICK_GPU "):
            pick = line.split()[1].strip()
    return pick


def launch(kind: str, gpu: str) -> None:
    scripts = {
        "tvr_i3d": ("run_tvr_gpu.sh", "tvr_i3d_seed%s_gpu%s" % (SEED, gpu), "main.py -d tvr"),
        "act_i3d": ("run_act_gpu.sh", "act_i3d_seed%s_gpu%s" % (SEED, gpu), "main.py -d act"),
        "tvr_e4": ("run_tvr_e4_gpu.sh", "tvr_e4_seed%s_gpu%s" % (SEED, gpu), "main.py -d tvr"),
        "act_e4": ("run_act_e4_gpu.sh", "act_e4_seed%s_gpu%s" % (SEED, gpu), "main.py -d act"),
    }
    sh, rundir, _ps = scripts[kind]
    remote = r"""
set -euo pipefail
ROOT=%s
GPU=%s
SEED=%s
SH="$ROOT/experiments/%s"
RUN="$ROOT/logs/runs/%s"
mkdir -p "$RUN" "$ROOT/logs"
chmod +x "$SH"
nohup "$SH" "$GPU" "$SEED" > "$RUN/launcher.out" 2>&1 &
echo LAUNCHER_PID=$!
sleep 10
echo ====pid====
cat "$RUN/pid" || echo NO_PID
echo ====runjson====
cat "$RUN/run.json" || echo NO_RUNJSON
echo ====ps====
ps -u zhaopu -o pid,cmd | grep -E 'main.py -d (act|tvr)' | grep -v grep || echo NO_PS
echo ====logtxt====
ls -l "$ROOT/tmp/"*"seed${SEED}"/results/*/gmmformer_v2/log.txt 2>/dev/null || true
""" % (ROOT, gpu, SEED, sh, rundir)
    rc, out = ssh(remote, timeout=40)
    log("LAUNCH %s gpu=%s\n%s" % (kind, gpu, out[-2000:]))
    if rc != 0 or "NO_PID" in out or "NO_PS" in out:
        raise RuntimeError("LAUNCH_FAIL %s" % kind)


def setup_tvr() -> None:
    cmd = r"""
set -euo pipefail
ROOT=%s
cp -a $ROOT/experiments/tvr_5090_1.py $ROOT/vendor/GMMFormer_v2/src/Configs/tvr.py
cp -a $ROOT/experiments/run_tvr_gpu.sh $ROOT/run_tvr_gpu.sh
cp -a $ROOT/experiments/run_tvr_e4_gpu.sh $ROOT/run_tvr_e4_gpu.sh
chmod +x $ROOT/run_tvr_gpu.sh $ROOT/run_tvr_e4_gpu.sh $ROOT/experiments/run_tvr_gpu.sh $ROOT/experiments/run_tvr_e4_gpu.sh
$ROOT/conda/bin/python $ROOT/experiments/patch_tvr_jsonl.py
$ROOT/conda/bin/python $ROOT/experiments/patch_act_i3d_loader.py
grep -n 'out = torch.mean(oo, dim = -1).squeeze()' $ROOT/vendor/GMMFormer_v2/src/Models/gmmformerV2/model_components.py && echo MEAN_STILL && exit 1 || true
grep -n 'out = torch.sum(oo \* weight' $ROOT/vendor/GMMFormer_v2/src/Models/gmmformerV2/model_components.py
ls -ld $ROOT/tmp/i3d_tc_seed9527 $ROOT/tmp/clip_e2_seed9527 $ROOT/tmp/clip_roberta_e3_seed9527 $ROOT/tmp/i3d_cliptext_e4_seed9527
echo TVR_SETUP_OK
""" % ROOT
    rc, out = ssh(cmd, timeout=180)
    log(out[-2500:])
    if rc != 0 or "TVR_SETUP_OK" not in out:
        raise RuntimeError("TVR_SETUP_FAIL")


def setup_act() -> None:
    rc, out = ssh("bash %s/experiments/t1_act_do_setup.sh" % ROOT, timeout=600)
    log(out[-3000:])
    if rc != 0 or "NTRAIN_CHECK_OK" not in out:
        raise RuntimeError("ACT_SETUP_FAIL")


def extract_clip(collection: str) -> None:
    dest = "%s/data/prvr/%s/TextData/%s_clip_L14.h5" % (ROOT, collection, collection)
    cmd = r"""
set -euo pipefail
ROOT=%s
COL=%s
DST=%s
if [ -f "$DST" ] && [ -s "$DST" ]; then echo CLIP_ALREADY "$DST"; ls -lh "$DST"; exit 0; fi
CAPS=""
for sp in train val test; do
  f="$ROOT/data/prvr/$COL/TextData/${COL}${sp}.caption.txt"
  if [ -f "$f" ]; then CAPS="$CAPS $f"; fi
done
if [ -z "$CAPS" ]; then echo NO_CAPTIONS; exit 1; fi
export CLIP_PT="$ROOT/.cache/clip/ViT-B-32.pt"
export CUDA_VISIBLE_DEVICES=
"$ROOT/conda/bin/python" -u "$ROOT/experiments/extract_clip_text_proj.py" --captions $CAPS --dst "$DST" --device cpu
ls -lh "$DST"
echo CLIP_OK "$DST"
""" % (ROOT, collection, dest)
    rc, out = ssh(cmd, timeout=7200)
    log(out[-2000:])
    if rc != 0 or ("CLIP_OK" not in out and "CLIP_ALREADY" not in out):
        raise RuntimeError("CLIP_EXTRACT_FAIL %s" % collection)


def scp_all() -> None:
    ssh("mkdir -p %s/experiments %s/logs" % (ROOT, ROOT), timeout=30)
    for name in SCP_FILES:
        log("SCP " + name)
        scp_file(name)


def scp_act_zip() -> None:
    log("SCP activitynet.zip start")
    rc, out = run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            str(LOCAL_ACT),
            "%s:%s/data/ms-sl/activitynet.zip" % (HOST, ROOT),
        ],
        timeout=7200,
    )
    log(out[-500:])
    if rc != 0:
        raise RuntimeError("SCP_ACT_FAIL")


def main() -> int:
    st = load_state()
    log("WATCH_START state=%s" % st)
    while True:
        ensure_local_jobs()
        tvr_done = parts_done(LOCAL_TVR)
        act_done = parts_done(LOCAL_ACT)
        sent = sent_done(LOCAL_TVR)
        log("PROGRESS tvr_local=%s/%s sent=%s act_local=%s" % (tvr_done, TVR_CHUNKS, sent, act_done))

        if tvr_done >= TVR_CHUNKS and not st.get("tvr_local_ok"):
            if local_zip_ok(LOCAL_TVR, TVR_EXPECT):
                st["tvr_local_ok"] = True
                save_state(st)
            else:
                log("TVR_LOCAL_ZIP_BAD")

        if act_done >= ((ACT_EXPECT + 8 * 1024 * 1024 - 1) // (8 * 1024 * 1024)) and not st.get("act_local_ok"):
            if local_zip_ok(LOCAL_ACT, ACT_EXPECT):
                st["act_local_ok"] = True
                save_state(st)
            else:
                log("ACT_LOCAL_ZIP_BAD")

        if st.get("tvr_local_ok") and sent >= TVR_CHUNKS and not st.get("tvr_remote_ok"):
            if remote_zip_ok("%s/data/prvr/tvr.zip" % ROOT, TVR_EXPECT):
                st["tvr_remote_ok"] = True
                save_state(st)
            else:
                log("TVR_REMOTE_WAIT")

        if st.get("tvr_remote_ok") and not st.get("tvr_unzip_ok"):
            scp_all()
            rc, out = ssh("python3 -u %s/experiments/unzip_ms_sl.py tvr" % ROOT, timeout=7200)
            log(out[-2000:])
            if rc == 0 and "UNZIP_OK" in out:
                st["tvr_unzip_ok"] = True
                save_state(st)
            else:
                log("TVR_UNZIP_FAIL")
                return 1

        if st.get("tvr_unzip_ok") and not st.get("tvr_i3d_launched"):
            scp_all()
            setup_tvr()
            gpu = pick_gpu()
            if gpu in (None, "none", "2"):
                log("BLOCKED no leftover GPU for I3D TVR")
            else:
                launch("tvr_i3d", gpu)
                st["tvr_i3d_launched"] = True
                st["tvr_i3d_gpu"] = gpu
                save_state(st)

        if st.get("act_local_ok") and not st.get("act_remote_ok"):
            ssh("mkdir -p %s/data/ms-sl" % ROOT, timeout=30)
            scp_act_zip()
            if remote_zip_ok("%s/data/ms-sl/activitynet.zip" % ROOT, ACT_EXPECT):
                st["act_remote_ok"] = True
                save_state(st)

        if st.get("act_remote_ok") and not st.get("act_unzip_ok"):
            scp_all()
            rc, out = ssh("python3 -u %s/experiments/unzip_ms_sl.py activitynet" % ROOT, timeout=7200)
            log(out[-2000:])
            if rc == 0 and "UNZIP_OK" in out:
                st["act_unzip_ok"] = True
                save_state(st)
            else:
                log("ACT_UNZIP_FAIL")
                return 1

        if st.get("act_unzip_ok") and not st.get("act_i3d_launched"):
            scp_all()
            try:
                setup_act()
            except Exception as e:
                log("ACT_SETUP_ERR %s" % e)
                return 1
            gpu = pick_gpu()
            if gpu in (None, "none", "2"):
                log("BLOCKED no leftover GPU for I3D Act")
            else:
                launch("act_i3d", gpu)
                st["act_i3d_launched"] = True
                st["act_i3d_gpu"] = gpu
                save_state(st)

        if st.get("tvr_unzip_ok") and not st.get("tvr_clip"):
            try:
                extract_clip("tvr")
                st["tvr_clip"] = True
                save_state(st)
            except Exception as e:
                log("TVR_CLIP_ERR %s" % e)

        if st.get("act_unzip_ok") and not st.get("act_clip"):
            try:
                extract_clip("activitynet")
                st["act_clip"] = True
                save_state(st)
            except Exception as e:
                log("ACT_CLIP_ERR %s" % e)

        if st.get("tvr_clip") and not st.get("tvr_e4_launched"):
            gpu = pick_gpu()
            if gpu in (None, "none", "2"):
                log("WAIT GPU for E4 TVR")
            else:
                launch("tvr_e4", gpu)
                st["tvr_e4_launched"] = True
                st["tvr_e4_gpu"] = gpu
                save_state(st)

        if st.get("act_clip") and not st.get("act_e4_launched"):
            gpu = pick_gpu()
            if gpu in (None, "none", "2"):
                log("WAIT GPU for E4 Act")
            else:
                launch("act_e4", gpu)
                st["act_e4_launched"] = True
                st["act_e4_gpu"] = gpu
                save_state(st)

        if (
            st.get("tvr_i3d_launched")
            and st.get("act_i3d_launched")
            and st.get("tvr_e4_launched")
            and st.get("act_e4_launched")
        ):
            log("ALL_LAUNCHED %s" % st)
            return 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
