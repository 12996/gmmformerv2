"""Poll 5090_1 remaining pip. Print only DONE/FAILED."""
import subprocess
import sys
import time

LOG = "/data/zhaopu/wang-2024-gmmformer-v2/logs/pip_rest.log"
REMOTE = (
    "if grep -q '^DONE ' {log}; then echo STATE=DONE; "
    "elif pgrep -f install_rest_deps.sh >/dev/null; then echo STATE=RUN; "
    "else echo STATE=FAIL; tail -30 {log}; fi"
).format(log=LOG)


def ssh():
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "5090_1", REMOTE],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


while True:
    try:
        code, out = ssh()
    except subprocess.TimeoutExpired:
        time.sleep(20)
        continue
    if "STATE=DONE" in out:
        print("DONE: rest deps installed")
        sys.exit(0)
    if "STATE=FAIL" in out or (code != 0 and "STATE=RUN" not in out):
        snippet = " ".join(out.strip().split())
        print("FAILED: rest pip exited without DONE: " + snippet[:500])
        sys.exit(1)
    time.sleep(20)
