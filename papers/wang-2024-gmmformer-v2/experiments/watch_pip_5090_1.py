"""Poll 5090_1 torch install. Print only DONE/FAILED."""
import subprocess
import sys
import time

LOG = "/data/zhaopu/wang-2024-gmmformer-v2/logs/pip_torch.log"
REMOTE = (
    "if grep -q '^DONE ' {log}; then echo STATE=DONE; "
    "elif pgrep -f install_torch_5090_1.sh >/dev/null; then echo STATE=RUN; "
    "else echo STATE=FAIL; tail -20 {log}; fi"
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
        time.sleep(30)
        continue
    if "STATE=DONE" in out:
        print("DONE: torch install finished")
        sys.exit(0)
    if "STATE=FAIL" in out or code != 0 and "STATE=RUN" not in out:
        snippet = " ".join(out.strip().split())
        print("FAILED: pip install exited without DONE: " + snippet[:500])
        sys.exit(1)
    time.sleep(30)
