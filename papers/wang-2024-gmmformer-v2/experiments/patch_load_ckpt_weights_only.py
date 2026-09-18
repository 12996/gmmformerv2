#!/usr/bin/env python3
"""torch 2.6+ defaults weights_only=True; resume ckpt stores WarmupLinearSchedule."""
from pathlib import Path

p = Path("/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src/Utils/utils.py")
t = p.read_text()
old = 'ckpt = torch.load(ckpt_file, map_location="cpu")'
new = 'ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)'
if new in t and old not in t:
    print("ALREADY")
elif t.count(old) != 1:
    raise SystemExit("load_ckpt pattern count=%s" % t.count(old))
else:
    bak = p.with_name("utils.py.bak_resume200")
    if not bak.exists():
        bak.write_text(t)
        print("BACKUP", bak)
    p.write_text(t.replace(old, new, 1))
    print("PATCHED")
