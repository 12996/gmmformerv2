#!/usr/bin/env python3
"""Read-only inspect for ActivityNet setup. Do not mutate Charades/TVR."""
from pathlib import Path
import os
import subprocess

ROOT = Path("/data/zhaopu/wang-2024-gmmformer-v2")


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="")
    return r.returncode, r.stdout + r.stderr


print("==== TAR ====")
tar = ROOT / "data/prvr/activitynet.tar"
if tar.is_file():
    print("TAR", tar.stat().st_size)
else:
    print("TAR_MISSING")

print("==== ACT DIR ====")
act = ROOT / "data/prvr/activitynet"
print("ACT_DIR", act.is_dir())
if act.is_dir():
    for p in sorted(act.rglob("*")):
        if p.is_file():
            print("F", p.relative_to(act), p.stat().st_size)

print("==== MAIN ====")
main = ROOT / "vendor/GMMFormer_v2/src/main.py"
print("MAIN", main.is_file(), main.stat().st_size if main.is_file() else 0)
print("HAS_DS6", "len(_ds) == 6" in main.read_text() if main.is_file() else False)

print("==== ACT.PY ====")
actpy = ROOT / "vendor/GMMFormer_v2/src/Configs/act.py"
print(actpy.read_text() if actpy.is_file() else "NO_ACT_PY")

print("==== BUILDER ====")
bd = ROOT / "vendor/GMMFormer_v2/src/Datasets/builder.py"
t = bd.read_text()
print("HAS_ACT_I3D", "collection == 'activitynet' and cfg['visual_feature'] == 'i3d'" in t)
print("HAS_CLIP_B32", "clip_B32_proj.h5" in t)
print("HAS_CLIP_L14", "%s_clip_L14.h5" in t)
print("HAS_E4_ENV", "i3d_cliptext_e4" in t)
print("HAS_JSONL", "tvr_train_release.jsonl" in t)
print("HAS_EVAL_SPLIT", "PRVR_EVAL_SPLIT" in t)
print("HAS_TEST_LOADER", "test_query_eval_loader" in t)
idx = t.find("if cfg['visual_feature'] in ('i3d'")
print("---- i3d branch ----")
print(t[idx : idx + 1000] if idx >= 0 else "NO")
idx2 = t.find("visual_feature'] == 'i3d_resnet'")
if idx2 < 0:
    idx2 = t.find("i3d_resnet")
print("---- i3d_resnet around ----")
print(t[max(0, idx2 - 500) : idx2 + 800] if idx2 >= 0 else "NO")

print("==== TC ====")
comp = (ROOT / "vendor/GMMFormer_v2/src/Models/gmmformerV2/model_components.py").read_text()
print("HAS_MEAN", "out = torch.mean(oo, dim = -1).squeeze()" in comp)
print("HAS_WSUM", "out = torch.sum(oo * weight" in comp)

print("==== CLIP PT ====")
pt = ROOT / ".cache/clip/ViT-B-32.pt"
print("CLIP_PT", pt.is_file(), pt.stat().st_size if pt.is_file() else 0)

print("==== TMP ====")
sh("ls -ld %s/tmp/*seed9527 2>/dev/null || true" % ROOT)
print("==== CHA KEYS ====")
cha = (ROOT / "vendor/GMMFormer_v2/src/Configs/cha.py").read_text()
for line in cha.splitlines():
    if any(k in line for k in ("visual_feature", "q_feat_size", "visual_feat_dim", "data_root")):
        print(line)
print("==== TVR KEYS ====")
tvr = (ROOT / "vendor/GMMFormer_v2/src/Configs/tvr.py").read_text()
for line in tvr.splitlines():
    if any(k in line for k in ("visual_feature", "q_feat_size", "visual_feat_dim", "data_root")):
        print(line)
print("==== TVR PS ====")
sh("ps -p 564493,613385 -o pid,etime,cmd")
print("INSPECT_OK")
