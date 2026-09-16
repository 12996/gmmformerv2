#!/usr/bin/env python3
from pathlib import Path
import importlib.util as u
import os
import subprocess

root = Path("/data/zhaopu/wang-2024-gmmformer-v2")
print("HOST", os.uname().nodename)
print("cv2", bool(u.find_spec("cv2")))
print("decord", bool(u.find_spec("decord")))
print("open_clip", bool(u.find_spec("open_clip")))
print("PIL", bool(u.find_spec("PIL")))
print("h5py", bool(u.find_spec("h5py")))
print("torch", bool(u.find_spec("torch")))
clip = root / ".cache/clip"
print("CLIP_DIR", clip, clip.exists())
if clip.exists():
    for p in sorted(clip.iterdir()):
        print("CLIP_PT", p.name, p.stat().st_size)
bd = (root / "vendor/GMMFormer_v2/src/Datasets/builder.py").read_text()
i = bd.find("elif cfg['visual_feature'] == 'clip':")
if i < 0:
    i = bd.find("if cfg['visual_feature'] == 'clip':")
print("==== BUILDER CLIP ====")
print(bd[i : i + 800])
print("==== CHA KEYS ====")
cha = (root / "vendor/GMMFormer_v2/src/Configs/cha.py").read_text()
for line in cha.splitlines():
    if any(k in line for k in ("visual_feature", "q_feat_size", "visual_feat_dim", "hidden_size", "batchsize", "seed", "max_ctx_l", "hard_negative", "max_es_cnt", "clip_scale", "frame_scale")):
        print(line)
comp = root / "vendor/GMMFormer_v2/src/Models/gmmformerV2/model_components.py"
txt = comp.read_text()
print("TC_SUM", "out = torch.sum(oo * weight.unsqueeze(2).repeat(1, 1, oo.shape[2], 1), dim=-1)" in txt)
print("TC_MEAN", "out = torch.mean(oo, dim = -1).squeeze()" in txt or "out = torch.mean(oo, dim=-1).squeeze()" in txt)
main_py = (root / "vendor/GMMFormer_v2/src/main.py").read_text()
print("MAIN_10P2", "10.2" in main_py)
print("==== FEAT ====")
for p in sorted((root / "data/prvr/charades/FeatureData").iterdir()):
    print(p.name, p.stat().st_size)
print("==== TEXT ====")
for p in sorted((root / "data/prvr/charades/TextData").iterdir()):
    print(p.name, p.stat().st_size)
print("==== TMP ====")
for p in sorted((root / "tmp").glob("*seed9527")):
    print(p)
print("==== GPU ====")
print(subprocess.check_output(["python3", str(root / "experiments/e2_gpu_occupancy.py")], text=True, errors="replace"))
