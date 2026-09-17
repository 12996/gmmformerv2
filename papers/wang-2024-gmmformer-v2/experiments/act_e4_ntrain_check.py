#!/usr/bin/env python3
"""CPU ntrain check for ActivityNet E4: I3D visual + CLIP projected token text."""
import os
import sys

import h5py

root = "/data/zhaopu/wang-2024-gmmformer-v2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PRVR_SEED", "9527")
os.environ["PRVR_ROOT"] = os.path.join(root, "tmp", "act_e4_ntrain_check_i3d_cliptext_e4")
src = os.path.join(root, "vendor", "GMMFormer_v2", "src")
sys.path.insert(0, src)
os.chdir(src)

from Configs.act import get_cfg_defaults
from Datasets.builder import get_datasets

cfg = get_cfg_defaults()
print("visual_feature", cfg["visual_feature"])
print("q_feat_size", cfg["q_feat_size"])
print("root", cfg["root"])
print("data_root", cfg["data_root"])
if cfg["visual_feature"] != "i3d":
    raise SystemExit("FAIL visual_feature=%s" % cfg["visual_feature"])
if int(cfg["q_feat_size"]) != 512:
    raise SystemExit("FAIL q_feat_size=%s" % cfg["q_feat_size"])
if "i3d_cliptext_e4" not in cfg["root"]:
    raise SystemExit("FAIL root not e4 %s" % cfg["root"])
if cfg["data_root"] != "/data/zhaopu/wang-2024-gmmformer-v2/data/prvr":
    raise SystemExit("FAIL data_root=%s" % cfg["data_root"])

txt = os.path.join(cfg["data_root"], "activitynet", "TextData", "activitynet_clip_B32_proj.h5")
print("proj", txt, os.path.isfile(txt), os.path.getsize(txt) if os.path.isfile(txt) else 0)
if not os.path.isfile(txt):
    raise SystemExit("FAIL missing CLIP proj h5")

with h5py.File(txt, "r") as h:
    keys = list(h.keys())[:20]
    ranks = {len(h[k].shape) for k in keys}
    dims = {h[k].shape[-1] for k in keys}
    print("sample", [(k, h[k].shape) for k in keys[:5]])
    print("ranks", ranks, "dims", dims)
    if ranks != {2}:
        raise SystemExit("FAIL CLIP text not token sequence ranks=%s" % ranks)
    if dims != {512}:
        raise SystemExit("FAIL CLIP dim %s" % dims)

builder = open(os.path.join(src, "Datasets", "builder.py"), encoding="utf-8").read()
act_idx = builder.find("collection == 'activitynet' and cfg['visual_feature'] == 'i3d'")
if act_idx < 0:
    raise SystemExit("FAIL builder missing activitynet i3d")
act_block = builder[act_idx : act_idx + 1100]
if "clip_B32_proj.h5" not in act_block:
    raise SystemExit("FAIL builder missing act proj h5")
if "clip_ViT_B_32" in act_block:
    raise SystemExit("FAIL builder wired to EOT hdf5")

ds = get_datasets(cfg)
if len(ds) == 6:
    cfg, train_loader, context_loader, query_loader, test_context_loader, test_query_loader = ds
    print("LOADERS 6")
else:
    cfg, train_loader, context_loader, query_loader = ds
    print("LOADERS 4")
print("visual_feat_dim", cfg.get("visual_feat_dim"))
if int(cfg.get("visual_feat_dim", -1)) != 1024:
    raise SystemExit("FAIL visual_feat_dim=%s" % cfg.get("visual_feat_dim"))
if int(cfg["q_feat_size"]) != 512:
    raise SystemExit("FAIL q_feat_size overwritten %s" % cfg["q_feat_size"])
print("ntrain", len(train_loader.dataset), "nctx", len(context_loader.dataset), "nq", len(query_loader.dataset))
ds0 = train_loader.dataset
print("text_feat_path", getattr(ds0, "text_feat_path", None))
if "activitynet_clip_B32_proj.h5" not in str(getattr(ds0, "text_feat_path", "")):
    raise SystemExit("FAIL text is not CLIP proj h5")
if "roberta_" in str(getattr(ds0, "text_feat_path", "")).lower():
    raise SystemExit("FAIL text still RoBERTa")
item = ds0[0]
print("item0_clip", tuple(item[0].shape), "item0_frame", tuple(item[1].shape), "qid_dim", tuple(item[2][0].shape))
if item[0].shape[-1] != 1024:
    raise SystemExit("FAIL vis dim %s" % (item[0].shape,))
if item[2][0].shape[-1] != 512:
    raise SystemExit("FAIL text dim %s" % (item[2][0].shape,))
print("E4_NTRAIN_CHECK_OK")
