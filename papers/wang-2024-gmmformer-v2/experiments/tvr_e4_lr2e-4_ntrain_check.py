#!/usr/bin/env python3
"""CPU check for TVR E4 lr=2e-4: same I3D+CLIP as E4, only lr changes."""
import os
import sys

import h5py

root = "/data/zhaopu/wang-2024-gmmformer-v2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PRVR_SEED", "9527")
os.environ["PRVR_ROOT"] = os.path.join(
    root, "tmp", "tvr_e4_ntrain_check_i3d_cliptext_e4_lr2e-4"
)
src = os.path.join(root, "vendor", "GMMFormer_v2", "src")
sys.path.insert(0, src)
os.chdir(src)

from Configs.tvr import get_cfg_defaults

cfg = get_cfg_defaults()
print("visual_feature", cfg["visual_feature"])
print("q_feat_size", cfg["q_feat_size"])
print("lr", cfg["lr"])
print("n_epoch", cfg["n_epoch"])
print("sft_factor", cfg["sft_factor"])
print("batchsize", cfg["batchsize"])
print("hidden_size", cfg["hidden_size"])
print("map_size", cfg["map_size"])
print("max_ctx_l", cfg["max_ctx_l"])
print("clip_scale_w", cfg["clip_scale_w"], "frame_scale_w", cfg["frame_scale_w"])
print("hard_negative_start_epoch", cfg["hard_negative_start_epoch"])
print("root", cfg["root"])
print("data_root", cfg["data_root"])
if cfg["visual_feature"] != "i3d_resnet":
    raise SystemExit("FAIL visual_feature=%s" % cfg["visual_feature"])
if int(cfg["q_feat_size"]) != 512:
    raise SystemExit("FAIL q_feat_size=%s" % cfg["q_feat_size"])
if abs(float(cfg["lr"]) - 0.0002) > 1e-12:
    raise SystemExit("FAIL lr=%s want 0.0002" % cfg["lr"])
if int(cfg["n_epoch"]) != 100:
    raise SystemExit("FAIL n_epoch=%s" % cfg["n_epoch"])
if "i3d_cliptext_e4_lr2e-4" not in cfg["root"]:
    raise SystemExit("FAIL root not lr2e-4 %s" % cfg["root"])
if "tvr_i3d_cliptext_e4_seed9527" in cfg["root"] and "lr2e-4" not in cfg["root"]:
    raise SystemExit("FAIL would overwrite original E4")

old_e4 = os.path.join(root, "tmp", "tvr_i3d_cliptext_e4_seed9527")
if not os.path.isdir(old_e4):
    raise SystemExit("FAIL missing original E4 dir")

txt = os.path.join(cfg["data_root"], "tvr", "TextData", "tvr_clip_B32_proj.h5")
eot = os.path.join(cfg["data_root"], "tvr", "TextData", "clip_ViT_B_32_tvr_query_feat.hdf5")
print("proj", txt, os.path.isfile(txt), os.path.getsize(txt) if os.path.isfile(txt) else 0)
print("eot_unused", eot, os.path.isfile(eot))
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
if "clip_B32_proj.h5" not in builder:
    raise SystemExit("FAIL builder missing proj h5")
if "i3d_cliptext_e4" not in builder:
    raise SystemExit("FAIL builder missing e4 PRVR_ROOT branch")
if "clip_ViT_B_32_tvr_query_feat.hdf5" in builder:
    raise SystemExit("FAIL builder wired to EOT hdf5")

comp = open(
    os.path.join(src, "Models", "gmmformerV2", "model_components.py"), encoding="utf-8"
).read()
if "out = torch.mean(oo, dim = -1).squeeze()" in comp:
    raise SystemExit("FAIL TC mean overwrite present")
if "out = torch.sum(oo * weight" not in comp:
    raise SystemExit("FAIL TC weighted sum missing")

from Utils.basic_utils import BigFile

vis_dir = os.path.join(cfg["data_root"], "tvr", "FeatureData", "i3d_resnet")
bf = BigFile(vis_dir)
print("bigfile", bf.nr_of_images, bf.ndims)
if int(bf.ndims) != 3072:
    raise SystemExit("FAIL i3d_resnet dim=%s" % bf.ndims)

yp = os.path.join(cfg["model_root"], "hyperparams.yaml")
print("hyperparams", yp, os.path.isfile(yp))
if "--fast" in sys.argv:
    print("E4_LR2E4_FAST_CHECK_OK")
    raise SystemExit(0)

from Datasets.builder import get_datasets

cfg, train_loader, context_loader, query_loader = get_datasets(cfg)
print("visual_feat_dim", cfg.get("visual_feat_dim"))
if int(cfg.get("visual_feat_dim", -1)) != 3072:
    raise SystemExit("FAIL visual_feat_dim=%s" % cfg.get("visual_feat_dim"))
print("ntrain", len(train_loader.dataset), "nctx", len(context_loader.dataset), "nq", len(query_loader.dataset))
print("E4_LR2E4_NTRAIN_CHECK_OK")
