#!/usr/bin/env python3
"""CPU ntrain check for T1-tvr I3D+ResNet+RoBERTa. Do not touch Charades tmp/h5."""
import os
import sys

root = "/data/zhaopu/wang-2024-gmmformer-v2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PRVR_SEED", "9527")
os.environ["PRVR_ROOT"] = os.path.join(root, "tmp", "tvr_ntrain_check")
src = os.path.join(root, "vendor", "GMMFormer_v2", "src")
sys.path.insert(0, src)
os.chdir(src)

from Configs.tvr import get_cfg_defaults
from Datasets.builder import get_datasets
from Datasets.data_provider import read_video_ids
from Utils.basic_utils import BigFile

cfg = get_cfg_defaults()
print("visual_feature", cfg["visual_feature"])
print("q_feat_size", cfg["q_feat_size"])
print("sub_feat_size", cfg["sub_feat_size"])
print("collection", cfg["collection"])
print("data_root", cfg["data_root"])
print("root", cfg["root"])
print("hidden_size", cfg["hidden_size"])
print("batchsize", cfg["batchsize"])
print("seed", cfg["seed"])
print("max_ctx_l", cfg["max_ctx_l"])
print("map_size", cfg["map_size"])
print("lr", cfg["lr"])
print("max_es_cnt", cfg["max_es_cnt"])
print("use_hard_negative", cfg["use_hard_negative"])

if cfg["visual_feature"] != "i3d_resnet":
    raise SystemExit("FAIL visual_feature=%s" % cfg["visual_feature"])
if int(cfg["q_feat_size"]) != 768:
    raise SystemExit("FAIL q_feat_size=%s" % cfg["q_feat_size"])
if cfg["collection"] != "tvr":
    raise SystemExit("FAIL collection=%s" % cfg["collection"])
if cfg["data_root"] != "/data/zhaopu/wang-2024-gmmformer-v2/data/prvr":
    raise SystemExit("FAIL data_root=%s" % cfg["data_root"])
if abs(float(cfg["lr"]) - 0.0003) > 1e-12:
    raise SystemExit("FAIL lr=%s" % cfg["lr"])
if int(cfg["batchsize"]) != 128:
    raise SystemExit("FAIL batchsize=%s" % cfg["batchsize"])
if int(cfg["seed"]) != 9527:
    raise SystemExit("FAIL seed=%s" % cfg["seed"])
if int(cfg["max_ctx_l"]) != 128 or int(cfg["map_size"]) != 32:
    raise SystemExit("FAIL clip/frame %s %s" % (cfg["map_size"], cfg["max_ctx_l"]))
if int(cfg["hidden_size"]) != 384:
    raise SystemExit("FAIL hidden_size=%s" % cfg["hidden_size"])

tvr_root = os.path.join(cfg["data_root"], "tvr")
vis_dir = os.path.join(tvr_root, "FeatureData", "i3d_resnet")
txt = os.path.join(tvr_root, "TextData", "roberta_tvr_query_feat.hdf5")
train_cap = os.path.join(tvr_root, "TextData", "tvrtrain.caption.txt")
val_cap = os.path.join(tvr_root, "TextData", "tvrval.caption.txt")
test_cap = os.path.join(tvr_root, "TextData", "tvrtest.caption.txt")
cha_i3d = os.path.join(cfg["data_root"], "charades", "FeatureData", "charades_i3d.h5")
print("vis_dir", vis_dir, os.path.isdir(vis_dir))
print("txt", txt, os.path.isfile(txt))
print("train_cap", train_cap, os.path.isfile(train_cap))
print("val_cap", val_cap, os.path.isfile(val_cap))
print("test_cap", test_cap, os.path.isfile(test_cap))
print("cha_i3d", cha_i3d, os.path.isfile(cha_i3d), os.path.getsize(cha_i3d) if os.path.isfile(cha_i3d) else 0)
for p in (vis_dir, txt, train_cap, val_cap, cha_i3d):
    if not os.path.exists(p):
        raise SystemExit("FAIL missing %s" % p)
for name in ("feature.bin", "id.txt", "shape.txt", "video2frames.txt"):
    fp = os.path.join(vis_dir, name)
    print("feat", name, os.path.isfile(fp), os.path.getsize(fp) if os.path.isfile(fp) else 0)
    if not os.path.isfile(fp):
        raise SystemExit("FAIL missing BigFile %s" % fp)

bf = BigFile(vis_dir)
print("bigfile", bf.nr_of_images, bf.ndims)
if int(bf.ndims) != 3072:
    raise SystemExit("FAIL i3d_resnet dim=%s" % bf.ndims)

comp = open(os.path.join(src, "Models", "gmmformerV2", "model_components.py"), encoding="utf-8").read()
if "out = torch.mean(oo, dim = -1).squeeze()" in comp:
    raise SystemExit("FAIL TC mean overwrite still present")
if "out = torch.sum(oo * weight" not in comp:
    raise SystemExit("FAIL TC weighted sum missing")

builder = open(os.path.join(src, "Datasets", "builder.py"), encoding="utf-8").read()
if "tvr_train_release.jsonl" in builder:
    raise SystemExit("FAIL jsonl path still present")

cfg, train_loader, context_loader, query_loader = get_datasets(cfg)
print("visual_feat_dim", cfg.get("visual_feat_dim"))
if int(cfg.get("visual_feat_dim", -1)) != 3072:
    raise SystemExit("FAIL visual_feat_dim=%s" % cfg.get("visual_feat_dim"))
ntrain = len(train_loader.dataset)
nctx = len(context_loader.dataset)
nq = len(query_loader.dataset)
print("ntrain", ntrain, "nctx", nctx, "nq", nq)
print("ntrain_videos_cap", len(read_video_ids(train_cap)))
print("nval_videos_cap", len(read_video_ids(val_cap)))
if ntrain <= 0 or nctx <= 0 or nq <= 0:
    raise SystemExit("FAIL empty loaders")
print("NTRAIN_CHECK_OK")
