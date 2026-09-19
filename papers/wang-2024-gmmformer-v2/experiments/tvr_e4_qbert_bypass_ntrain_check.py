#!/usr/bin/env python3
"""CPU check: parent E4 hparams, CLIP token sequences T>1, query Bert bypass."""
import os
import sys

import h5py
import torch

root = "/data/zhaopu/wang-2024-gmmformer-v2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PRVR_SEED", "9527")
os.environ["PRVR_ROOT"] = os.path.join(root, "tmp", "tvr_e4_ntrain_check_qbert_bypass")
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
print("max_es_cnt", cfg["max_es_cnt"])
print("loss_factor", cfg["loss_factor"])
print("neg_factor", cfg["neg_factor"])
print("margin", cfg["margin"])
print("drop", cfg["drop"], "input_drop", cfg["input_drop"])
print("bypass_query_encoder", cfg.get("bypass_query_encoder"))
print("root", cfg["root"])
print("data_root", cfg["data_root"])
if cfg["visual_feature"] != "i3d_resnet":
    raise SystemExit("FAIL visual_feature=%s" % cfg["visual_feature"])
if int(cfg["q_feat_size"]) != 512:
    raise SystemExit("FAIL q_feat_size=%s" % cfg["q_feat_size"])
if abs(float(cfg["lr"]) - 0.0003) > 1e-12:
    raise SystemExit("FAIL lr=%s want 0.0003" % cfg["lr"])
if abs(float(cfg["sft_factor"]) - 0.09) > 1e-12:
    raise SystemExit("FAIL sft_factor=%s want 0.09" % cfg["sft_factor"])
if not bool(cfg.get("bypass_query_encoder", False)):
    raise SystemExit("FAIL bypass_query_encoder=%s" % cfg.get("bypass_query_encoder"))
if int(cfg["n_epoch"]) != 100:
    raise SystemExit("FAIL n_epoch=%s" % cfg["n_epoch"])
if int(cfg["max_es_cnt"]) != 10:
    raise SystemExit("FAIL max_es_cnt=%s" % cfg["max_es_cnt"])
if int(cfg["hard_negative_start_epoch"]) != 20:
    raise SystemExit("FAIL HN=%s" % cfg["hard_negative_start_epoch"])
if int(cfg["batchsize"]) != 128:
    raise SystemExit("FAIL batchsize=%s" % cfg["batchsize"])
if int(cfg["hidden_size"]) != 384:
    raise SystemExit("FAIL hidden_size=%s" % cfg["hidden_size"])
if list(cfg["loss_factor"]) != [0.05, 0.04, 8e-5, 0.09]:
    raise SystemExit("FAIL loss_factor=%s" % cfg["loss_factor"])
if list(cfg["neg_factor"]) != [0.15, 32, 1]:
    raise SystemExit("FAIL neg_factor=%s" % cfg["neg_factor"])
if "qbert_bypass" not in cfg["root"]:
    raise SystemExit("FAIL root not qbert_bypass %s" % cfg["root"])
if cfg["root"].rstrip("/").endswith("tvr_i3d_cliptext_e4_seed9527"):
    raise SystemExit("FAIL would overwrite original E4")
if "lr2e-4" in cfg["root"]:
    raise SystemExit("FAIL would overwrite lr2e-4")
if "sft06" in cfg["root"]:
    raise SystemExit("FAIL would overwrite sft06")
if "e4_eot" in cfg["root"]:
    raise SystemExit("FAIL would overwrite EOT")
if cfg["root"].rstrip("/").endswith("tvr_i3d_seed9527"):
    raise SystemExit("FAIL would overwrite I3D TVR")
if cfg["root"].rstrip("/").endswith("i3d_cliptext_e4_seed9527"):
    raise SystemExit("FAIL would overwrite Charades E4")
if "act_i3d_cliptext_e4" in cfg["root"]:
    raise SystemExit("FAIL would overwrite Act E4")

old_e4 = os.path.join(root, "tmp", "tvr_i3d_cliptext_e4_seed9527")
old_lr = os.path.join(root, "tmp", "tvr_i3d_cliptext_e4_lr2e-4_seed9527")
old_sft = os.path.join(root, "tmp", "tvr_i3d_cliptext_e4_sft06_seed9527")
old_eot = os.path.join(root, "tmp", "tvr_i3d_cliptext_e4_eot_seed9527")
old_i3d = os.path.join(root, "tmp", "tvr_i3d_seed9527")
old_cha = os.path.join(root, "tmp", "i3d_cliptext_e4_seed9527")
old_act = os.path.join(root, "tmp", "act_i3d_cliptext_e4_seed9527")
for p, name in (
    (old_e4, "original E4"),
    (old_lr, "lr2e-4"),
    (old_sft, "sft06"),
    (old_eot, "EOT"),
    (old_i3d, "I3D TVR"),
    (old_cha, "Charades E4"),
    (old_act, "Act E4"),
):
    if not os.path.isdir(p):
        raise SystemExit("FAIL missing %s dir" % name)

txt = os.path.join(cfg["data_root"], "tvr", "TextData", "tvr_clip_B32_proj.h5")
eot = os.path.join(cfg["data_root"], "tvr", "TextData", "tvr_clip_B32_eot.h5")
zip_eot = os.path.join(cfg["data_root"], "tvr", "TextData", "clip_ViT_B_32_tvr_query_feat.hdf5")
print("proj", txt, os.path.isfile(txt), os.path.getsize(txt) if os.path.isfile(txt) else 0)
print("eot_unused", eot, os.path.isfile(eot))
print("zip_eot_unused", zip_eot, os.path.isfile(zip_eot))
if not os.path.isfile(txt):
    raise SystemExit("FAIL missing CLIP proj h5")

with h5py.File(txt, "r") as h:
    keys = list(h.keys())[:20]
    k0 = keys[0]
    arr = h[k0][...]
    print("one_shape", k0, arr.shape)
    ranks = {len(h[k].shape) for k in keys}
    dims = {h[k].shape[-1] for k in keys}
    tlen = {h[k].shape[0] for k in keys if len(h[k].shape) >= 2}
    print("sample", [(k, h[k].shape) for k in keys[:5]])
    print("ranks", ranks, "dims", dims, "tlen_sample", tlen)
    if ranks != {2}:
        raise SystemExit("FAIL CLIP text not token sequence ranks=%s" % ranks)
    if dims != {512}:
        raise SystemExit("FAIL CLIP dim %s" % dims)
    if arr.ndim != 2 or int(arr.shape[0]) <= 1:
        raise SystemExit("FAIL proj not sequence T>1 %s" % (arr.shape,))

model_py = open(os.path.join(src, "Models", "gmmformerV2", "model.py"), encoding="utf-8").read()
if "self.query_encoder = BertAttention" not in model_py:
    raise SystemExit("FAIL query_encoder init removed")
if "bypass_query_encoder" not in model_py:
    raise SystemExit("FAIL model.py missing bypass flag")
if "self.query_pos_embed(self.query_input_proj(query_feat))" not in model_py:
    raise SystemExit("FAIL encode_query bypass body missing")
if "def encode_input(feat, mask, input_proj_layer, encoder_layer" not in model_py:
    raise SystemExit("FAIL encode_input missing")
comp = open(
    os.path.join(src, "Models", "gmmformerV2", "model_components.py"), encoding="utf-8"
).read()
if "out = torch.mean(oo, dim = -1).squeeze()" in comp:
    raise SystemExit("FAIL TC mean overwrite present")
if "out = torch.sum(oo * weight" not in comp:
    raise SystemExit("FAIL TC weighted sum missing")
if "class BertAttention" not in comp:
    raise SystemExit("FAIL BertAttention class missing")

builder = open(os.path.join(src, "Datasets", "builder.py"), encoding="utf-8").read()
if "qbert_bypass" not in builder:
    raise SystemExit("FAIL builder missing qbert_bypass")
if "clip_B32_proj.h5" not in builder:
    raise SystemExit("FAIL builder missing proj h5")
if "clip_ViT_B_32_tvr_query_feat.hdf5" in builder:
    raise SystemExit("FAIL builder wired to zip EOT hdf5")
mb = open(os.path.join(src, "Models", "builder.py"), encoding="utf-8").read()
if "bypass_query_encoder" not in mb:
    raise SystemExit("FAIL Models/builder.py does not pass bypass flag")

from Utils.basic_utils import BigFile

vis_dir = os.path.join(cfg["data_root"], "tvr", "FeatureData", "i3d_resnet")
bf = BigFile(vis_dir)
print("bigfile", bf.nr_of_images, bf.ndims)
if int(bf.ndims) != 3072:
    raise SystemExit("FAIL i3d_resnet dim=%s" % bf.ndims)

yp = os.path.join(cfg["model_root"], "hyperparams.yaml")
print("hyperparams", yp, os.path.isfile(yp))
if "--fast" in sys.argv:
    print("QBERT_BYPASS_FAST_CHECK_OK")
    raise SystemExit(0)

from Datasets.builder import get_datasets
from Models.builder import get_models

cfg, train_loader, context_loader, query_loader = get_datasets(cfg)
print("visual_feat_dim", cfg.get("visual_feat_dim"))
if int(cfg.get("visual_feat_dim", -1)) != 3072:
    raise SystemExit("FAIL visual_feat_dim=%s" % cfg.get("visual_feat_dim"))
ds = train_loader.dataset
tpath = str(getattr(ds, "text_feat_path", ""))
print("text_feat_path", tpath)
if "tvr_clip_B32_proj.h5" not in tpath:
    raise SystemExit("FAIL dataset text path %s" % tpath)
if "eot.h5" in tpath or "roberta_" in tpath.lower():
    raise SystemExit("FAIL dataset text is not CLIP proj %s" % tpath)
batch = next(iter(train_loader))
tf = batch["text_feat"]
print("text_feat", tuple(tf.shape))
if int(tf.shape[-1]) != 512:
    raise SystemExit("FAIL text_feat dim %s" % (tuple(tf.shape),))
if int(tf.shape[1]) <= 1:
    raise SystemExit("FAIL text_feat is not sequence T>1 %s" % (tuple(tf.shape),))
print("ntrain", len(train_loader.dataset), "nctx", len(context_loader.dataset), "nq", len(query_loader.dataset))

model = get_models(cfg)
if not hasattr(model, "query_encoder"):
    raise SystemExit("FAIL model has no query_encoder module")
print("bypass_on_model", bool(model.config.get("bypass_query_encoder", False)))
if not bool(model.config.get("bypass_query_encoder", False)):
    raise SystemExit("FAIL model config bypass false")
called = []
orig = model.query_encoder.forward

def _hook(*args, **kwargs):
    called.append(1)
    return orig(*args, **kwargs)

model.query_encoder.forward = _hook
model.eval()
with torch.no_grad():
    q = model.encode_query(tf, batch["text_mask"])
print("video_query", tuple(q.shape), "encoder_calls", len(called))
if called:
    raise SystemExit("FAIL query_encoder was called under bypass")
print("QBERT_BYPASS_NTRAIN_CHECK_OK")
