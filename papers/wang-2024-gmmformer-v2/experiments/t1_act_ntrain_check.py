#!/usr/bin/env python3
"""CPU ntrain check for T1-act. Must not touch Charades h5 or E1-E4 tmp."""
import os
import sys

root = "/data/zhaopu/wang-2024-gmmformer-v2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PRVR_SEED", "9527")
os.environ["PRVR_ROOT"] = os.path.join(root, "tmp", "act_ntrain_check")
src = os.path.join(root, "vendor", "GMMFormer_v2", "src")
sys.path.insert(0, src)
os.chdir(src)

from Configs.act import get_cfg_defaults
from Datasets.builder import get_datasets
from Datasets.data_provider import read_video_ids
from Utils.basic_utils import BigFile

cfg = get_cfg_defaults()
print("visual_feature", cfg["visual_feature"])
print("q_feat_size", cfg["q_feat_size"])
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
print("loss_factor", cfg["loss_factor"])
print("input_drop", cfg["input_drop"])

if cfg["visual_feature"] != "i3d":
    raise SystemExit("FAIL visual_feature=%s" % cfg["visual_feature"])
if int(cfg["q_feat_size"]) != 1024:
    raise SystemExit("FAIL q_feat_size=%s" % cfg["q_feat_size"])
if cfg["collection"] != "activitynet":
    raise SystemExit("FAIL collection=%s" % cfg["collection"])
if abs(float(cfg["lr"]) - 0.00025) > 1e-12:
    raise SystemExit("FAIL lr=%s" % cfg["lr"])
if int(cfg["max_es_cnt"]) != 10:
    raise SystemExit("FAIL max_es_cnt=%s" % cfg["max_es_cnt"])
if bool(cfg["use_hard_negative"]):
    raise SystemExit("FAIL use_hard_negative should be False at init")
if int(cfg["batchsize"]) != 128:
    raise SystemExit("FAIL batchsize=%s" % cfg["batchsize"])
if int(cfg["seed"]) != 9527:
    raise SystemExit("FAIL seed=%s" % cfg["seed"])
if int(cfg["max_ctx_l"]) != 128 or int(cfg["map_size"]) != 32:
    raise SystemExit("FAIL clip/frame %s %s" % (cfg["map_size"], cfg["max_ctx_l"]))
if int(cfg["hidden_size"]) != 384:
    raise SystemExit("FAIL hidden_size=%s" % cfg["hidden_size"])

act_root = os.path.join(cfg["data_root"], "activitynet")
vis_dir = os.path.join(act_root, "FeatureData", "i3d")
txt = os.path.join(act_root, "TextData", "roberta_activitynet_query_feat.hdf5")
train_cap = os.path.join(act_root, "TextData", "activitynettrain.caption.txt")
val_cap = os.path.join(act_root, "TextData", "activitynetval.caption.txt")
test_cap = os.path.join(act_root, "TextData", "activitynettest.caption.txt")
print("vis_dir", vis_dir, os.path.isdir(vis_dir))
print("txt", txt, os.path.isfile(txt))
print("train_cap", train_cap, os.path.isfile(train_cap))
print("val_cap", val_cap, os.path.isfile(val_cap))
print("test_cap", test_cap, os.path.isfile(test_cap))
for p in (vis_dir, txt, train_cap, val_cap):
    if not os.path.exists(p):
        raise SystemExit("FAIL missing %s" % p)
for name in ("feature.bin", "id.txt", "shape.txt", "video2frames.txt"):
    fp = os.path.join(vis_dir, name)
    print("feat", name, os.path.isfile(fp), os.path.getsize(fp) if os.path.isfile(fp) else 0)
    if not os.path.isfile(fp):
        raise SystemExit("FAIL missing BigFile %s" % fp)

bf = BigFile(vis_dir)
print("bigfile", bf.nr_of_images, bf.ndims)
if int(bf.ndims) != 1024:
    raise SystemExit("FAIL i3d dim=%s" % bf.ndims)

ntrain_v = len(read_video_ids(train_cap))
nval_v = len(read_video_ids(val_cap))
ntrain_q = sum(1 for _ in open(train_cap, "r", encoding="utf-8", errors="replace"))
nval_q = sum(1 for _ in open(val_cap, "r", encoding="utf-8", errors="replace"))
print("ntrain_v", ntrain_v, "ntrain_q", ntrain_q)
print("nval_v", nval_v, "nval_q", nval_q)
if ntrain_v < 8000 or ntrain_q < 30000:
    raise SystemExit("FAIL train split too small")

builder = open(os.path.join(src, "Datasets", "builder.py"), encoding="utf-8").read()
if "collection == 'activitynet' and cfg['visual_feature'] == 'i3d'" not in builder:
    raise SystemExit("FAIL builder missing activitynet BigFile branch")
if "roberta_%s_query_feat.hdf5" not in builder:
    raise SystemExit("FAIL builder missing roberta hdf5")
cha_i3d = builder[builder.find("if cfg['visual_feature'] in ('i3d'"): builder.find("elif cfg['visual_feature'] == 'clip':")]
if "activitynet_i3d.h5" in cha_i3d and "collection == 'activitynet'" not in builder[: builder.find("if cfg['visual_feature'] in ('i3d'")]:
    raise SystemExit("FAIL activitynet would hit charades h5 branch")

comp = open(os.path.join(src, "Models", "gmmformerV2", "model_components.py"), encoding="utf-8").read()
if "out = torch.mean(oo, dim = -1).squeeze()" in comp:
    raise SystemExit("FAIL TC mean overwrite still present")
if "out = torch.sum(oo * weight" not in comp:
    raise SystemExit("FAIL TC weighted sum missing")

cha = open(os.path.join(src, "Configs", "cha.py"), encoding="utf-8").read()
if "q_feat_size" in cha and "512" not in cha:
    print("WARN cha.py q_feat_size may have changed")
if 'visual_feature"] = "clip"' in cha or "visual_feature'] = 'clip'" in cha:
    print("WARN cha.py is clip; leave it, do not overwrite")

keep = [
    os.path.join(root, "tmp", "i3d_tc_seed9527"),
    os.path.join(root, "tmp", "clip_e2_seed9527"),
    os.path.join(root, "tmp", "clip_roberta_e3_seed9527"),
    os.path.join(root, "tmp", "i3d_cliptext_e4_seed9527"),
    os.path.join(root, "tmp", "tvr_i3d_seed9527"),
    os.path.join(root, "tmp", "tvr_i3d_cliptext_e4_seed9527"),
    os.path.join(root, "data", "prvr", "charades", "FeatureData", "charades_i3d.h5"),
    os.path.join(root, "data", "prvr", "charades", "TextData", "roberta_charades_query_feat.hdf5"),
]
for p in keep:
    if not os.path.exists(p):
        raise SystemExit("FAIL charades keep missing %s" % p)
    print("KEEP", p)

ds = get_datasets(cfg)
if len(ds) == 6:
    cfg, train_loader, context_dataloader, query_eval_loader, test_context_dataloader, test_query_eval_loader = ds
    print("LOADERS 6")
else:
    cfg, train_loader, context_dataloader, query_eval_loader = ds
    test_context_dataloader = test_query_eval_loader = None
    print("LOADERS 4")
print("visual_feat_dim", cfg.get("visual_feat_dim"))
if int(cfg.get("visual_feat_dim", 0)) != 1024:
    raise SystemExit("FAIL visual_feat_dim=%s" % cfg.get("visual_feat_dim"))
if int(cfg["q_feat_size"]) != 1024:
    raise SystemExit("FAIL q_feat_size overwritten %s" % cfg["q_feat_size"])

ntrain = len(train_loader.dataset)
nctx = len(context_dataloader.dataset)
nq = len(query_eval_loader.dataset)
print("ntrain", ntrain, "nctx", nctx, "nq", nq)
if ntrain < 8000:
    raise SystemExit("FAIL ntrain=%s" % ntrain)
ds0 = train_loader.dataset
print("is_clip", getattr(ds0, "is_clip", None))
print("text_feat_path", getattr(ds0, "text_feat_path", None))
if getattr(ds0, "is_clip", True):
    raise SystemExit("FAIL is_clip True; want BigFile")
if "roberta_activitynet_query_feat.hdf5" not in str(getattr(ds0, "text_feat_path", "")):
    raise SystemExit("FAIL text is not MS-SL RoBERTa hdf5")
if "charades" in str(getattr(ds0, "text_feat_path", "")).lower():
    raise SystemExit("FAIL loaded charades text")

item = ds0[0]
print("item0_clip", tuple(item[0].shape), "item0_frame", tuple(item[1].shape), "ncap", len(item[2]), "qid_dim", tuple(item[2][0].shape))
if tuple(item[0].shape[-2:]) != (32, 1024):
    raise SystemExit("FAIL clip feat shape %s" % (item[0].shape,))
if item[1].shape[-1] != 1024:
    raise SystemExit("FAIL frame feat dim %s" % (item[1].shape,))
if item[2][0].shape[-1] != 1024:
    raise SystemExit("FAIL text dim %s" % (item[2][0].shape,))

if test_context_dataloader is not None:
    ntest_v = len(test_context_dataloader.dataset)
    ntest_q = len(test_query_eval_loader.dataset)
    print("ntest_v", ntest_v, "ntest_q", ntest_q)
    if ntest_v < 4000 or ntest_q < 14000:
        raise SystemExit("FAIL test split too small")
else:
    print("NO_TEST_LOADER")
    if os.path.isfile(test_cap):
        raise SystemExit("FAIL test caption exists but loader missing")

print("NTRAIN_CHECK_OK")
