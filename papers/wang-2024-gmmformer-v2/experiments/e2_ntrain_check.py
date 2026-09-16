#!/usr/bin/env python3
import os
import sys

root = "/data/zhaopu/wang-2024-gmmformer-v2"
os.environ.setdefault("PRVR_SEED", "9527")
os.environ["PRVR_ROOT"] = os.path.join(root, "tmp", "e2_ntrain_check")
src = os.path.join(root, "vendor", "GMMFormer_v2", "src")
sys.path.insert(0, src)
os.chdir(src)

from Configs.cha import get_cfg_defaults
from Datasets.builder import get_datasets

cfg = get_cfg_defaults()
print("visual_feature", cfg["visual_feature"])
print("q_feat_size", cfg["q_feat_size"])
print("visual_feat_dim", cfg["visual_feat_dim"])
print("batchsize", cfg["batchsize"])
print("seed", cfg["seed"])
print("max_ctx_l", cfg["max_ctx_l"])
cfg, train_loader, context_dataloader, query_eval_loader = get_datasets(cfg)
ntrain = len(train_loader.dataset)
print("ntrain", ntrain)
print("n_val_text", len(query_eval_loader.dataset))
print("n_val_video", len(context_dataloader.dataset))
batch = next(iter(train_loader))
print("batch_type", type(batch).__name__, len(batch) if hasattr(batch, "__len__") else None)
if isinstance(batch, dict):
    for k, v in batch.items():
        shape = getattr(v, "shape", None)
        print("batch", k, tuple(shape) if shape is not None else type(v).__name__)
elif isinstance(batch, (list, tuple)):
    for i, v in enumerate(batch):
        shape = getattr(v, "shape", None)
        print("batch_item", i, type(v).__name__, tuple(shape) if shape is not None else None)
print("NTRAIN_CHECK_OK")
