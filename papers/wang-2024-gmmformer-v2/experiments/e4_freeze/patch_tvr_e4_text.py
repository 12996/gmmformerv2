#!/usr/bin/env python3
"""TVR E4: keep i3d_resnet BigFile visual; switch only text to CLIP proj tokens.

Does not change Charades i3d_rgb_lgi / clip branches.
Does not use zip EOT vectors clip_ViT_B_32_tvr_query_feat.hdf5.
"""
from pathlib import Path

ROOT = Path("/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src")
bd = ROOT / "Datasets" / "builder.py"
t = bd.read_text()
old = (
    "        text_feat_path = os.path.join(rootpath, collection, 'TextData', "
    "'roberta_%s_query_feat.hdf5' % collection)\n"
    "        video2frames = read_dict(\n"
    "            os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))\n"
)
new = (
    "        if 'i3d_cliptext_e4' in os.environ.get('PRVR_ROOT', ''):\n"
    "            text_feat_path = os.path.join(rootpath, collection, 'TextData', '%s_clip_B32_proj.h5' % collection)\n"
    "        else:\n"
    "            text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)\n"
    "        video2frames = read_dict(\n"
    "            os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))\n"
)
if "i3d_cliptext_e4" in t and "%s_clip_B32_proj.h5" in t:
    print("E4_TEXT_ALREADY")
elif old not in t:
    raise SystemExit("E4_TEXT_FAIL no official BigFile roberta text block")
else:
    # Only the BigFile else-branch contains video2frames=read_dict after roberta path.
    t = t.replace(old, new, 1)
    bd.write_text(t)
    print("E4_TEXT_OK")

t = bd.read_text()
if "tvr_train_release.jsonl" in t:
    raise SystemExit("JSONL_STILL_PRESENT")
if "PRVR_EVAL_SPLIT" not in t:
    raise SystemExit("SPLIT_MISSING")
print("BUILDER_JSONL_GONE", "tvr_train_release.jsonl" not in t)
print("BUILDER_EVAL_SPLIT", "PRVR_EVAL_SPLIT" in t)
print("CHA_STRING_STILL", "charades" in t)
