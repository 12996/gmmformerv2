#!/usr/bin/env python3
"""TVR E4 EOT: point i3d_cliptext_e4_eot PRVR_ROOT at tvr_clip_B32_eot.h5.

Keeps parent i3d_cliptext_e4 on proj tokens. Never zip EOT hdf5.
"""
from pathlib import Path

ROOT = Path("/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src")
bd = ROOT / "Datasets" / "builder.py"
t = bd.read_text()
old = (
    "        if 'i3d_cliptext_e4' in os.environ.get('PRVR_ROOT', ''):\n"
    "            text_feat_path = os.path.join(rootpath, collection, 'TextData', "
    "'%s_clip_B32_proj.h5' % collection)\n"
    "        else:\n"
    "            text_feat_path = os.path.join(rootpath, collection, 'TextData', "
    "'roberta_%s_query_feat.hdf5' % collection)\n"
)
new = (
    "        if 'i3d_cliptext_e4_eot' in os.environ.get('PRVR_ROOT', ''):\n"
    "            text_feat_path = os.path.join(rootpath, collection, 'TextData', "
    "'%s_clip_B32_eot.h5' % collection)\n"
    "        elif 'i3d_cliptext_e4' in os.environ.get('PRVR_ROOT', ''):\n"
    "            text_feat_path = os.path.join(rootpath, collection, 'TextData', "
    "'%s_clip_B32_proj.h5' % collection)\n"
    "        else:\n"
    "            text_feat_path = os.path.join(rootpath, collection, 'TextData', "
    "'roberta_%s_query_feat.hdf5' % collection)\n"
)
if "i3d_cliptext_e4_eot" in t and "%s_clip_B32_eot.h5" in t:
    print("E4_EOT_TEXT_ALREADY")
elif old not in t:
    raise SystemExit("E4_EOT_TEXT_FAIL no i3d_cliptext_e4 proj block")
else:
    n = t.count(old)
    t = t.replace(old, new)
    bd.write_text(t)
    print("E4_EOT_TEXT_OK", n)

t = bd.read_text()
if "clip_ViT_B_32_tvr_query_feat.hdf5" in t:
    raise SystemExit("ZIP_EOT_WIRED")
if "tvr_train_release.jsonl" in t:
    raise SystemExit("JSONL_STILL_PRESENT")
if "PRVR_EVAL_SPLIT" not in t:
    raise SystemExit("SPLIT_MISSING")
if "%s_clip_B32_eot.h5" not in t:
    raise SystemExit("EOT_H5_MISSING")
if "i3d_cliptext_e4_eot" not in t:
    raise SystemExit("EOT_ROOT_MISSING")
print("BUILDER_JSONL_GONE", "tvr_train_release.jsonl" not in t)
print("BUILDER_EVAL_SPLIT", "PRVR_EVAL_SPLIT" in t)
print("BUILDER_EOT_H5", "%s_clip_B32_eot.h5" in t)
print("BUILDER_PROJ_KEPT", "%s_clip_B32_proj.h5" in t)
print("ZIP_EOT_ABSENT", "clip_ViT_B_32_tvr_query_feat.hdf5" not in t)
