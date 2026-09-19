"""I3D visual + E2 CLIP projected token text. Do not revert text to RoBERTa."""
from pathlib import Path

ROOT = Path("/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src")
bd = ROOT / "Datasets" / "builder.py"
t = bd.read_text()
old = (
    "    if cfg['visual_feature'] in ('i3d', 'i3d_rgb_lgi'):\n"
    "        visual_feats = os.path.join(rootpath, collection, 'FeatureData', '%s_i3d.h5' % collection)\n"
    "        text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)\n"
    "        video2frames = None\n"
    "        test_visual_feats = visual_feats\n"
    "        test_video2frames = None\n"
    "        is_clip = True\n"
    "        cfg['visual_feat_dim'] = 1024\n"
    "        cfg['q_feat_size'] = 1024\n"
)
new = (
    "    if cfg['visual_feature'] in ('i3d', 'i3d_rgb_lgi'):\n"
    "        visual_feats = os.path.join(rootpath, collection, 'FeatureData', '%s_i3d.h5' % collection)\n"
    "        text_feat_path = os.path.join(rootpath, collection, 'TextData', '%s_clip_L14.h5' % collection)\n"
    "        video2frames = None\n"
    "        test_visual_feats = visual_feats\n"
    "        test_video2frames = None\n"
    "        is_clip = True\n"
    "        cfg['visual_feat_dim'] = 1024\n"
    "        cfg['q_feat_size'] = 512\n"
)
if old not in t:
    if "i3d_rgb_lgi" in t and "%s_clip_L14.h5" in t and "roberta_%s_query_feat.hdf5" not in t.split("i3d_rgb_lgi")[1][:500]:
        print("BUILDER_ALREADY")
    elif new in t:
        print("BUILDER_ALREADY")
    else:
        raise SystemExit("BUILDER_FAIL")
else:
    bd.write_text(t.replace(old, new, 1))
    print("BUILDER_OK")

cha_src = Path("/data/zhaopu/wang-2024-gmmformer-v2/experiments/cha_e4_5090_1.py").read_text()
cha_dst = ROOT / "Configs" / "cha.py"
bak = ROOT / "Configs" / "cha.py.bak_e3_before_e4"
if not bak.exists():
    bak.write_text(cha_dst.read_text())
    print("CHA_BAK", bak)
cha_dst.write_text(cha_src)
print("CHA_E4_OK")
