"""Switch teacher zip to I3D h5 + official 32-clip / 128-frame sampling."""
from pathlib import Path

ROOT = Path("/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src")

OLD_VIS = """        clip_video_feature, merge_size, merged_frame_idx = average_to_fixed_length(frame_vecs, self.map_size, clip=False)
        # import pdb
        # pdb.set_trace()
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)
        merge_size = torch.from_numpy(merge_size).unsqueeze(0)
        merged_frame_idx = torch.from_numpy(merged_frame_idx).unsqueeze(0)

        frame_video_feature = average_to_fixed_length(frame_vecs, self.map_size, clip=None)
        # frame_video_feature = uniform_feature_sampling(frame_vecs, self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)
"""
NEW_VIS = """        clip_arr = average_to_fixed_length(frame_vecs, self.map_size, clip=None)
        clip_arr = l2_normalize_np_array(clip_arr)
        clip_video_feature = torch.from_numpy(clip_arr).unsqueeze(0)
        merge_size = torch.ones(1, clip_arr.shape[0], dtype=torch.float32)
        merged_frame_idx = torch.arange(clip_arr.shape[0]).unsqueeze(0)

        frame_arr = uniform_feature_sampling(frame_vecs, self.max_ctx_len)
        frame_arr = l2_normalize_np_array(frame_arr)
        frame_video_feature = torch.from_numpy(frame_arr)
"""
OLD_VIS2 = """        clip_video_feature, merge_size, merged_frame_idx = average_to_fixed_length(frame_vecs, self.map_size, clip=False)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)
        merge_size = torch.from_numpy(merge_size).unsqueeze(0)

        frame_video_feature = average_to_fixed_length(frame_vecs, self.map_size, clip=None)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)
"""
NEW_VIS2 = """        clip_arr = average_to_fixed_length(frame_vecs, self.map_size, clip=None)
        clip_arr = l2_normalize_np_array(clip_arr)
        clip_video_feature = torch.from_numpy(clip_arr).unsqueeze(0)
        merge_size = torch.ones(1, clip_arr.shape[0], dtype=torch.float32)

        frame_arr = uniform_feature_sampling(frame_vecs, self.max_ctx_len)
        frame_arr = l2_normalize_np_array(frame_arr)
        frame_video_feature = torch.from_numpy(frame_arr)
"""

dp = ROOT / "Datasets" / "data_provider.py"
t = dp.read_text()
n1 = t.count(OLD_VIS)
n2 = t.count(OLD_VIS2)
if n1 + n2 == 0:
    if "uniform_feature_sampling(frame_vecs, self.max_ctx_len)" in t:
        print("DP_ALREADY")
    else:
        raise SystemExit("DP_FAIL")
else:
    t = t.replace(OLD_VIS, NEW_VIS).replace(OLD_VIS2, NEW_VIS2)
    dp.write_text(t)
    print("DP_OK", n1, n2)

bd = ROOT / "Datasets" / "builder.py"
b = bd.read_text()
needle = "    if cfg['visual_feature'] == 'clip':"
insert = """    if cfg['visual_feature'] in ('i3d', 'i3d_rgb_lgi'):
        visual_feats = os.path.join(rootpath, collection, 'FeatureData', '%s_i3d.h5' % collection)
        text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
        video2frames = None
        test_visual_feats = visual_feats
        test_video2frames = None
        is_clip = True
        cfg['visual_feat_dim'] = 1024
        cfg['q_feat_size'] = 1024

    el"""
if "i3d_rgb_lgi" in b and "charades_i3d.h5" in b:
    print("BUILDER_ALREADY")
elif needle not in b:
    raise SystemExit("BUILDER_FAIL")
else:
    bd.write_text(b.replace(needle, insert + "if cfg['visual_feature'] == 'clip':", 1))
    print("BUILDER_OK")
