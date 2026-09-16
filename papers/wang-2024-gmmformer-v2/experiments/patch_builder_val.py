from pathlib import Path

p = Path("/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src/Datasets/builder.py")
t = p.read_text()
old = (
    "    val_video_ids_list = read_video_ids(caption_files['val'])\n"
    "    val_video_dataset = VisDataSet4PRVR(visual_feats, text_feat_path, cfg, video_ids=val_video_ids_list, is_clip=is_clip)\n"
)
new = (
    "    val_video_ids_list = read_video_ids(caption_files['val'])\n"
    "    # validations.py encode_context needs collate_train dicts with captions\n"
    "    val_video_dataset = Dataset4PRVR(caption_files['val'], visual_feats, text_feat_path, cfg,\n"
    "                                     video2frames=video2frames, is_clip=is_clip, path_query_json=None)\n"
)
if "Dataset4PRVR(caption_files['val']" in t and "val_video_dataset = Dataset4PRVR" in t:
    print("PATCH_ALREADY")
elif old not in t:
    raise SystemExit("PATCH_FAIL: old val_video_dataset block not found")
else:
    p.write_text(t.replace(old, new, 1))
    print("PATCH_OK")
