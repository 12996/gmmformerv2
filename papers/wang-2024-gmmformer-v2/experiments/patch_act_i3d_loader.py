"""ActivityNet MS-SL I3D+RoBERTa loader. Do not change Charades CLIP/E4 branches."""
from pathlib import Path

ROOT = Path("/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src")

ACT_BRANCH = """    if collection == 'activitynet' and cfg['visual_feature'] == 'i3d':
        visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'])
        visual_feats = BigFile(visual_feat_path)
        cfg['visual_feat_dim'] = visual_feats.ndims
        if int(cfg.get('q_feat_size', 1024)) == 512:
            text_feat_path = os.path.join(rootpath, collection, 'TextData', '%s_clip_L14.h5' % collection)
        else:
            text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
        video2frames = read_dict(
            os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))
        test_visual_feats = visual_feats
        test_video2frames = video2frames
        is_clip = False

    el"""

OLD_I3D = "    if cfg['visual_feature'] in ('i3d', 'i3d_rgb_lgi'):"

RETURN_4 = "    return cfg, train_loader, context_dataloader, query_eval_loader"
RETURN_6 = """    test_caption_file = os.path.join(rootpath, collection, 'TextData', '%stest.caption.txt' % collection)
    if collection == 'activitynet' and os.path.isfile(test_caption_file):
        test_text_dataset = TxtDataSet4PRVR(test_caption_file, text_feat_path, cfg, path_query_json=None)
        test_video_dataset = Dataset4PRVR(
            test_caption_file, visual_feats, text_feat_path, cfg,
            video2frames=video2frames, is_clip=is_clip, path_query_json=None)
        test_context_dataloader = DataLoader(
            test_video_dataset,
            collate_fn=collate_train,
            batch_size=cfg['eval_context_bsz'],
            num_workers=cfg['num_workers'],
            shuffle=False,
            pin_memory=cfg['pin_memory'])
        test_query_eval_loader = DataLoader(
            test_text_dataset,
            collate_fn=collate_text_val,
            batch_size=cfg['eval_query_bsz'],
            num_workers=cfg['num_workers'],
            shuffle=False,
            pin_memory=cfg['pin_memory'])
        return cfg, train_loader, context_dataloader, query_eval_loader, test_context_dataloader, test_query_eval_loader
    return cfg, train_loader, context_dataloader, query_eval_loader"""

GET_GT_OLD = """def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            query_id= re.sub(r"^v_", "", query_id)
            if query_id.split('#', 1)[0] == vid_id:

                v2t_gt[-1].append(i)
"""

GET_GT_NEW = """def _norm_vid(x):
    x = x.split('#', 1)[0]
    if x.startswith('v_'):
        x = x[2:]
    return x


def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        vid_n = _norm_vid(vid_id)
        for i, query_id in enumerate(query_metas):
            if _norm_vid(query_id) == vid_n:
                v2t_gt[-1].append(i)
"""

MAIN_UNPACK_OLD = "    cfg, train_loader, context_dataloader, query_eval_loader = get_datasets(cfg)"
MAIN_UNPACK_NEW = """    _ds = get_datasets(cfg)
    if len(_ds) == 6:
        cfg, train_loader, context_dataloader, query_eval_loader, test_context_dataloader, test_query_eval_loader = _ds
    else:
        cfg, train_loader, context_dataloader, query_eval_loader = _ds
        test_context_dataloader, test_query_eval_loader = context_dataloader, query_eval_loader"""

MAIN_EVAL_OLD = "                validation(context_dataloader, query_eval_loader, model, val_criterion, cfg, logger, args.resume)"
MAIN_EVAL_NEW = "                validation(test_context_dataloader, test_query_eval_loader, model, val_criterion, cfg, logger, args.resume)"


ELSE_ROBERTA = """        text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
        video2frames = read_dict(
            os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))"""

ELSE_SWITCH = """        if int(cfg.get('q_feat_size', 0)) == 512:
            text_feat_path = os.path.join(rootpath, collection, 'TextData', '%s_clip_L14.h5' % collection)
        else:
            text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
        video2frames = read_dict(
            os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))"""


def patch_builder():
    p = ROOT / "Datasets" / "builder.py"
    t = p.read_text()
    if "collection == 'activitynet' and cfg['visual_feature'] == 'i3d'" in t and "%s_clip_L14.h5" in t.split("activitynet")[1][:800]:
        print("BUILDER_ACT_ALREADY")
    elif "collection == 'activitynet' and cfg['visual_feature'] == 'i3d'" in t:
        old_act = (
            "        text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)\n"
            "        video2frames = read_dict(\n"
            "            os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))"
        )
        # only replace the first occurrence after the activitynet i3d branch
        idx = t.find("collection == 'activitynet' and cfg['visual_feature'] == 'i3d'")
        if idx < 0 or old_act not in t[idx : idx + 900]:
            print("BUILDER_ACT_KEEP")
        else:
            t = t[:idx] + t[idx:].replace(old_act, ELSE_SWITCH.lstrip(), 1)
            print("BUILDER_ACT_TEXT_SWITCH")
    elif OLD_I3D not in t:
        raise SystemExit("BUILDER_FAIL: i3d h5 branch not found")
    else:
        t = t.replace(OLD_I3D, ACT_BRANCH + OLD_I3D, 1)
        print("BUILDER_ACT_OK")
    if "int(cfg.get('q_feat_size', 0)) == 512" in t and "else:\n        visual_feat_path" in t:
        print("BUILDER_ELSE_SWITCH_ALREADY")
    elif ELSE_ROBERTA in t:
        # replace only the else-BigFile copy (second roberta+video2frames if act branch already has switch)
        t = t.replace(ELSE_ROBERTA, ELSE_SWITCH, 1)
        print("BUILDER_ELSE_SWITCH_OK")
    else:
        print("BUILDER_ELSE_SWITCH_SKIP")
    if "test_query_eval_loader" in t and "return cfg, train_loader, context_dataloader, query_eval_loader, test_context_dataloader" in t:
        print("BUILDER_TEST_ALREADY")
    elif RETURN_4 not in t:
        raise SystemExit("BUILDER_FAIL: return 4-tuple not found")
    else:
        t = t.replace(RETURN_4, RETURN_6, 1)
        print("BUILDER_TEST_OK")
    p.write_text(t)


def patch_get_gt():
    p = ROOT / "Validations" / "validations.py"
    t = p.read_text()
    if "def _norm_vid(x):" in t:
        print("GET_GT_ALREADY")
        return
    if GET_GT_OLD not in t:
        raise SystemExit("GET_GT_FAIL: old get_gt not found")
    p.write_text(t.replace(GET_GT_OLD, GET_GT_NEW, 1))
    print("GET_GT_OK")


def patch_main():
    p = ROOT / "main.py"
    t = p.read_text()
    if "len(_ds) == 6" in t:
        print("MAIN_UNPACK_ALREADY")
    elif MAIN_UNPACK_OLD not in t:
        raise SystemExit("MAIN_FAIL: unpack not found")
    else:
        t = t.replace(MAIN_UNPACK_OLD, MAIN_UNPACK_NEW, 1)
        print("MAIN_UNPACK_OK")
    if MAIN_EVAL_NEW in t:
        print("MAIN_EVAL_ALREADY")
    elif MAIN_EVAL_OLD not in t:
        raise SystemExit("MAIN_FAIL: eval call not found")
    else:
        t = t.replace(MAIN_EVAL_OLD, MAIN_EVAL_NEW, 1)
        print("MAIN_EVAL_OK")
    p.write_text(t)


if __name__ == "__main__":
    patch_builder()
    patch_get_gt()
    patch_main()
    print("PATCH_ACT_DONE")
