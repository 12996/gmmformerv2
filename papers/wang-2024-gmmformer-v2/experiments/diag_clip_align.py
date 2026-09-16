"""Diagnose Charades CLIP feature/id alignment. No training."""
import os
import re
import h5py
import numpy as np

root = "/data/zhaopu/wang-2024-gmmformer-v2/data/prvr/charades"
cap_val = os.path.join(root, "TextData/charadesval.caption.txt")
cap_tr = os.path.join(root, "TextData/charadestrain.caption.txt")
vis = os.path.join(root, "FeatureData/charades_clip_L14.h5")
txt = os.path.join(root, "TextData/charades_clip_L14.h5")


def parse_caps(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cap_id, cap = line.split(" ", 1)
            vid = cap_id.split("#")[0]
            rows.append((cap_id, vid, cap))
    return rows


val = parse_caps(cap_val)
tr = parse_caps(cap_tr)
print("N_TRAIN_CAP", len(tr), "N_VAL_CAP", len(val))
print("EX_TRAIN", tr[0])
print("EX_VAL", val[0], val[1] if len(val) > 1 else None)
print("VAL_VID_UNIQUE", len({v for _, v, _ in val}))

with h5py.File(vis, "r") as fv, h5py.File(txt, "r") as ft:
    vk = list(fv.keys())
    tk = list(ft.keys())
    print("VIS_N", len(vk), "TXT_N", len(tk))
    print("VIS_KEY0", vk[0], "shape", fv[vk[0]].shape, "dtype", fv[vk[0]].dtype)
    print("TXT_KEY0", tk[0], "shape", ft[tk[0]].shape, "dtype", ft[tk[0]].dtype)
    print("VIS_KEY1", vk[1], "shape", fv[vk[1]].shape)
    print("TXT_KEY1", tk[1], "shape", ft[tk[1]].shape)
    vis_set, txt_set = set(vk), set(tk)
    val_vids = {v for _, v, _ in val}
    val_caps = {c for c, _, _ in val}
    print("VAL_VID_IN_VIS", sum(v in vis_set for v in val_vids), "/", len(val_vids))
    print("VAL_CAP_IN_TXT", sum(c in txt_set for c in val_caps), "/", len(val_caps))
    miss_v = [v for v in list(val_vids)[:20] if v not in vis_set]
    miss_c = [c for c in list(val_caps)[:20] if c not in txt_set]
    print("MISS_VID_EX", miss_v[:5])
    print("MISS_CAP_EX", miss_c[:5])
    # strip v_ ?
    alt = sum(re.sub(r"^v_", "", v) in vis_set or ("v_" + v) in vis_set for v in val_vids)
    print("VAL_VID_ALT_PREFIX", alt)

    # Frozen mean-pool cosine retrieval on val (subset)
    n_vid = min(1334, len(val_vids))
    vids = []
    seen = set()
    for _, v, _ in val:
        if v in vis_set and v not in seen:
            seen.add(v)
            vids.append(v)
        if len(vids) >= n_vid:
            break
    V = np.stack([fv[v][...].mean(axis=0) for v in vids]).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-6
    vid_index = {v: i for i, v in enumerate(vids)}
    caps = [(c, v) for c, v, _ in val if c in txt_set and v in vid_index]
    print("PAIRS", len(caps), "GALLERY", len(vids))
    Q = np.stack([ft[c][...].mean(axis=0) for c, _ in caps]).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-6
    scores = Q @ V.T  # nq x nv
    gt = np.array([vid_index[v] for _, v in caps], dtype=np.int64)
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(len(caps), dtype=np.int32)
    for i in range(len(caps)):
        ranks[i] = int(np.where(order[i] == gt[i])[0][0]) + 1
    def rec(k):
        return 100.0 * float((ranks <= k).mean())
    print(
        "CLIP_MEANPOOL R@1 %.2f R@5 %.2f R@10 %.2f R@100 %.2f Rsum %.2f mean_rank %.1f"
        % (rec(1), rec(5), rec(10), rec(100), rec(1) + rec(5) + rec(10) + rec(100), ranks.mean())
    )
    # last-token query instead of mean
    Q2 = np.stack([ft[c][...][-1] for c, _ in caps]).astype(np.float32)
    Q2 /= np.linalg.norm(Q2, axis=1, keepdims=True) + 1e-6
    scores2 = Q2 @ V.T
    order2 = np.argsort(-scores2, axis=1)
    ranks2 = np.array([int(np.where(order2[i] == gt[i])[0][0]) + 1 for i in range(len(caps))])
    def rec2(k):
        return 100.0 * float((ranks2 <= k).mean())
    print(
        "CLIP_LASTTOK R@1 %.2f R@5 %.2f R@10 %.2f R@100 %.2f Rsum %.2f mean_rank %.1f"
        % (rec2(1), rec2(5), rec2(10), rec2(100), rec2(1) + rec2(5) + rec2(10) + rec2(100), ranks2.mean())
    )
    print("SCORE_MEAN", float(scores.mean()), "POS_MEAN", float(scores[np.arange(len(caps)), gt].mean()))
