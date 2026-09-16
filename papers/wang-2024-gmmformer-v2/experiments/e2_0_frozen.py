#!/usr/bin/env python3
"""E2-0 frozen CLIP gates. No GMMFormer training."""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch

ROOT = "/data/zhaopu/wang-2024-gmmformer-v2"
CAP_VAL = os.path.join(ROOT, "data/prvr/charades/TextData/charadesval.caption.txt")
VIS_H5 = os.path.join(ROOT, "data/prvr/charades/FeatureData/charades_clip_L14.h5")
OUT_JSON = os.path.join(ROOT, "logs/e2_0_frozen.json")
LOCAL_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_0_frozen.json")

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_PT = os.environ.get(
    "CLIP_PT",
    os.path.join(ROOT, ".cache/clip/ViT-B-32.pt"),
)
N_SHUFFLE = 1000
SHUFFLE_SEED = 9527
COS_THRESH = 0.9999
BATCH = 64
EPS = 1e-12


def parse_caps(path: str) -> List[Tuple[str, str, str]]:
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


def l2_normalize(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def recall_at(ranks: np.ndarray, k: int) -> float:
    return 100.0 * float((ranks <= k).mean())


def sumr(ranks: np.ndarray) -> Dict[str, float]:
    r1, r5, r10, r100 = (recall_at(ranks, k) for k in (1, 5, 10, 100))
    return {"R@1": r1, "R@5": r5, "R@10": r10, "R@100": r100, "SumR": r1 + r5 + r10 + r100}


def ranks_from_scores(scores: np.ndarray, gt: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranks = np.empty(len(gt), dtype=np.int32)
    for i in range(len(gt)):
        ranks[i] = int(np.where(order[i] == gt[i])[0][0]) + 1
    return ranks


def rank_lookup(scores: np.ndarray) -> np.ndarray:
    nq, nv = scores.shape
    order = np.argsort(-scores, axis=1, kind="mergesort")
    lookup = np.empty((nq, nv), dtype=np.int32)
    ar = np.arange(1, nv + 1, dtype=np.int32)
    for i in range(nq):
        lookup[i, order[i]] = ar
    return lookup


def permute_p(lookup: np.ndarray, gt: np.ndarray, perms: List[np.ndarray], real_sumr: float) -> Dict[str, float]:
    nq = len(gt)
    idx = np.arange(nq)
    n_ge = 0
    shuffle_sumr = []
    for perm in perms:
        new_gt = perm[gt]
        ranks = lookup[idx, new_gt]
        s = sumr(ranks)["SumR"]
        shuffle_sumr.append(s)
        if s >= real_sumr - 1e-12:
            n_ge += 1
    p = (1.0 + n_ge) / (len(perms) + 1.0)
    return {
        "p": p,
        "n_ge": int(n_ge),
        "shuffle_mean_SumR": float(np.mean(shuffle_sumr)),
        "shuffle_max_SumR": float(np.max(shuffle_sumr)),
        "shuffle_min_SumR": float(np.min(shuffle_sumr)),
    }


def encode_text_manual(model, tokens: torch.Tensor) -> torch.Tensor:
    """ln_final[EOT] @ text_projection, independent of encode_text."""
    src = inspect.getsource(model.encode_text)
    x = model.token_embedding(tokens)
    if x.dtype != torch.float32:
        x = x.float()
    pos = model.positional_embedding
    if pos.dtype != torch.float32:
        pos = pos.float()
    if pos.shape[0] >= x.shape[1]:
        x = x + pos[: x.shape[1]]
    else:
        x = x + pos

    attn = getattr(model, "attn_mask", None)
    transformer = model.transformer
    nld = "permute" not in src or "x.permute(1, 0, 2)" not in src.replace(" ", "")
    # Prefer the layout encode_text actually uses.
    if "x = x.permute(1, 0, 2)" in src or "x.permute(1, 0, 2)" in src:
        x = x.permute(1, 0, 2)
        try:
            x = transformer(x, attn_mask=attn) if attn is not None else transformer(x)
        except TypeError:
            x = transformer(x)
        x = x.permute(1, 0, 2)
    else:
        try:
            x = transformer(x, attn_mask=attn) if attn is not None else transformer(x)
        except TypeError:
            x = transformer(x)

    x = model.ln_final(x)
    if x.dtype != torch.float32:
        x = x.float()
    eot = tokens.argmax(dim=-1)
    pooled = x[torch.arange(x.shape[0], device=x.device), eot]
    proj = model.text_projection
    if isinstance(proj, torch.nn.Linear):
        pooled = proj(pooled)
    else:
        if proj.dtype != pooled.dtype:
            proj = proj.float()
        pooled = pooled @ proj
    return pooled


def main() -> int:
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    device = torch.device("cpu")
    print("DEVICE", device, flush=True)

    checkpoint_note = (
        "5090_1 无 /home/zhaopu/wang-2024-gmmformer-v2/data/a3prvr；"
        "A3PRVR README/code 仅写 CLIP-B/32 与 Radford et al. 2021，无权重 hash。"
        "按闸门默认 open_clip ViT-B-32 / openai。"
    )
    print("CKPT_ASSUMPTION", checkpoint_note, flush=True)

    import open_clip
    from open_clip.openai import load_openai_model

    print("OPEN_CLIP", getattr(open_clip, "__version__", "unknown"), flush=True)
    pretrained_used = CLIP_PT if os.path.isfile(CLIP_PT) else CLIP_PRETRAINED
    print("PRETRAINED_USED", pretrained_used, "exists", os.path.isfile(CLIP_PT), flush=True)
    if os.path.isfile(CLIP_PT):
        model = load_openai_model(name=CLIP_PT, precision="fp32", device=device)
    else:
        try:
            model = load_openai_model(name="ViT-B-32", precision="fp32", device=device, cache_dir=os.path.dirname(CLIP_PT))
            pretrained_used = "openai:load_openai_model"
        except Exception as e:
            print("LOAD_OPENAI_FAIL", type(e).__name__, e, flush=True)
            model, _, _ = open_clip.create_model_and_transforms(
                CLIP_MODEL, pretrained=CLIP_PRETRAINED, precision="fp32", device=device
            )
            pretrained_used = CLIP_PRETRAINED
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model = model.float().eval()
    print("TEXT_POOL", getattr(model, "text_pool_type", "argmax-or-missing"), flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    print("ENCODE_TEXT_SRC_HEAD", inspect.getsource(model.encode_text).splitlines()[0:8], flush=True)

    caps = parse_caps(CAP_VAL)
    texts = [c for _, _, c in caps]
    nq = len(caps)
    print("N_VAL_CAP", nq, "UNIQUE_VID_IN_CAPS", len({v for _, v, _ in caps}), flush=True)

    enc_list = []
    man_list = []
    with torch.no_grad():
        for i in range(0, nq, BATCH):
            batch_text = texts[i : i + BATCH]
            tokens = tokenizer(batch_text)
            if not torch.is_tensor(tokens):
                tokens = tokens.to(device)
            else:
                tokens = tokens.to(device)
            enc = model.encode_text(tokens)
            if isinstance(enc, tuple):
                enc = enc[0]
            enc = enc.float()
            man = encode_text_manual(model, tokens).float()
            enc_list.append(enc.cpu().numpy())
            man_list.append(man.cpu().numpy())
            if i % (BATCH * 10) == 0:
                print("encoded", min(i + BATCH, nq), "/", nq, flush=True)

    enc = np.concatenate(enc_list, axis=0).astype(np.float32)
    man = np.concatenate(man_list, axis=0).astype(np.float32)
    enc_norm = np.linalg.norm(enc, axis=1)
    man_norm = np.linalg.norm(man, axis=1)
    n_nan = int((~np.isfinite(enc)).any(axis=1).sum() + (~np.isfinite(man)).any(axis=1).sum())
    n_zero = int(((enc_norm <= EPS).sum()) + ((man_norm <= EPS).sum()))
    enc_l2 = l2_normalize(enc)
    man_l2 = l2_normalize(man)
    cos = np.sum(enc_l2 * man_l2, axis=1)
    min_cos = float(np.min(cos))
    mean_cos = float(np.mean(cos))
    n_below = int((cos < COS_THRESH).sum())
    encode_pass = bool(n_nan == 0 and n_zero == 0 and n_below == 0 and np.isfinite(cos).all())
    print(
        "ENCODE_GATE min_cos=%.8f mean_cos=%.8f n_below=%d n_nan=%d n_zero=%d pass=%s"
        % (min_cos, mean_cos, n_below, n_nan, n_zero, encode_pass),
        flush=True,
    )
    if not encode_pass:
        print("ENCODE_GATE_FAIL", flush=True)

    with h5py.File(VIS_H5, "r") as fv:
        vis_keys = set(fv.keys())
        ex = next(iter(fv.keys()))
        vis_dim = int(fv[ex].shape[-1])
        vis_n = len(vis_keys)
        print("VIS_N", vis_n, "VIS_DIM", vis_dim, "EX", ex, "shape", fv[ex].shape, flush=True)

        val_vids = []
        seen = set()
        missing = []
        for _, vid, _ in caps:
            if vid in seen:
                continue
            seen.add(vid)
            if vid in vis_keys:
                val_vids.append(vid)
            else:
                missing.append(vid)
        print("GALLERY", len(val_vids), "MISSING_VID", len(missing), missing[:5], flush=True)
        if missing:
            raise SystemExit("missing val videos in visual h5: %d" % len(missing))
        if vis_dim != 512:
            raise SystemExit("expected 512-d CLIP-B/32 visual, got %d" % vis_dim)

        frames = []
        lengths = []
        mean_vecs = []
        frame_norms = []
        for vid in val_vids:
            feat = np.asarray(fv[vid][...], dtype=np.float32)
            if feat.ndim == 1:
                feat = feat[None, :]
            if not np.isfinite(feat).all():
                raise SystemExit("non-finite visual feat for %s" % vid)
            lengths.append(feat.shape[0])
            frame_norms.append(float(np.linalg.norm(feat, axis=1).mean()))
            mean_vecs.append(feat.mean(axis=0))
            frames.append(feat)

    mean_vecs = np.stack(mean_vecs, axis=0)
    V_mean = l2_normalize(mean_vecs)
    Q = l2_normalize(enc)
    vid_index = {v: i for i, v in enumerate(val_vids)}
    gt = np.array([vid_index[v] for _, v, _ in caps], dtype=np.int64)
    print(
        "FRAME_LEN min/median/max",
        int(np.min(lengths)),
        int(np.median(lengths)),
        int(np.max(lengths)),
        "mean_frame_l2",
        float(np.mean(frame_norms)),
        "mean_video_l2_before_norm",
        float(np.linalg.norm(mean_vecs, axis=1).mean()),
        flush=True,
    )

    scores_mean = Q @ V_mean.T
    ranks_mean = ranks_from_scores(scores_mean, gt)
    mean_metrics = sumr(ranks_mean)
    pos_mean = float(scores_mean[np.arange(nq), gt].mean())
    all_mean = float(scores_mean.mean())
    print("MEAN_VIDEO", mean_metrics, "pos", pos_mean, "all", all_mean, flush=True)

    n_vid = len(val_vids)
    scores_max = np.empty((nq, n_vid), dtype=np.float32)
    for j, feat in enumerate(frames):
        F = l2_normalize(feat)
        scores_max[:, j] = (Q @ F.T).max(axis=1)
        if j % 200 == 0:
            print("max-frame video", j, "/", n_vid, flush=True)
    ranks_max = ranks_from_scores(scores_max, gt)
    max_metrics = sumr(ranks_max)
    pos_max = float(scores_max[np.arange(nq), gt].mean())
    all_max = float(scores_max.mean())
    print("MAX_FRAME", max_metrics, "pos", pos_max, "all", all_max, flush=True)

    rng = np.random.default_rng(SHUFFLE_SEED)
    perms = [rng.permutation(n_vid) for _ in range(N_SHUFFLE)]
    look_mean = rank_lookup(scores_mean)
    look_max = rank_lookup(scores_max)
    p_mean = permute_p(look_mean, gt, perms, mean_metrics["SumR"])
    p_max = permute_p(look_max, gt, perms, max_metrics["SumR"])
    print("P_MEAN", p_mean, flush=True)
    print("P_MAX", p_max, flush=True)

    retr_pass = bool(p_mean["p"] <= 0.005 or p_max["p"] <= 0.005)
    e2_0_pass = bool(encode_pass and retr_pass)
    out = {
        "clip_checkpoint": {
            "identified": False,
            "assumption": True,
            "library": "open_clip",
            "model": CLIP_MODEL,
            "pretrained": CLIP_PRETRAINED,
            "pretrained_used": pretrained_used,
            "local_pt": CLIP_PT if os.path.isfile(CLIP_PT) else None,
            "precision": "fp32",
            "device": str(device),
            "notes": checkpoint_note,
        },
        "data": {
            "captions": CAP_VAL,
            "visual_h5": VIS_H5,
            "n_queries": nq,
            "n_gallery": n_vid,
            "visual_dim": vis_dim,
            "visual_n_keys": vis_n,
            "frame_len_min": int(np.min(lengths)),
            "frame_len_median": int(np.median(lengths)),
            "frame_len_max": int(np.max(lengths)),
        },
        "encode_text_gate": {
            "n": nq,
            "min_cosine": min_cos,
            "mean_cosine": mean_cos,
            "n_below_0.9999": n_below,
            "n_nan": n_nan,
            "n_zero": n_zero,
            "pass": encode_pass,
        },
        "retrieval": {
            "n_queries": nq,
            "n_gallery": n_vid,
            "n_shuffles": N_SHUFFLE,
            "shuffle_seed": SHUFFLE_SEED,
            "mean_video": {
                **mean_metrics,
                **p_mean,
                "pos_cosine_mean": pos_mean,
                "all_pair_cosine_mean": all_mean,
            },
            "max_frame": {
                **max_metrics,
                **p_max,
                "pos_cosine_mean": pos_max,
                "all_pair_cosine_mean": all_max,
            },
        },
        "gates": {
            "encode_text": encode_pass,
            "frozen_retrieval": retr_pass,
            "e2_0_pass": e2_0_pass,
            "e2_train_allowed": e2_0_pass,
        },
        "seconds": time.time() - t0,
    }
    for path in (OUT_JSON, LOCAL_JSON):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print("WROTE", path, flush=True)
        except OSError as e:
            print("WRITE_FAIL", path, e, flush=True)
    print("E2_0_PASS", e2_0_pass, "TRAIN_ALLOWED", e2_0_pass, flush=True)
    return 0 if e2_0_pass else 2


if __name__ == "__main__":
    sys.exit(main())
