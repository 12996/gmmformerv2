#!/usr/bin/env python3
"""Extract frozen OpenAI CLIP ViT-L/14 frame features. Does not overwrite B/32 h5."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image


ROOT = "/data/zhaopu/wang-2024-gmmformer-v2"
SRC_H5 = os.path.join(ROOT, "data/prvr/charades/FeatureData/charades_clip_L14.h5")
DST_H5 = os.path.join(ROOT, "data/prvr/charades/FeatureData/charades_clip_vitl14_224.h5")
CLIP_PT = os.path.join(ROOT, ".cache/clip/ViT-L-14.pt")


def list_videos(video_dir: str) -> dict:
    out = {}
    for p in Path(video_dir).rglob("*"):
        if p.suffix.lower() in {".mp4", ".avi", ".mkv", ".webm"}:
            out[p.stem] = str(p)
    return out


def sample_indices(n_frames: int, t: int) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("empty video")
    if t <= 1:
        return np.array([min(n_frames - 1, n_frames // 2)], dtype=np.int64)
    return np.linspace(0, n_frames - 1, num=t, dtype=np.int64)


def load_frames_cv2(path: str, idxs: np.ndarray) -> list:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("open fail %s" % path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    want = set(int(i) for i in idxs)
    max_i = max(want)
    i = 0
    while i <= max_i:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    if len(frames) != len(idxs):
        # fallback random-access
        cap = cv2.VideoCapture(path)
        frames = []
        for j in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("seek fail %s idx=%s n=%s" % (path, j, n))
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--src-h5", default=SRC_H5)
    ap.add_argument("--dst-h5", default=DST_H5)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    if os.path.abspath(args.dst_h5) == os.path.abspath(args.src_h5):
        raise SystemExit("refuse to overwrite src h5")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import open_clip
    from open_clip.openai import load_openai_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.isfile(CLIP_PT):
        model = load_openai_model(CLIP_PT, device=device)
    else:
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        model = model.to(device)
    model.eval()
    preprocess = open_clip.image_transform(
        model.visual.image_size, is_train=False, mean=None, std=None
    )

    videos = list_videos(args.video_dir)
    os.makedirs(os.path.dirname(args.dst_h5), exist_ok=True)
    tmp = args.dst_h5 + ".partial"
    n_ok = 0
    with h5py.File(args.src_h5, "r") as src, h5py.File(tmp, "w") as dst:
        keys = list(src.keys())
        print("N_SRC", len(keys), "N_VID", len(videos), flush=True)
        missing = [k for k in keys if k not in videos]
        if missing:
            raise SystemExit("MISSING_VIDEOS %s e.g. %s" % (len(missing), missing[:5]))
        for i, key in enumerate(keys):
            t = int(np.asarray(src[key]).shape[0])
            path = videos[key]
            import cv2

            cap = cv2.VideoCapture(path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            idxs = sample_indices(n_frames, t)
            pil = load_frames_cv2(path, idxs)
            feats = []
            with torch.no_grad():
                for s in range(0, len(pil), args.batch):
                    batch = torch.stack([preprocess(im) for im in pil[s : s + args.batch]]).to(device)
                    f = model.encode_image(batch)
                    feats.append(f.float().cpu().numpy())
            feat = np.concatenate(feats, axis=0).astype(np.float32)
            if feat.shape != (t, 768):
                raise RuntimeError("bad feat %s %s t=%s" % (key, feat.shape, t))
            dst.create_dataset(key, data=feat, compression="gzip", compression_opts=1)
            n_ok += 1
            if n_ok % 20 == 0 or n_ok == 1:
                print("PROGRESS", n_ok, "/", len(keys), key, feat.shape, flush=True)
    os.replace(tmp, args.dst_h5)
    print("EXTRACT_OK", n_ok, args.dst_h5, flush=True)


if __name__ == "__main__":
    main()
