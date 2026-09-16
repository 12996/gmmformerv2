"""Extract CLIP token features for PRVR captions into teacher-style h5.

Stores [T, D] per cap_id (CLIP transformer tokens, padding stripped).
"""
import argparse
import os

import h5py
import numpy as np
import torch
import open_clip


def iter_captions(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cap_id, caption = line.split(" ", 1)
            yield cap_id, caption


def encode_batch(model, tokenizer, texts, device):
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        x = model.token_embedding(tokens)
        x = x + model.positional_embedding
        x = x.permute(1, 0, 2)
        x = model.transformer(x)
        x = x.permute(1, 0, 2)
        x = model.ln_final(x)
        x = x.float().cpu().numpy()
    lengths = (tokens != 0).sum(dim=1).cpu().numpy()
    return x, lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", nargs="+", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    device = torch.device(args.device)
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    items = []
    seen = set()
    for cap_file in args.captions:
        for cap_id, caption in iter_captions(cap_file):
            if cap_id in seen:
                continue
            seen.add(cap_id)
            items.append((cap_id, caption))
    print("captions", len(items), "model", args.model, "device", args.device)

    with h5py.File(args.dst, "w") as dst:
        for i in range(0, len(items), args.batch_size):
            batch = items[i : i + args.batch_size]
            texts = [t for _, t in batch]
            feats, lengths = encode_batch(model, tokenizer, texts, device)
            for j, (cap_id, _) in enumerate(batch):
                seq = feats[j, : int(lengths[j])]
                dst.create_dataset(cap_id, data=seq.astype(np.float32), compression="gzip", compression_opts=1)
            if i % (args.batch_size * 20) == 0:
                print("encoded", min(i + args.batch_size, len(items)), "/", len(items))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
