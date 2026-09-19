"""E2: CLIP token features WITH text_projection; keep sequence, not single EOT."""
import argparse
import os

import h5py
import numpy as np
import open_clip
import torch


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
        x = model.transformer(x, attn_mask=model.attn_mask)
        x = model.ln_final(x)
        x = x @ model.text_projection
        x = x.float().cpu().numpy()
    lengths = (tokens != 0).sum(dim=1).cpu().numpy()
    return x, lengths


def load_model(model_name, pretrained, device):
    # Prefer the local OpenAI .pt used by E2-0. create_model_and_transforms("openai")
    # may try HuggingFace and hang when the network is unreachable.
    clip_pt = os.environ.get("CLIP_PT", "")
    pt = clip_pt if clip_pt and os.path.isfile(clip_pt) else (pretrained if os.path.isfile(pretrained) else "")
    if pt:
        from open_clip.openai import load_openai_model
        print("load_openai_model", pt)
        model = load_openai_model(name=pt, precision="fp32", device=device)
    else:
        print("create_model_and_transforms", model_name, pretrained)
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model.eval(), tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", nargs="+", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, tokenizer = load_model(args.model, args.pretrained, device)
    items = []
    seen = set()
    for cap_file in args.captions:
        for cap_id, caption in iter_captions(cap_file):
            if cap_id in seen:
                continue
            seen.add(cap_id)
            items.append((cap_id, caption))
    print("captions", len(items), "proj", True, "device", device)
    with h5py.File(args.dst, "w") as dst:
        for i in range(0, len(items), args.batch_size):
            batch = items[i : i + args.batch_size]
            feats, lengths = encode_batch(model, tokenizer, [t for _, t in batch], device)
            for j, (cap_id, _) in enumerate(batch):
                seq = feats[j, : int(lengths[j])]
                dst.create_dataset(cap_id, data=seq.astype(np.float32), compression="gzip", compression_opts=1)
            if i % (args.batch_size * 10) == 0:
                print("encoded", min(i + args.batch_size, len(items)), "/", len(items))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
