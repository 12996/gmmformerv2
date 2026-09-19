"""TVR EOT CLIP text: same ln_final @ text_projection as E4 tokens, one real EOT vector.

Stores (1, 512) per caption. Does not use zip clip_ViT_B_32_tvr_query_feat.hdf5.
"""
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
        eot = tokens.argmax(dim=-1)
        eot_tok = tokens[torch.arange(tokens.shape[0], device=tokens.device), eot]
        if (eot_tok == 0).any():
            raise RuntimeError("EOT index landed on PAD")
        pooled = x[torch.arange(x.shape[0], device=x.device), eot]
        pooled = pooled.float().cpu().numpy()
    eot = eot.cpu().numpy()
    eot_tok = eot_tok.cpu().numpy()
    return pooled, eot, eot_tok


def load_model(model_name, pretrained, device):
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
    print("captions", len(items), "eot", True, "device", device)
    sample_printed = False
    with h5py.File(args.dst, "w") as dst:
        for i in range(0, len(items), args.batch_size):
            batch = items[i : i + args.batch_size]
            feats, eot_idx, eot_tok = encode_batch(model, tokenizer, [t for _, t in batch], device)
            for j, (cap_id, _) in enumerate(batch):
                vec = np.asarray(feats[j], dtype=np.float32)
                if vec.ndim == 1:
                    vec = vec.reshape(1, -1)
                if vec.shape != (1, 512):
                    raise RuntimeError("bad eot shape %s %s" % (cap_id, vec.shape))
                dst.create_dataset(cap_id, data=vec, compression="gzip", compression_opts=1)
                if not sample_printed:
                    print("SAMPLE_ID", cap_id)
                    print("SAMPLE_SHAPE", vec.shape)
                    print("SAMPLE_EOT_IDX", int(eot_idx[j]), "SAMPLE_EOT_TOK", int(eot_tok[j]))
                    sample_printed = True
            if i % (args.batch_size * 10) == 0:
                print("encoded", min(i + args.batch_size, len(items)), "/", len(items))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
