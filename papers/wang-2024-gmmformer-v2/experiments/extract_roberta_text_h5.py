"""Extract RoBERTa-large token features (1024-d) for PRVR captions."""
import argparse
import os

import h5py
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer


def iter_captions(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cap_id, caption = line.split(" ", 1)
            yield cap_id, caption


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", nargs="+", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--model", default="roberta-large")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-len", type=int, default=30)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tok = RobertaTokenizer.from_pretrained(args.model)
    model = RobertaModel.from_pretrained(args.model).to(device).eval()

    items = []
    seen = set()
    for cap_file in args.captions:
        for cap_id, caption in iter_captions(cap_file):
            if cap_id in seen:
                continue
            seen.add(cap_id)
            items.append((cap_id, caption))
    print("captions", len(items), "model", args.model, "device", device)

    with h5py.File(args.dst, "w") as dst, torch.no_grad():
        for i in range(0, len(items), args.batch_size):
            batch = items[i : i + args.batch_size]
            texts = [t for _, t in batch]
            enc = tok(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc).last_hidden_state.float().cpu().numpy()
            lengths = enc["attention_mask"].sum(dim=1).cpu().numpy()
            for j, (cap_id, _) in enumerate(batch):
                seq = out[j, : int(lengths[j])]
                dst.create_dataset(cap_id, data=seq.astype(np.float32), compression="gzip", compression_opts=1)
            if i % (args.batch_size * 20) == 0:
                print("encoded", min(i + args.batch_size, len(items)), "/", len(items))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
