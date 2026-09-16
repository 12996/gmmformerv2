"""Convert A3PRVR hdf5 (video_id/clip_feat) into teacher GMMFormer_v2 CLIP h5.

Teacher loader:
  visual: h5[video_id][...]  with ActivityNet keys stripped of leading 'v_'
  text:   h5[cap_id][...]    token sequence [T, D]
"""
import argparse
import os

import h5py
import numpy as np


def convert_visual(src_path, dst_path, collection):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    n = 0
    dim = None
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key in src.keys():
            grp = src[key]
            if "clip_feat" not in grp:
                raise KeyError("missing clip_feat in %s" % key)
            feat = np.asarray(grp["clip_feat"][...], dtype=np.float32)
            if feat.ndim == 1:
                feat = feat[None, :]
            dim = int(feat.shape[-1])
            out_key = key[2:] if collection == "activitynet" and key.startswith("v_") else key
            dst.create_dataset(out_key, data=feat, compression="gzip", compression_opts=1)
            n += 1
    print("visual videos=%d dim=%s -> %s" % (n, dim, dst_path))
    return dim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--collection", required=True, choices=["activitynet", "charades", "tvr"])
    args = p.parse_args()
    convert_visual(args.src, args.dst, args.collection)


if __name__ == "__main__":
    main()
