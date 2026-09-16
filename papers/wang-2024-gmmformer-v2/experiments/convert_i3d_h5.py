"""Export A3PRVR i3d_feat groups into a flat video_id -> (T,1024) h5."""
import argparse
import os

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    n = 0
    with h5py.File(args.src, "r") as src, h5py.File(args.dst, "w") as dst:
        for key in src.keys():
            feat = np.asarray(src[key]["i3d_feat"][...], dtype=np.float32)
            if feat.ndim == 1:
                feat = feat[None, :]
            dst.create_dataset(key, data=feat, compression="gzip", compression_opts=1)
            n += 1
            if n % 500 == 0:
                print("i3d", n)
    print("I3D_OK videos", n, "->", args.dst)


if __name__ == "__main__":
    main()
