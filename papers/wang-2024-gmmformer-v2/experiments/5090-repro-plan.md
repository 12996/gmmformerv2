# 5090 复现计划（intake 记录，尚未 setup）

Observed 2026-09-14 on host `5090`. Not a `SETUP.md`. Do not treat numbers below as verify results.

## Two code trees (do not mix)

| | Official paper repo | Teacher zip `agent_planning/gmmformerv2.zip` |
|---|---|---|
| Origin | `github.com/huangmozhi9527/GMMFormer_v2` @ `5bacac96` | clone of that commit, then edited |
| Features | MS-SL I3D / I3D+ResNet + RoBERTa hdf5 (`BigFile`) | CLIP ViT-L/14 h5 (`activitynet_clip_L14.h5`) |
| ActivityNet `q_feat_size` | 1024 | 768 |
| ActivityNet `visual_feature` | `i3d` | `clip` |
| `get_datasets` return | train + val + **test** loaders (6-tuple) | train + val only (4-tuple) |
| `--eval` | **test** set | val set |
| Paper Table 1 comparable | yes, after MS-SL features | no (CLIP backbone; val metrics) |
| Lab log peak | n/a | ActivityNet Best Rsum **197.1** (CLIP, 2026-06-13) vs paper 154.9 |

Vendor for `repro-setup` must be the official clone, not the zip. Keep the zip as a lab fork for a later CLIP/A3PRVR adapt.

## 5090 snapshot

- 8x RTX 5090 32 GB, driver 595.84, CUDA 13.2 / toolkit 13.0
- All 8 GPUs occupied by two vLLM jobs (`guojinx+` 4-way TP, `qiaohai+` 4-way TP, both >3 days). Free VRAM ~1.5–3.2 GB/card. **Do not start train/eval until a card is free.**
- `/data` is 100% full. `/home` has ~524 GB free. Put code + features under `/home/zhaopu/wang-2024-gmmformer-v2/`.
- No PRVR / ActivityNet / GMMFormer tree on this account yet.
- Existing torch that can talk to 5090: conda `z-image` / `gate_attention` = `torch 2.9.1+cu128` with `sm_120`. `fmtravelplanner` is CPU-side 3.9, not for this paper.
- Official README `pytorch==1.9.0 cudatoolkit=11.3` **will not run** on Blackwell. Create env `prvr` with Python 3.9 or 3.10 and `torch 2.9.1+cu128` (or newer cu128 build), then `pip install -r requirements.txt` from Tsinghua.

## Data layout (official)

After unzipping MS-SL dumps, point every `src/Configs/{tvr,act,cha}.py` `root` and `data_root` at the 5090 paths:

```
$data_root/<collection>/
  TextData/
    <collection>train.caption.txt
    <collection>val.caption.txt
    <collection>test.caption.txt
    roberta_<collection>_query_feat.hdf5
  FeatureData/<visual_feature>/
    video2frames.txt
    (MS-SL BigFile feature shards)
```

`visual_feature`: ActivityNet `i3d`, TVR `i3d_resnet`, Charades `i3d_rgb_lgi`.

Teacher CLIP layout (only if adapting the zip later):

```
$data_root/activitynet/FeatureData/activitynet_clip_L14.h5
$data_root/activitynet/TextData/activitynet_clip_L14.h5
```

## First official command (when GPU + data exist)

Prefer eval if a paper ckpt is downloaded:

```
cd /home/zhaopu/wang-2024-gmmformer-v2/vendor/src
python main.py -d act --gpu 0 --eval --resume /path/to/official_activitynet_best.ckpt
```

Else one ActivityNet train (README quickstart; ~hours on 5090, not multi-day):

```
python main.py -d act --gpu 0
```

GitHub / Drive from 5090 still need the laptop reverse tunnel in `papers/hao-2025-formal-travel-planner/5090-next.md` (`17897`, not server `7897`).

## Blockers to record at setup

1. GPU occupancy (vLLM).
2. `/data` full; use `/home`.
3. README CUDA 11.3 pin vs 5090.
4. Features are not in the zip; must fetch MS-SL (Drive/Baidu).
5. Teacher `best.ckpt` is CLIP ActivityNet (~366 MB in the zip) and cannot match Table 1.
