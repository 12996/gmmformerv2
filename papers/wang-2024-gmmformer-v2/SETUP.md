# Setup: wang-2024-gmmformer-v2 (CLIP fork)

- official repo: https://github.com/huangmozhi9527/GMMFormer_v2 (teacher zip is a CLIP fork of commit `5bacac96`)
- vendor path: `/home/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2` (5090). Not the vanilla I3D GitHub tree.
- commit: `5bacac96f77408c2bef27c25637e2cf67fb062b9` plus teacher CLIP edits
- env (conda/venv/docker + CUDA): 5090 conda `prvr-clip`, Python 3.10.21, `torch 2.9.1+cu128` (`sm_120`). README 1.9+cu11.3 was not used.
- data location: `/home/zhaopu/wang-2024-gmmformer-v2/data/prvr/charades/`
  - visual: `FeatureData/charades_clip_L14.h5` (name kept; tensors are CLIP-B/32, 6672 videos, dim 512)
  - text: `TextData/charades_clip_L14.h5` (CLIP-B/32 token features, 16128 captions)
  - captions from A3PRVR HF: train 12408 / val 3720
- first official command: `cd vendor/GMMFormer_v2/src && python main.py -d cha --gpu 0`
- started: 2026-09-14
- finished: blocked
- result: fail
- blockers: all 8x RTX 5090 occupied by two vLLM jobs (`697259`, `57610`). Leftover VRAM ~1.5–3.2 GB. Official batch 128 CUDA OOM. CPU dataloader smoke passed. GPU batch-8 one-step train passed (loss 2.05, peak 507 MB). Full `python main.py -d cha --gpu 0` not run.

## Notes

- User chose the teacher CLIP line, not paper Table 1 (I3D/RoBERTa). Teacher ActivityNet CLIP-L/14 dump was not on 5090. First run uses A3PRVR Charades `clip_feat` (512-d B/32) plus extracted CLIP-B/32 text. Incomparable to paper SumR 78.2 and to the zip ActivityNet CLIP-L/14 log (Rsum 197.1).
- Path-only edits on 5090: `Configs/cha.py` `root`/`data_root`/`visual_feature=clip`/`q_feat_size=512`; `act.py` data paths; `Datasets/builder.py` val videos use `Dataset4PRVR`+`collate_train` because `validations.py` needs caption fields.
- CPU smoke: `ntrain=5338 nctx=1334 nq=3720`; one train batch shapes OK (`clip_video_features (128,32,512)`).
- GPU one-batch train: OOM on physical GPU 0 (~3.1 GB free) and GPU 6 (~1.5 GB free).
- Next: when one 5090 is free, rerun `python main.py -d cha --gpu 0` then `paper-verify` against Charades CLIP numbers (not Table 1).

## 5090_1 (P6X8G, 202.207.1.21) env — 2026-09-15

- vendor: `/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2` (same teacher CLIP zip)
- conda prefix: `/data/zhaopu/wang-2024-gmmformer-v2/conda` (Python 3.10.21)
- torch: `2.9.1+cu128`, CUDA 12.8, `cc 12.0`, device RTX 5090 (`CUDA_OK`)
- deps: h5py 3.16.0, numpy 2.2.6, einops 0.8.2, PyYAML 6.0.3, tqdm 4.70.1, scikit-learn 1.7.2, scipy 1.15.3, matplotlib 3.10.9, ipdb 0.13.13, easydict 1.13
- data: `/data/zhaopu/wang-2024-gmmformer-v2/data/prvr/charades/`
  - visual h5 6672 videos, example `(92, 512)` float32
  - text h5 16128 captions, example `(9, 512)`
  - captions train 12408 / val 3720
- CLIP `cha.py`: `visual_feature=clip`, `q_feat_size=512`, `visual_feat_dim=512`, `batchsize=128`
- `Datasets/builder.py` val videos patched to `Dataset4PRVR`+`collate_train`
- pip via laptop reverse tunnel `ssh -R 17897:127.0.0.1:7897 5090_1`; server proxy `http://127.0.0.1:17897`
- first official command **finished** 2026-09-15 18:16–18:48. Five independent `python main.py -d cha --gpu {0,4,5,6,7}` seeds 9527–9531, all `exit=0`, all `Early Stop`. No Traceback/OOM.
- best val (not paper Table 1): seed9527 R@1 0.0 Rsum 9.6; 9528 0.1/9.6; 9529 0.1/9.7; 9530 0.1/9.4; 9531 0.1/10.1. Loss stayed ~1.53. Runtime OK, retrieval numbers are not.
- logs: `/data/zhaopu/wang-2024-gmmformer-v2/logs/runs/seed*_gpu*/` plus `tmp/seed*/results/charades/gmmformer_v2/log.txt`
- evidence: seed9527 `tmp/seed9527/results/charades/gmmformer_v2/log.txt` `Early Stop !!!` at 18:48:14, `Best: R@1: 0.0 ... Rsum: 9.6`

## 5090_1 I3D+RoBERTa rerun — 2026-09-15

CLIP run was random: frozen CLIP mean-pool Rsum 8.55 (pos cosine = all-pair mean). Visual was A3PRVR `clip_feat`; text was open_clip `ln_final` tokens without `text_projection`. Paper T1-cha is I3D RGB + 1024-d RoBERTa, not CLIP.

Fixes: export `i3d_feat` from A3PRVR hdf5; extract `roberta-large` token h5; official 32-clip / 128-frame sampling; `cha.py` `visual_feature=i3d_rgb_lgi`, `q_feat_size=1024`.

Retrain finished Early Stop, exit 0. seed9527 Best R@1 2.4 R@5 7.6 R@10 13.3 R@100 50.2 Rsum 73.5 (epoch 48, 21:32). seed9528 Rsum 74.3. Paper T1-cha SumR 78.2. See `VERIFY.md`.
