# GMMFormer v2: An Uncertainty-aware Framework for Partially Relevant Video Retrieval

- slug: wang-2024-gmmformer-v2
- year: 2024
- venue: preprint (arXiv cs.CV, 22 May 2024; marked "Under review")
- links:
  - arxiv: https://arxiv.org/abs/2405.13824
  - pdf: https://arxiv.org/pdf/2405.13824v1
  - local pdf: `agent_planning/2405.13824v1.pdf`
  - official code: https://github.com/huangmozhi9527/GMMFormer_v2
  - feature source (MS-SL): https://github.com/HuiGuanLab/ms-sl
- license: unspecified in the official repo (no LICENSE file); arXiv preprint uses arXiv.org perpetual non-exclusive license
- hardware (from paper): single Nvidia RTX 3080Ti; PyTorch; Adam; batch 128; 100 epochs; early stop 10. Table 2 runtime also measured on 3080Ti. No multi-GPU claim.

## Problem

Partially relevant video retrieval (PRVR): given a text query, rank untrimmed videos that contain a relevant moment. Moment locations are not annotated at train time, so clip extent and text-clip correspondence are uncertain. Explicit sliding-window clip models are redundant; implicit static aggregation (GMMFormer) is inflexible for unexpected moment-to-video ratios; text queries of one video often collapse onto a few homogeneous clips (semantic collapse).

## Method

- Frame branch plus clip branch (uniform sample `Mc=32` clips, cap `Mf=128` frames); RoBERTa query encoder with attention pooling.
- Replace GMMFormer static pooling with TC-GMMBlock: eight Gaussian-constrained Transformer blocks plus a temporal consolidation module that learns per-timestep aggregation weights over multi-scale features.
- Revamped query-diverse loss: focal factor `(1+cos(qi,qj))^gamma` down-weights already-separated text pairs.
- Optimal matching loss: Hungarian assignment of queries to clips, then `(1-cos)` on the matching.
- Total loss `L = L_basic + λd L_div + λo L_om`. Retrieve with `S = αf Sf + αc Sc` (max cosine on frames/clips).

## Official code

- repo: https://github.com/huangmozhi9527/GMMFormer_v2
- commit (pin if possible): `5bacac96f77408c2bef27c25637e2cf67fb062b9` (origin/main as cloned in the lab zip)
- claimed train command:
  - `cd src && python main.py -d tvr --gpu 0`
  - `cd src && python main.py -d act --gpu 0`
  - `cd src && python main.py -d cha --gpu 0`
- claimed eval command: `python main.py -d {tvr,act,cha} --gpu 0 --eval --resume <best.ckpt>` (official `main.py` evaluates the **test** loaders)
- README env pin: Python 3.9, `pytorch==1.9.0`, `cudatoolkit=11.3` — this pin is for 3080Ti, not RTX 5090
- features: I3D (ActivityNet / Charades-STA) and I3D+ResNet152 (TVR) plus RoBERTa query features from MS-SL
  - Google Drive: https://drive.google.com/drive/folders/11dRUeXmsWU25VMVmeuHc9nffzmZhPJEj
  - Baidu: https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4
- checkpoints (Baidu, pwd `9527`): TVR / ActivityNet / Charades-STA links in the official README
- lab zip `agent_planning/gmmformerv2.zip` is **not** a drop-in official checkout: CLIP-L/14 features, different `act.py` / `Datasets/builder.py` / `data_provider.py`, eval on val not test. Do not put that zip in `vendor/` as the official tree.

## Tables to match

| id | what | paper number | split | notes |
|---|---|---|---|---|
| T1-act | ActivityNet Captions R@1/5/10/100 / SumR | 8.9 / 27.1 / 40.2 / 78.7 / 154.9 | PRVR val/test as in MS-SL | README expected numbers match T1; official `--eval` uses test loaders |
| T1-tvr | TVR R@1/5/10/100 / SumR | 16.2 / 37.6 / 48.8 / 86.4 / 189.1 | MS-SL TVR split | I3D+ResNet152 3072-D + 768-D RoBERTa |
| T1-cha | Charades-STA R@1/5/10/100 / SumR | 2.5 / 8.6 / 13.9 / 53.2 / 78.2 | official Charades-STA partition | I3D RGB LGI + 1024-D RoBERTa |
| T2 | FLOPs / params / runtime / memory vs MS-SL and GMMFormer | 5.43G / 32.27M / 6.73 ms / 61.57M | 2500 TVR videos, 3080Ti | efficiency table; hardware-bound, not first verify target |
| T3 | ActivityNet ablation FB / TC-GB / QDL / OM | full model SumR 154.9 | ActivityNet | optional after T1 |
| T5 | Gaussian-block count SumR | GMMFormer v2 8 blocks 154.9 | ActivityNet | optional |
| T6 | plugin L_div / L_om on MS-SL and DL-DKD | MS-SL 146.6 / DL-DKD 152.2 | ActivityNet | needs other codebases |

## Adaptation target

- problem: TBD

## Status

verify
