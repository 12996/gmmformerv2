# Read this pack before proposing anything

This is the **actual running code** from host `5090_1` (`/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src`) used for E1/E2/E3, plus arXiv PDF `2405.13824v1`.

Do **not** invent architecture from memory or from other PRVR papers. Cite class/function names from these files. One variable per next experiment. CLIP/hybrid numbers must not be written into paper Table 1 (I3D+RoBERTa).

## Paper vs our runs (Charades-STA PRVR, seed 9527, batch 128, Early Stop 10)

Paper Table 1 (I3D RGB LGI + RoBERTa 1024-d): 2.5 / 8.6 / 13.9 / 53.2 / SumR **78.2**

| id | visual | text | Best SumR | note |
|---|---|---|---|---|
| E1 | I3D 1024-d | RoBERTa 1024-d | **78.4** | TC mean-overwrite removed |
| E2 | CLIP-B/32 512-d (`charades_clip_L14.h5`, actually B/32) | CLIP token seq @ `text_projection` | **68.0** | not Table 1 |
| E3 | same CLIP vis | E1 RoBERTa h5, `q_feat_size=1024` | **65.4** | not Table 1; text swap did not help |

Missing cell: I3D vis + CLIP text.

## What this code actually is

- Teacher zip + patches, not a clean official clone. Official repo is I3D+RoBERTa.
- `Configs/cha.py` currently E3: `visual_feature=clip`, `visual_feat_dim=512`, `q_feat_size=1024`.
- `Datasets/builder.py`: `visual_feature=='clip'` loads `FeatureData/charades_clip_L14.h5`; E3 text path is `TextData/roberta_charades_query_feat.hdf5`.
- `Models/gmmformerV2/model_components.py` DyGMMBlock: official weighted `torch.sum(oo * weight...)`; teacher `torch.mean(oo)` overwrite **deleted**.
- CLIP text extraction (not in vendor): `ln_final(x) @ text_projection`, keep token sequence, strip pad. File: `extract_clip_text_proj.py`.
- Hidden size 384, dual clip/frame branches, `clip_scale_w=0.7`, `frame_scale_w=0.3`.

## Your job

1. Read the PDF (method + Table 1 Charades + feature protocol) and these sources.
2. Explain E2 68.0 vs E1 78.4 using **this** model (TC-GMM, query pooling, dual branch, feature dims), not a literature dump.
3. Lock **one** next experiment: what file/flag changes, what stays fixed, how to read Best SumR vs 68.0 / 78.2 / 78.4.
4. Reply with exactly 6 numbered one-line consensus items. Last line is the next action.
