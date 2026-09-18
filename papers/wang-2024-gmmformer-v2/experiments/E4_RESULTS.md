# E4 results: I3D visual + CLIP-B/32 projected token text

Protocol: GMMFormer v2, TC weighted sum (no mean overwrite), seed 9527, CLIP text = `ln_final @ text_projection` token sequences (not zip EOT). **Not Table 1.**

Host: `5090_1` `/data/zhaopu/wang-2024-gmmformer-v2`

| dataset | I3D+RoBERTa (ours / paper) | E4 I3D+CLIP text | E4 Best R@1/5/10/100 | dir |
|---|---|---|---|---|
| Charades-STA | 78.4 / 78.2 | **80.1** | 2.4 / 9.3 / 15.5 / 52.8 | `tmp/i3d_cliptext_e4_seed9527` |
| ActivityNet | 155.0 / 154.9 | **155.2** | 9.0 / 27.6 / 40.2 / 78.4 | `tmp/act_i3d_cliptext_e4_seed9527` |
| TVR | 185.3 / 189.1 | **178.6** | 14.6 / 34.4 / 45.7 / 83.9 | `tmp/tvr_i3d_cliptext_e4_seed9527` |

TVR E4 used official tvr hparams except `q_feat_size=512` (lr 3e-4, sft 0.09). Charades E4 used cha hparams (lr 2e-4, sft 0.6).

Follow-up TVR E4 **lr 2e-4 only** (`tmp/tvr_i3d_cliptext_e4_lr2e-4_seed9527`): Best epoch 76 **13.7 / 33.5 / 44.8 / 84.3 / SumR 176.3**. Early Stop 87. **No gain** vs original E4 178.6.

Follow-up TVR E4 **sft_factor 0.6**, lr 3e-4 (`tmp/tvr_i3d_cliptext_e4_sft06_seed9527`): Best epoch 99 **14.2 / 34.4 / 45.6 / 84.6 / SumR 178.8**. +0.2 vs original E4 178.6; still −6.5 vs I3D 185.3. Hparam search closed per GPT-6.
