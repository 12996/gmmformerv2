# E4 freeze (do not overwrite)

I3D visual + CLIP-B/32 **token-sequence** text, TC-fixed GMMFormer v2, seed 9527.

## Results (not Table 1)

| dataset | I3D+RoBERTa | E4 |
|---|---|---|
| Charades | 78.4 | **80.1** |
| ActivityNet | 155.0 | **155.2** |
| TVR | 185.3 | **178.6** (sft=0.6: 178.8; EOT: 165.3) |

TVR controlled baseline = **178.6** (`tmp/tvr_i3d_cliptext_e4_seed9527`, lr 3e-4, sft 0.09).

## Remote (5090_1)

- Charades: `tmp/i3d_cliptext_e4_seed9527`
- Act: `tmp/act_i3d_cliptext_e4_seed9527`
- TVR: `tmp/tvr_i3d_cliptext_e4_seed9527`
- Text h5: `data/prvr/tvr/TextData/tvr_clip_B32_proj.h5` (sequences). Do not replace with EOT h5 for this freeze.

## Scripts in this folder

Copies of the E4 pipeline. Structure experiments must use **new** `tmp/` names and new patches, not edit these copies in place as the only source.
