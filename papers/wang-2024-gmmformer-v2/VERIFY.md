# Verify: wang-2024-gmmformer-v2

| table/fig | metric | paper | ours | delta | match | command |
|---|---|---|---|---|---|---|
| T1-cha | R@1 | 2.5 | 2.4 | -0.1 | close | E1 `python main.py -d cha --gpu 4` seed 9527 I3D+RoBERTa, TC mean removed |
| T1-cha | R@5 | 8.6 | 8.8 | +0.2 | close | same |
| T1-cha | R@10 | 13.9 | 14.2 | +0.3 | close | same |
| T1-cha | R@100 | 53.2 | 53.0 | -0.2 | close | same |
| T1-cha | SumR | 78.2 | 78.4 | +0.2 | close | same |
| T1-cha (mean-bugged) | SumR | 78.2 | 73.5 | -4.7 | no | same hparams, `DyGMMBlock` mean overwrite still on |
| T1-act | R@1/5/10/100 / SumR | 8.9 / 27.1 / 40.2 / 78.7 / 154.9 | pending | — | skipped | not run yet; same TC-fixed method as E1 |
| T1-tvr | R@1 | 16.2 | 15.3 | -0.9 | close | `python main.py -d tvr --gpu 6` seed 9527 I3D+ResNet+RoBERTa, TC-fixed, 100 epoch no ES |
| T1-tvr | R@5 | 37.6 | 36.0 | -1.6 | close | same |
| T1-tvr | R@10 | 48.8 | 47.6 | -1.2 | close | same |
| T1-tvr | R@100 | 86.4 | 86.4 | 0 | yes | same |
| T1-tvr | SumR | 189.1 | 185.3 | -3.8 | no | same; Best epoch 98; ran full 100 epoch |

Evidence (E1): `/data/zhaopu/wang-2024-gmmformer-v2/tmp/i3d_tc_seed9527/results/charades/gmmformer_v2/log.txt` Best epoch 34, Early Stop epoch 45, exit 0. Mean baseline: `tmp/i3d_seed9527/.../log.txt` SumR 73.5.

CLIP run (wrong features) was Rsum 9.6 and is not comparable.

## What matched

- Official command, seed 9527, batch 128, 100 epoch / early stop 10, Adam 2e-4.
- After deleting `out = torch.mean(oo, dim = -1).squeeze()` in teacher `DyGMMBlock`, Charades SumR 78.4 vs paper 78.2 (R@1/5/10/100 all within 0.3).
- Causal check (E1): only that line changed; SumR 73.5 → 78.4 (+4.9). See `experiments/方向探索日志.md`.

## What did not, and why

- Visual is still A3PRVR `i3d_feat`, text is HuggingFace `roberta-large`, not the MS-SL release files. E1 shows this feature pair can reach paper SumR; it does not prove numerical identity with `i3d_rgb_lgi` + `roberta_charades_query_feat.hdf5`.
- Single seed. Epoch 0 had a brief duplicate process on GPU 4 (killed).
- T1-tvr MS-SL I3D+ResNet + RoBERTa, seed 9527, 100 epoch (no Early Stop). Best 15.3/36.0/47.6/86.4 / SumR **185.3** vs paper **189.1** (−3.8). R@100 matched. Evidence: `tmp/tvr_i3d_seed9527/results/tvr/gmmformer_v2/log.txt`.
- T1-act still training (2026-09-18).
- TVR E4 (I3D vis + CLIP text) Best SumR **178.6** is a comparison run, **not** Table 1.
