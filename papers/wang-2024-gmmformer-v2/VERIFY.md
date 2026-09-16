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
| T1-tvr | R@1/5/10/100 / SumR | 16.2 / 37.6 / 48.8 / 86.4 / 189.1 | pending | — | skipped | not run yet; I3D+ResNet 3072-d + RoBERTa 768-d |

Evidence (E1): `/data/zhaopu/wang-2024-gmmformer-v2/tmp/i3d_tc_seed9527/results/charades/gmmformer_v2/log.txt` Best epoch 34, Early Stop epoch 45, exit 0. Mean baseline: `tmp/i3d_seed9527/.../log.txt` SumR 73.5.

CLIP run (wrong features) was Rsum 9.6 and is not comparable.

## What matched

- Official command, seed 9527, batch 128, 100 epoch / early stop 10, Adam 2e-4.
- After deleting `out = torch.mean(oo, dim = -1).squeeze()` in teacher `DyGMMBlock`, Charades SumR 78.4 vs paper 78.2 (R@1/5/10/100 all within 0.3).
- Causal check (E1): only that line changed; SumR 73.5 → 78.4 (+4.9). See `experiments/方向探索日志.md`.

## What did not, and why

- Visual is still A3PRVR `i3d_feat`, text is HuggingFace `roberta-large`, not the MS-SL release files. E1 shows this feature pair can reach paper SumR; it does not prove numerical identity with `i3d_rgb_lgi` + `roberta_charades_query_feat.hdf5`.
- Single seed. Epoch 0 had a brief duplicate process on GPU 4 (killed).
- T1-act and T1-tvr not run yet (2026-09-17). Same TC-fixed method will be applied; features not on 5090_1 at last check.
