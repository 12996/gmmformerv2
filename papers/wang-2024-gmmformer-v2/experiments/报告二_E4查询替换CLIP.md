# 报告二：E4 查询替换为 CLIP

来源：GPT-6（ChatGPT），对话 https://chatgpt.com/c/6aaa72b3-afe8-83ea-8ad2-1c1257b4aeef  
对照是各数据集上**本次实测的 RoBERTa 基线**，不是用论文数字替代本地基线。E4 及后续**不并入**论文 Table 1。单 seed 9527。

## 1. 实验目的

保持各数据集原有 I3D 系列视觉输入和 GMMFormer v2 主体流程，将查询从 RoBERTa 换成 CLIP-B/32 的投影 token 序列，检验该替换能否改善检索，以及效果是否依赖数据集。

## 2. 实验设定

视觉不变：Charades / ActivityNet 用 I3D；TVR 用 3072 维 I3D＋ResNet。查询为 caption 抽取的 CLIP-B/32 token 序列：`ln_final` 后逐 token 乘 `text_projection`，512 维。E4 主实验**不是**单 EOT。

保留 `query_input_proj`、查询侧 BertAttention、`modular_vector_mapping`；hidden 384。视觉 DyGMMBlock/TC 已修为官方加权 sum；双分支 32/128，融合 0.7/0.3。E4 主实验不新增模块。BertAttention 旁路是后续独立结构消融。

代码冻结于 `e4_freeze/`。TVR 原始 E4 目录 `tmp/tvr_i3d_cliptext_e4_seed9527` 对应 **178.6**。

Δquery = SumR(E4) − SumR(同数据集 RoBERTa 基线)。

## 3. 三个数据集的查询替换

| 数据集 | 查询 | R@1 | R@5 | R@10 | R@100 | SumR | Δquery |
|---|---|---|---|---|---|---|---|
| Charades | RoBERTa | 2.4 | 8.8 | 14.2 | 53.0 | 78.4 | — |
| Charades | E4 CLIP token | 2.4 | 9.3 | 15.5 | 52.8 | **80.1** | **+1.7** |
| ActivityNet | RoBERTa | 9.2 | 27.3 | 39.8 | 78.7 | 155.0 | — |
| ActivityNet | E4 CLIP token | 9.0 | 27.6 | 40.2 | 78.4 | **155.2** | **+0.2** |
| TVR | RoBERTa | 15.3 | 36.0 | 47.6 | 86.4 | 185.3 | — |
| TVR | E4 CLIP token | 14.6 | 34.4 | 45.7 | 83.9 | **178.6** | **−6.7** |

Charades 正向（主要是 R@5/R@10；R@1 不变，R@100 略降）。ActivityNet 基本持平。TVR 四项 Recall 均低于 RoBERTa，下降明确。

## 4. TVR 后续单变量（均相对原始 E4=178.6，不叠加）

| 实验 | 唯一改动 | SumR | vs E4 | vs RoBERTa 185.3 |
|---|---|---|---|---|
| 原始 E4 | 无 | 178.6 | 0 | −6.7 |
| lr 2e-4 | 只改学习率 | 176.3 | −2.3 | −9.0 |
| sft 0.6 | 只改 TC 温度 | 178.8 | +0.2 | −6.5 |
| EOT | token 序列→同源 EOT | 165.3 | −13.3 | −20.0 |
| 旁路 query BertAttention | 只旁路再编码块 | 174.7 | −3.9 | −10.6 |

lr/sft 不是补回 TVR 缺口的有效办法。EOT 比 token 序列差 13.3，否定「改成 EOT 即可改善」。旁路 BertAttention 也无收益（−3.9）。

## 5. 结论与局限

RoBERTa→CLIP 查询替换**不是**跨数据集统一有效的改进：Charades +1.7，Act +0.2，TVR −6.7。核心发现是**数据集依赖**，不是 CLIP 普遍更强或更弱。

E4 是查询特征组合实验，没有提出新的 GMMFormer 结构。Charades 的 80.1 不能填进 Table 1，也不能单凭该数字当方法创新。单 seed。未单独分离预训练表征、分词截断、维度适配与训练的影响，不能把 TVR 缺口直接说成「512 维不足」或「CLIP 语义不足」。没有找到有效修复，不等于 178.8 是上限。
