# FastGCN 的 PyTorch 实现
---

本仓库实现了论文 [FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://arxiv.org/abs/1801.10247) 中的 FastGCN 算法。为便于复现与理解，文档中对训练与实验实现上的差异做了透明说明。

## 安装与硬件

请按 [requirements.txt](requirements.txt) 中的命令安装依赖。项目最初假设使用 PyTorch `1.12.1`（文档编写时的最新版本）；历史版本可参考 [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)。

实验使用的硬件为 Quadro RTX 5000（16 GB 显存）。

## 训练细节说明

下面列出相对原始 FastGCN 论文的若干实现差异：

1. **邻居采样方式**：FastGCN 可能在逐层采样时选到互不连通的节点（见下方参考文献 [2]）。本实现改为：每层从当前层的 1-hop 邻居中采样，并与当前层节点做并集；这与 [2] 的 LADIES 思路相似，但我们不重算邻接矩阵（不同于 [2] 算法 1 第 5 行）。此外，本实现采用**不放回采样**。
2. **预计算**：如 [1] 所述，GCN 的输入层相对梯度是固定的。因此本实现在第一层不做采样，而是使用完整邻居信息（等价于对初始层做预计算）。
3. **概率缩放**：根据 [1] 的公式 (10)，前向传播中每个节点更新需要按采样概率做缩放。由于本实现按 1-hop 邻居进行局部采样（见第 1 点），概率分布不再是全局节点分布，而是局部邻居分布。该做法与 FastGCN 原始数学定义有差异，但在测试准确率上有明显提升。
4. **重要性采样**：采用邻接矩阵每列的 2-范数平方作为采样权重。
5. **数据集划分**：与 [1] 保持一致：除验证集和测试集外，其余全部作为训练集。
6. **批大小与采样大小**：默认 2 层 GCN。除 Reddit 外，损失在 256 大小 mini-batch 上计算，下一层采样 400 节点（Reddit 为 batch=1024，sample=5120）。由于输入层做预计算，后续不再采样（尽管是 2 层 GCN）。
7. **其他超参数**：Adam，学习率 0.01，训练 200 轮；若验证损失 10 轮不下降则早停（Reddit 为 20 轮）；使用权重衰减（L2），仅 PubMed 做特征行归一化，不使用 dropout。Cora/PubMed/CiteSeer 隐层维度为 16，Reddit 为 128。
8. **推理方式**：推理阶段默认使用全图节点进行前向传播（无批采样），用于验证损失与测试准确率计算。

##### OGB 更新（2023）

已新增 [OGB 数据集](https://ogb.stanford.edu) 的实验，包含 arxiv 与 products。为提升效率，原 [fastgcn_model.py](models/fastgcn_model.py) 基础上做了优化，并放在 [updated_fastgcn_model.py](models/updated_fastgcn_model.py)。

模型结构参考 [OGB 的 GCN 示例实现](https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred)。两组数据都使用 3 层 GCN、隐层维度 256；arxiv 在中间层后加入 BatchNorm。两者训练 batch 都为 1024；arxiv 的 sample size 为 10240，products 为 15360。新增参数 `--scale_factor`，用于更深网络时在深层采样更多邻居（arxiv 取 5，products 取 1）。

两组 OGB 实验均训练 1000 轮（不早停）。通过 `--report` 控制间隔汇报 mini-batch 推理结果：arxiv 每 10 轮，products 每 250 轮。由于推理时间较长，每次仅跑 1 次试验。arxiv 的 mini-batch 推理波动较大，建议 `--report=1`（但会更慢）。

## 结果

在 Cora 上执行如下命令，可得到类似输出：

```bash
python3 fastgcn_test.py --dataset='Cora' --fast="true" --hidden_dim=16 --norm_feat="false" --lr=0.01 --init_batch=256 --sample_size=400

# OUTPUT
========================= STARTING TRAINING =========================
TRAINING INFORMATION:
[DATA] Cora dataset
[FAST] using FastGCN? True
[INF] using sampling for inference? False
[FEAT] normalized features? False
[DEV] device: cuda:0
[ITERS] performing 200 Adam updates
[LR] Adam learning rate: 0.01
[BATCH] batch size: 256
[SAMP] layer sample size: 400

[STOP] early stopping at iteration: 57

RESULTS:
[LOSS] minimum loss: 0.14825539290905
[ACC] maximum micro F1 testing accuracy: 87.8 %
[TIME] 0.7757 seconds
========================== ENDING TRAINING ==========================
```

示例收敛曲线：

![Loss 曲线](results/Cora_train_loss.png)
![Accuracy 曲线](results/Cora_testing_accuracy.png)

我们将当前方法与原始 GCN 做比较，统计：
- 最大测试准确率
- 每次迭代训练耗时
- 总训练耗时（不含推理时间）

除 Reddit 外，其余数据集均对 3 次随机初始化取平均；Reddit 仅做 1 次（时间开销较大）。最后一列为采样推理设置（见 [3]）：训练 batch 不变，第二层推理采样数分别设为 1280、2560、2560、15360。

```text
|          | FastGCN (full-batch inference) | GCN (full-batch inference) | FastGCN (mini-batch inference) |
| Dataset  | acc     | per-batch  | total   | acc   | per-batch | total  | acc     | per-batch  | total   |
| -------- | ------- | ---------- | ------- | ----- | --------- | ------ | ------- | ---------- | ------- |
| Cora     | 87.5%   | 0.0071s    | 0.77s   | 87.0% | 0.0048s   | 0.58s  | 87.9%   | 0.0071s    | 0.77s   |
| CiteSeer | 78.4%   | 0.0071s    | 0.58s   | 78.7% | 0.0049s   | 0.47s  | 79.3%   | 0.0071s    | 0.56s   |
| PubMed   | 88.3%   | 0.0075s    | 1.27s   | 88.5% | 0.0053s   | 1.4s   | 88.3%   | 0.0075s    | 1.3s    |
| Reddit   | 94.5%   | 0.1663s    | 33.3s   | 94.8% | 1.61s     | 323.6s | 92.7%   | 0.169s     | 21.03s  |
| arxiv    | 71.6%   | 0.2031s    | 203.1s  | 71.6% | 0.258s    | 258.1s | 70.0%   | 0.205s     | 204.7s  |
| products | OOM     | N/A        | N/A     | OOM   | N/A       | N/A    | 75.9%   | 0.4804s    | 480.4s  |
```

#### 命令行复现实验

下表命令可复现上面的实验表。`Command (full-batch inference)` 对应第一列结果；把 `--fast='true'` 改为 `--fast='false'` 可复现第二列；在第一列命令基础上追加 `mini-batch inference` 列参数（且保持 `--fast='true'`）可复现第三列。

| 数据集 | full-batch inference 命令 | mini-batch inference 追加参数 |
|---|---|---|
| Cora | `python3 fastgcn_test.py --dataset='Cora' --norm_feat='false' --fast='true' --hidden_dim=16 --init_batch=256 --sample_size=400 --early_stop=10 --wd=5e-4` | `--samp_inference='true' --inference_init_batch=256 --inference_sample_size=1280` |
| CiteSeer | `python3 fastgcn_test.py --dataset='CiteSeer' --norm_feat='false' --fast='true' --hidden_dim=16 --init_batch=256 --sample_size=400 --early_stop=10 --wd=5e-4` | `--samp_inference='true' --inference_init_batch=256 --inference_sample_size=2560` |
| PubMed | `python3 fastgcn_test.py --dataset='PubMed' --norm_feat='true' --fast='true' --hidden_dim=16 --init_batch=256 --sample_size=400 --early_stop=10 --wd=5e-4` | `--samp_inference='true' --inference_init_batch=256 --inference_sample_size=2560` |
| Reddit | `python3 fastgcn_test.py --dataset='Reddit' --norm_feat='false' --fast='true' --hidden_dim=128 --init_batch=1024 --sample_size=5120 --early_stop=20 --wd=1e-4` | `--samp_inference='true' --inference_init_batch=1024 --inference_sample_size=15360` |
| arxiv | `python3 fastgcn_test.py --dataset='ogbn-arxiv' --norm_feat='false' --fast='true' --init_batch=1024 --sample_size=10240 --early_stop=-1 --wd=0.0 --batch_norm='true' --hidden_dim=256 --num_layers=2 --drop=0.5 --scale_factor=5 --epochs=1000 --lr=0.001` | `--samp_inference='true' --inference_init_batch=8192 --inference_sample_size=169343 --report=10` |
| products* | `python3 fastgcn_test.py --dataset='ogbn-products' --norm_feat='false' --fast='true' --init_batch=1024 --sample_size=15360 --early_stop=-1 --wd=0.0 --batch_norm='false' --hidden_dim=256 --num_layers=2 --drop=0.5 --scale_factor=1 --epochs=1000 --lr=0.01` | `--samp_inference='true' --inference_init_batch=32768 --inference_sample_size=491520 --report=250` |

*products 数据集在 full-batch inference 下会显存溢出（OOM），建议使用 mini-batch inference 评估准确率。该数据集推理成本较高，主要因为测试集占比较大 [4]。更大显存的 GPU 可能支持 full-batch inference [4]。

## 参考文献

[1] Jie Chen, Tengfei Ma, and Cao Xiao. [FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://arxiv.org/abs/1801.10247). ICLR 2018.

[2] Difan Zou, Ziniu Hu, Yewen Wang, Song Jiang, Yizhou Sun, and Quanquan Gu. [Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks](https://proceedings.neurips.cc/paper/2019/file/91ba4a4478a66bee9812b0804b6f9d1b-Paper.pdf). NeurIPS 2021.

[3] Tim Kaler, Nickolas Stathas, Anne Ouyang, Alexandros-Stavros Iliopoulos, Tao B. Schardl, Charles E. Leiserson, and Jie Chen. [Accelerating Training and Inference of Graph Neural Networks with Fast Sampling and Pipelining](https://arxiv.org/pdf/2110.08450.pdf). MLSys Conference 2022.

[4] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://papers.neurips.cc/paper/2020/file/fb60d411a5c5b72b2e7d3527cfc84fd0-Paper.pdf). NeurIPS 2020.
