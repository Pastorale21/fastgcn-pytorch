# Linux 运行说明

本目录脚本按 Linux 命令行环境设计（`bash` + `python3`）。

## 1. 准备环境

```bash
cd /path/to/fastgcn-pytorch

# 可选：创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖（按你的 CUDA/PyTorch 版本调整）
pip install -U pip
pip install -r requirements.txt
```

## 2. 运行基准实验

```bash
chmod +x scripts/run_readme_benchmarks.sh
./scripts/run_readme_benchmarks.sh
```

默认行为：
- 数据集：`Cora CiteSeer PubMed Reddit ogbn-arxiv ogbn-products`
- 模式：`fast_full gcn_full fast_mini`
- 重复次数：除 `Reddit` 外每组 3 次，`Reddit` 1 次
- 自动画图

## 3. 常用子集运行

```bash
# 先跑轻量数据集
DATASETS="Cora CiteSeer PubMed" ./scripts/run_readme_benchmarks.sh

# 只跑 FastGCN 与 GCN 的 full inference
MODES="fast_full gcn_full" ./scripts/run_readme_benchmarks.sh

# 只采集 CSV，不画图
RUN_PLOTS=0 ./scripts/run_readme_benchmarks.sh
```

## 4. 输出结果

脚本会在 `benchmarks/<timestamp>/` 下生成：
- `results.csv`：每次运行明细
- `summary.csv`：按 dataset+mode 聚合统计
- `logs/*.log`：每次训练日志
- `comparison_f1.png`
- `comparison_total_time.png`
- `comparison_batch_time.png`
- `tradeoff_f1_vs_total_time.png`

## 5. 运行参数敏感性实验

```bash
chmod +x scripts/run_param_sweeps.sh
./scripts/run_param_sweeps.sh
```

默认行为：
- 数据集：`PubMed Reddit`
- 实验：`sample_size init_batch samp_dist`
- 重复次数：`PubMed` 3 次，`Reddit` 1 次
- 自动画图

常用子集运行：

```bash
# 只做 sample_size 分析
EXPERIMENTS="sample_size" ./scripts/run_param_sweeps.sh

# 只跑 PubMed
DATASETS="PubMed" ./scripts/run_param_sweeps.sh

# 只打印命令，不真正执行
DRY_RUN=1 RUN_PLOTS=0 ./scripts/run_param_sweeps.sh
```

参数实验会在 `param_sweeps/<timestamp>/` 下生成：
- `results.csv`：每次 sweep 运行明细
- `summary.csv`：按 dataset+experiment+param_value 聚合统计
- `logs/*.log`：每次训练日志
- `<dataset>_<experiment>.png`：每组参数 sweep 的三联图（acc / batch time / total time）
