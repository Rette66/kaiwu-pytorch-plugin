# Language Versions: [中文](README_ZH.md) | [English](README.md)

# `example/qdiffusion`

本目录提供 `Q-Diffusion` 的 DPLM 蛋白序列示例。`src/kaiwu/torch_plugin/qdiffusion.py` 只保留通用训练框架和能量模型接口；DPLM backbone、BM/RBM 能量模型、生成策略、训练和 ESM2 评估都放在本 example 目录里。

## 数据

[example/qdiffusion 使用的数据](https://www.uniprot.org/proteomes/UP000005640)

下载 FASTA 后，默认放在：

```text
example/qdiffusion/data/UP000005640_9606.fasta
```

也可以使用自己的 FASTA，并在入口脚本配置里修改 `fasta_path` 或 `reference_fasta`。

## 快速开始

```bash
pip install -r example/qdiffusion/requirements.txt
python example/qdiffusion/dplm/train_workflow.py
python example/qdiffusion/dplm/eval_esm2_distances.py
```

## 目录结构

- `dplm/model/`：DPLM backbone、BM/RBM energy reranker、生成策略和 `build_qdiffusion(...)`
- `dplm/trainer/`：训练数据、epoch loop、checkpoint 和训练报告
- `dplm/downstream/`：ESM2 distance 评估与生成 helper
- `dplm/utils/`：FASTA I/O、指标、运行时工具
- `dplm/train_workflow.py`：训练入口
- `dplm/eval_esm2_distances.py`：ESM2 distance 评估入口

## 主流程

1. 读取并过滤 FASTA 记录
2. 划分 train / validation / test
3. 通过 `build_qdiffusion(...)` 构建 DPLM proposal 和 BM/RBM energy reranker
4. 将序列 tokenize 成 `targets`
5. 训练时调用 `generator.objective({"targets": ...})`
6. 优化 `energy_objective.mean()`，主要训练 BM/RBM energy backend
7. 保存能量模型 checkpoint
8. 评估时重建 baseline 和 guided generator
9. 生成序列并用 ESM2 embedding distance 评估

## 说明

- 可复用库代码在 `src/kaiwu/torch_plugin/qdiffusion.py`
- DPLM 加载、生成策略和 BM/RBM 条件打分属于 example 层
- 正式模型构造入口是 `from dplm.model import build_qdiffusion`
- 当前 guided 路径本质是 `DPLM proposal + BM/RBM energy reranker`
