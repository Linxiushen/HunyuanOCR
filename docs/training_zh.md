# 训练指南

[English Version](./training.md)

本文档说明本仓库支持的三种训练模式：

1. **对 HunyuanOCR 基座模型做 SFT**（端到端全量微调）
2. **从零训练 DFlash 草稿**（大规模数据，例如约 1M 打包样本）
3. **DFlash 草稿继续微调**（小规模领域数据，例如约 10k 打包样本）

---

## 1. 对基座 HunyuanOCR 做 SFT

对 HunyuanOCR 做端到端的全量监督微调：视觉编码器 + MLP + LLM。

### 入口

`train/train_hunyuan.py`，由 `scripts/sft_base.sh` 调用。

### 快速运行

```bash
# 单机 8 卡
MODEL_PATH=/path/to/HunyuanOCR \
TRAIN_DATA=./data/parsing_packed_20480.jsonl \
    bash scripts/sft_base.sh
```

### 关键超参数

| 环境变量         | 默认值 | 含义                 |
| ---------------- | -----: | -------------------- |
| `LR`             | `2e-5` | 学习率               |
| `EPOCHS`         |    `5` | 训练轮数             |
| `BATCH_SIZE`     |    `1` | 每卡训练 batch 大小  |
| `GRAD_ACCUM`     |    `1` | 梯度累积步数         |
| `SAVE_STEPS`     |  `200` | 每 N 步保存一次 ckpt |
| `NPROC_PER_NODE` |    `8` | 每节点使用的 GPU 数  |

### 多机

设置标准的 PyTorch 分布式环境变量：

```bash
# Node 0（主节点）
NNODES=4 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
    bash scripts/sft_base.sh

# Node 1
NNODES=4 NODE_RANK=1 NPROC_PER_NODE=8 \
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
    bash scripts/sft_base.sh

# 节点 2、3 依此类推
```

按你集群的网卡布局调整 `scripts/env_common.sh` 中的 `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA`。

---

## 2. 从零训练 DFlash 草稿

从随机初始化开始训练一个小的 MTP 风格草稿模型。**当数据量充足时推荐使用**（约 1M 打包样本）。

### 入口

`train/train_draft.py`，由 `scripts/sft_dflash.sh` 调用。

### 快速运行

```bash
MODEL_PATH=/path/to/HunyuanOCR \
TRAIN_DATA=./data/parsing_packed_20480.jsonl \
    bash scripts/sft_dflash.sh
```

### 关键超参数

| 环境变量           | 默认值 | 含义                        |
| ------------------ | -----: | --------------------------- |
| `LR`               | `1e-4` | 从零训练适用的较大学习率    |
| `EPOCHS`           |    `2` | 大数据量下的少量轮次        |
| `NUM_MASK_TOKENS`  |   `16` | 每个位置的 K 个投机 token   |
| `SAMPLE_BLOCK_NUM` |    `8` | 每个 pack 采样的 block 数量 |
| `LOOP_NUM`         |    `1` | MTP 风格的迭代次数          |
| `SAVE_STEPS`       | `2000` | 每 N 步保存一次 ckpt        |

### 草稿模型 config 模板

草稿模型的架构（层数、hidden size、block size 等）从一个小型模板目录加载，该目录只需包含 `config.json` + `dflash.py`。默认路径是仓库自带的 [`train/configs/`](../train/configs)。可通过环境变量 `HYOCR_DFLASH_CONFIG_DIR` 覆盖为任意包含这两个文件的目录，例如 HuggingFace HunyuanOCR 发布模型里的 `dflash/` 子目录：

```bash
HYOCR_DFLASH_CONFIG_DIR=/path/to/HunyuanOCR/dflash \
    bash scripts/sft_dflash.sh
```

### 输出

训练会生成：

- `output/{run_name}/model.safetensors`：DFlash 草稿权重（约 350 MB）
- `output/{run_name}/config.json`：DFlash 草稿配置
- `output/{run_name}/checkpoint-XXXX/`：中间检查点

推理只需要 `model.safetensors` + `config.json` + `dflash.py`，详见 [`docs/inference/inference_zh.md`](inference/inference_zh.md)。

---

## 3. 从已有 DFlash 检查点继续微调

从一个已训练好的 DFlash 检查点（例如我们发布的 v1）出发，适配到更小 / 领域特定的数据集。

**当数据量较小（约 10k 打包样本）且已有较优 DFlash 草稿时推荐**。

### 入口

`train/train_draft_from_dflash.py`，由 `scripts/sft_dflash_finetune.sh` 调用。

### 快速运行

```bash
MODEL_PATH=/path/to/HunyuanOCR \
DFLASH_INIT=/path/to/existing/dflash/checkpoint \
TRAIN_DATA=./data/domain_packed_20480.jsonl \
    bash scripts/sft_dflash_finetune.sh
```

### 关键超参数

| 环境变量      |   默认值 | 含义                      |
| ------------- | -------: | ------------------------- |
| `LR`          |   `2e-5` | 比从零训练小 **5 倍**     |
| `EPOCHS`      |     `10` | 小数据下**多训几轮**      |
| `WARMUP`      |   `0.05` | 稍长的 warmup             |
| `SAVE_STEPS`  |    `500` | 整个训练约保存 10 个 ckpt |
| `DFLASH_INIT` | _(必填)_ | 已有草稿检查点目录路径    |

当 `DFLASH_INIT` 指向一个有效目录时，草稿 config 从这个目录加载；否则会回退到 `HYOCR_DFLASH_CONFIG_DIR`（默认为 `train/configs/`）。

### 实测结果（在 930 张 OCR 评测集上对比 v3 vs v1）

|                          | v1（1M pack，从零训练） |      v3（14.7k pack，微调） |
| ------------------------ | ----------------------: | --------------------------: |
| 数据量                   |             1.02 M pack | **14.7 k pack**（小 70 倍） |
| 相对基座 AR 的端到端加速 |                   1.92× |         **2.14×**（+11.6%） |
| 平均接受长度（/15）      |                    6.06 |          **7.36**（+21.4%） |
| 平均草稿接受率           |                   33.8% |        **42.4%**（+8.7 pp） |

**结论**：做领域自适应时，用较小学习率 + 较多轮次从预训练草稿微调，效果始终优于用同等或更少数据从零训练。

---

## 训练数据格式

三种模式使用同一份由 `tools/pipeline_count_and_pack.py` 生成的打包 JSONL 格式，详见 [`data_format_zh.md`](data_format_zh.md)。

## 分布式 / DeepSpeed

若要使用 DeepSpeed ZeRO-2（当模型 > 3B 参数且 GPU ≤ 24GB 时推荐），在训练脚本的 `args` 块中加：

```bash
--deepspeed scripts/zero2.json \
```

默认未启用。H20 80GB + `gradient_checkpointing=True` 大多数情况下无需 ZeRO 即可完成全量训练。

## 调试

- 打开 NCCL 调试日志：`export NCCL_DEBUG=INFO`（已在 `env_common.sh` 中）
- 打开 CUDA 同步以便看到堆栈：`export CUDA_LAUNCH_BLOCKING=1`
- 打开完整的 torch 分布式调试：`export TORCH_DISTRIBUTED_DEBUG=DETAIL`

常见问题：

| 现象                | 修复                                                     |
| ------------------- | -------------------------------------------------------- |
| NCCL 若干分钟后超时 | 检查 IB 网络；增大 `NCCL_TIMEOUT`                        |
| 80GB GPU 上 OOM     | 将 `packed_max_length` 从 20480 降到 16384               |
| Loss 不下降         | DFlash 模式下检查草稿初始化目录里的 config.json 是否正确 |
