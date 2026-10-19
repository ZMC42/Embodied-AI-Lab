# 监督微调（SFT）

先完成 [环境准备](setup.md) 与 [数据准备](data.md)，准备 `models/GR00T-N1.6-3B` 和 `data/locomanipulation/generated/lerobot`。训练调用 Arena 内嵌的 GR00T，使用 `.venvs/groot`。

## 配方与 batch

入口为 `scripts/train/sft.sh <recipe> [--per-device-batch N] [-- 上游参数...]`，默认配方 `smoke`。配方位于 `configs/sft/`。

| 配方 | 有效 batch | 默认每卡 batch | 步数 / 保存间隔 | 训练范围 |
| --- | --- | --- | --- | --- |
| smoke | 1 | 1 | 200 / 200 | 冻结视觉与 LLM，训练 projector、DiT 和层归一化；BF16，top LLM layers=0，loader workers=0 |
| g1-full | 96 | 24 | 20000 / 5000 | 训练视觉、projector、diffusion model 和 LLM 最后 4 层；精度与优化设置沿用上游默认值 |

启动器检测可见 GPU 数 `n`，计算 `global_batch=n×per_device_batch` 和 `gradient_accumulation_steps=effective_batch/global_batch`。无法整除直接报错。smoke 的有效 batch 为 1，因此多卡机器必须通过 `CUDA_VISIBLE_DEVICES` 只暴露一张卡。g1-full 在 1/2/4 卡且每卡 batch=24 时累积步数分别为 4/2/1；8 卡需改用每卡 batch=12。调整每卡 batch 后仍保持有效 batch=96。

上游 `--no-tune-llm` 仍会按 `tune_top_llm_layers` 解冻末尾层，默认值为 4；g1-full 沿用该行为，smoke 则显式设为 0。

配方中的相对路径以项目根目录展开；`--` 后的参数原样追加到上游 CLI，覆盖值按上游规则处理。训练工作目录是 GR00T 源码目录，因此尾参中的输出、数据等路径请使用绝对路径。

## 冒烟与正式训练

以下命令运行 20 步并保存一次，不修改默认 smoke 配方。每次使用新的运行目录：

```bash
RUN_DIR="$PWD/models/sft-smoke-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 scripts/train/sft.sh smoke -- \
  --output-dir "$RUN_DIR" --max-steps 20 --save-steps 20
```

完成时退出 0，生成 `checkpoint-20`（模型、优化器、scheduler、RNG、processor 等）及输出目录下的 `DONE`。`DONE` 只在正常退出后写入，启动时清除旧标记。日志同时追加到 `logs/train-sft/smoke.log`；不同运行共用配方日志，checkpoint 通过不同 output-dir 区分。已有输出目录中的 checkpoint 可能触发上游自动恢复，不保证逐位复现。

正式配方示例（4 张可见 GPU，每卡 batch=24）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train/sft.sh g1-full --per-device-batch 24 -- \
  --output-dir "$PWD/models/sft-full-$(date +%Y%m%d-%H%M%S)"
```

需要四卡及相应显存；24 GiB 单卡 smoke 的可运行性不能证明正式配方的资源需求。

## 检查与调试

用刚生成的 checkpoint 执行 GPU 回归：

```bash
LAB_GPU_SMOKE="$RUN_DIR/checkpoint-20" LAB_GPU_SMOKE_STEPS=20 \
  scripts/test.sh groot --gpu
```

检查训练步数、保存状态、冻结与更新权重，以及 LeRobot episode 0/49/99 第 100 帧的推理。数据必须包含这些帧，检查不测闭环成功率。模型使用 31 维状态、35 维动作、50 步动作块；processor 补齐到 128 维，loss mask 只覆盖有效动作维度，原始 Parquet 的 43 维关节向量不能直接视为模型动作。

入口参数可用 `scripts/train/sft.sh --help` 查看；上游 CLI 的全部选项由 GR00T 定义。`TRAIN_DEBUG=1` 在训练进程启动 debugpy 并等待连接，使用方法见 [训练断点调试](debugging.md#训练断点调试)。
