# 闭环评测与 RL 导出

从项目根目录执行，环境准备见 [setup.md](setup.md)。评测需要 Arena 资产、CUDA 相机以及可直接推理的模型目录；首次运行会加载和编译场景资产。同一时间只运行一批评测，启动前确认本地端口 5555 空闲。`models/`、`logs/` 等大型产物通过项目软链接写入大容量存储；Python 环境保留在本地 `.venvs/`。

Arena 任务资产使用 NVIDIA production 源；`scripts/env.sh` 将资产版本固定为 6.1，与当前 Isaac Sim 环境一致。

## 官方 G1 checkpoint

可使用 `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation`，revision `dfe74af855007f26093f362cd2d7a2f404b64b93`。只下载推理文件，约 6.2 GB：

```bash
HF_HOME="$PWD/cache/huggingface" uvx --from huggingface_hub hf download \
  nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
  --revision dfe74af855007f26093f362cd2d7a2f404b64b93 \
  --include '*.json' --include '*.yaml' --include '*.safetensors' \
  --local-dir models/GR00T-N1.6-G1/checkpoint-20000
```

## 多 checkpoint、多 seed 评测

`scripts/eval/run.sh` 统一启动推理服务器和 Arena 客户端，按 checkpoint × seed 串行评测，并在每次尝试结束后清理子进程。单 checkpoint、单 seed 也使用这个入口：

```bash
scripts/eval/run.sh \
  --checkpoints models/GR00T-N1.6-G1/checkpoint-20000 \
  --seeds 42 \
  --output logs/eval/g1-seed42
```

默认每个组合运行 100 个完整 episode，使用 5 个并行环境和 GPU 物理。`--physics cpu` 可选择 CPU 物理，模型推理和相机仍需 GPU。`--checkpoints`、`--seeds`、`--output` 必填，首次运行的输出目录必须不存在。

批量评测时，显式列出模型目录和 seed，例如将下列 checkpoint 路径替换为实际训练产物：

```bash
scripts/eval/run.sh \
  --checkpoints models/run-a/checkpoint-1000 models/run-a/checkpoint-2000 \
  --seeds 42 43 44 \
  --episodes 50 --num-envs 5 --physics gpu \
  --output logs/eval/experiment-1 \
  --retries 1
```

这里执行 6 个组合，每个组合 50 局，共 300 局。每个 checkpoint 使用相同的 seed 列表；每次尝试重新启动服务器和仿真，同时设置推理 RNG 和仿真 seed。协议固定为 50 Hz、每局最多 1500 控制步、动作块 50 步，任务为 `galileo_g1_locomanip_pick_and_place`、物体 `brown_box`、机器人 `g1_wbc_joint`。固定 seed 不保证 PhysX 逐位复现。

### 重试与续跑

`--retries N` 表示首次失败后最多额外重试 N 次，默认 `0`。重试耗尽后继续其他组合；最终存在未完成组合时，整批返回非零退出码。在启动命令的终端按 Ctrl+C 会停止当前评测并清理子进程。

使用原命令追加 `--resume`，在同一输出目录续跑：

```bash
scripts/eval/run.sh \
  --checkpoints models/run-a/checkpoint-1000 models/run-a/checkpoint-2000 \
  --seeds 42 43 44 \
  --episodes 50 --num-envs 5 --physics gpu \
  --output logs/eval/experiment-1 \
  --retries 2 --resume
```

续跑要求 checkpoint 列表及顺序、seed 列表及顺序、局数、环境数、物理后端和推理栈与 `run.json` 一致；`--retries` 可调整，预算按本次调用重新计算。已完成组合直接跳过，失败或中断的组合从头运行，并追加新的 attempt 目录。恢复粒度是 checkpoint × seed，不拼接不同尝试中的 episode。

### 产物与汇总

`ckpt-000`、`ckpt-001` 按命令中的 checkpoint 顺序编号，实际路径记录在 `run.json`，因此不同模型的同名 checkpoint 不会覆盖。

| 路径 | 内容 |
| --- | --- |
| `run.json` | checkpoint、seed 列表及整批评测参数 |
| `summary.json` | 各组合状态、成功数、局数、成功率、耗时及日志位置；各 checkpoint 的跨 seed 汇总 |
| `ckpt-000/seed-42/attempt-0001/episodes.jsonl` | Arena 原生逐局记录，成功率沿用原生 success 信号 |
| 同目录 `protocol.json` | 实际模型路径、seed、环境数、物理设备、相机、reset 渲染次数、包版本及推理栈 |
| 同目录 `result.json` | 客户端生成的局数、成功数和 rollout 耗时 |
| 同目录 `complete.json` | 编排进程在客户端退出码为 0 且局数、成功数核对通过后写入的完成标记 |
| 同目录 `server.log`、`eval.log`、`kit.log` | 推理服务器、客户端和仿真日志 |
| 同目录 `error.log` | 本次尝试失败的错误信息 |

只有带有效 `complete.json` 的组合计入正式结果；模型加载日志或部分逐局记录不足以证明完成。`summary.json` 和终端表格列出完成、失败与未运行的组合。只有一个 checkpoint 的全部 seed 完成后，才计算其各 seed 成功率的均值和样本标准差；单 seed 的标准差为 `null`。耗时统计不包含模型加载、等待服务器和失败尝试。

已有结果可单独重新生成汇总，无需再次运行仿真：

```bash
python3 -m embodied_ai_lab.eval.summary logs/eval/experiment-1
```

## 导出 RL 权重

先完成 [RL 训练](rl.md)，取得 `global_step_N/actor/model_state_dict/full_weights.pt`，再独立导出为可推理模型目录。`scripts/eval/export-rl.sh --help` 可查看参数；`--weights` 和 `--output` 必填，`--base-model` 默认为上述官方 G1 checkpoint。基座必须是该次 RL 训练的实际起点，输出必须是不存在的新目录：

```bash
# RUN_DIR 是 rl.md 中两迭代短跑的运行目录；BASE_MODEL 设为实际训练基座。
BASE_MODEL="$PWD/models/GR00T-N1.6-G1/checkpoint-20000"
WEIGHTS_FILE="$RUN_DIR/train/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
MODEL_DIR="$PWD/models/GR00T-N1.6-PPO-$(date +%Y%m%d-%H%M%S)"
scripts/eval/export-rl.sh \
  --weights "$WEIGHTS_FILE" --base-model "$BASE_MODEL" --output "$MODEL_DIR"
```

新终端中先将 `RUN_DIR` 设为该次运行目录。导出移除 `action_head.value_head.*`，严格核对其余参数名与基座索引，按官方分片索引写入 BF16 safetensors，并复制 config、processor、statistics、embodiment 映射及 experiment_cfg。`rl_export.json` 记录来源权重、基座和推理栈。

确认导出命令退出码为 0 且生成 `rl_export.json` 后，用同一评测入口运行：

```bash
scripts/eval/run.sh \
  --checkpoints "$MODEL_DIR" --seeds 42 43 44 \
  --episodes 50 --num-envs 5 --physics gpu \
  --output "logs/eval/rl-$(date +%Y%m%d-%H%M%S)"
```

带 `rl_export.json` 的目录自动使用 `.venvs/rl`、`Isaac-GR00T-reference` 的 SDPA 和逐样本 BF16 前向，与本地 RL rollout 一致；两侧使用 480×640 图像和归一化前 FP32 状态。普通 SFT 目录默认使用官方推理栈，需要使用 reference 推理栈时可显式设置：

```bash
EVAL_RL_SERVER=1 scripts/eval/run.sh \
  --checkpoints models/GR00T-N1.6-G1/checkpoint-20000 --seeds 42 \
  --output logs/eval/g1-reference-seed42
```

续跑时应保持这一环境变量选择一致。BF16 batch 形状、FlashAttention/SDPA 和 Euler 去噪顺序会影响动作，相关约定见 [rl.md](rl.md)。导出前后动作、初始噪声及真实 server/client 的一致性检查已纳入 GPU 测试，前置产物和运行方式见 [testing.md](testing.md)。
