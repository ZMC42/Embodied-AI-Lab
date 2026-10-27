# Arena + N1.6 PPO

先完成 [环境准备](setup.md)，准备 `models/GR00T-N1.6-G1/checkpoint-20000`，模型下载见 [评测说明](eval.md)。使用 `.venvs/rl`，配置位于 `configs/rl/locomanip/`，入口为 `scripts/train/rl.sh <recipe> [Hydra overrides...]`；省略 recipe 时选择 `ppo-offload`。

需要 VSCode 断点时，按 [RL / PPO 调试流程](debugging.md#rl--ppo) 用 `RL_DEBUG=driver|actor|rollout|env` 选择本次角色，并连接对应 attach 配置；`SFT_DEBUG=1` 仅用于 SFT。

## 配置与实现

`embodied_ai_lab.environments` 基于 Arena 官方 G1 搬运任务构建 RL 环境并复用 G1 IO，RLinf 通过 `Arena-v0` 桥接该环境。成功奖励 +1，掉箱和超时为 0；动作块内部延后 reset，保留终局奖励和观测，供 RLinf 对超时进行 value bootstrap。环境成功率使用 Arena 官方 success 信号。`fall_once` 是骨盆相对默认站立位置下降超过 0.4 m 的诊断指标，终止后屏蔽、reset 时清零。

RL 环境由 `LocomanipRLEnvironment` 的 typed cfg 和 `ArenaEnvBuilder` 构建；`env.train.arena` / `env.eval.arena` 映射官方 `GalileoG1LocomanipPickAndPlaceEnvironmentCfg` 字段，可通过 Hydra 修改 `object`、`destination` 等场景参数。配方显式指定 `embodiment=g1_wbc_joint`、`enable_cameras=true`，不再接受任意 `arena_args` 或 legacy `ExampleEnvironmentBase` 工厂。AppLauncher 先启动，再导入 Arena；builder 负责 seed、设备、环境数及 variation recorder 的传递。

`ArenaEnv` 目前仅支持 `auto_reset=true`、`ignore_terminations=false`；`max_episode_steps=1500` 对应 Arena 的 30 s × 50 Hz。这三个字段供 RLinf 的 reset、bootstrap 和 loss 逻辑使用，不能单独改成其他组合；改变 episode 上限须同步修改 Arena 任务配置。train 和 eval 均提供 `viser_envs: 0`，可使用 RLinf 的训练中验证及 `only_eval`。

| 配方 | 模型与更新设置 |
| --- | --- |
| ppo | 冻结 VLM，训练整个 action head 和 value head；无 actor/rollout offload，patch 权重同步 |
| ppo-offload | actor/rollout offload；训练 decoder、最后 1 个 DiT block 和 value head；512 MiB bucket 同步；4 个 update epoch |

冻结参数始终使用 BF16 存储。offload 配方的可训练主权重为 FP32，FSDP 前向 BF16，梯度归约 FP32，Adam 状态 FP32，rollout BF16。`actor.model.train_last_blocks=N` 调整末尾 DiT block 数。`actor.micro_batch_size=1` 和 `actor.model.inference_micro_batch_size=1` 需一并保留，避免 BF16 下重组 batch 改变 chunk log-prob。

默认 seed=1000、2 个环境、50 步动作块、1500 步 episode。offload 每个 update epoch 累积 60 个动作块后更新。入口默认 `MALLOC_ARENA_MAX=2`，减少主机内存保留，为 CPU checkpoint 保存留出空间；可用同名环境变量覆盖。

只解析配置，不启动训练：

```bash
scripts/train/rl.sh ppo-offload --cfg job --resolve
scripts/train/rl.sh ppo --cfg job --resolve
```

GR00T RL fork 的 letterbox 预处理与 reference 不同，因此实际 RL 使用 GR00T reference；fork 保留作推理对照。reference 与 RLinf 的 attention 使用 SDPA；Euler 去噪更新顺序与 BF16 舍入通过 fork 保持一致。无需再手动安装补丁。

## 并行环境数

默认 `ppo` 与 `ppo-offload` 仍使用 2 个环境。在训练命令中，用 `scripts/train/rl.sh env.train.total_num_envs=N` 指定环境数，或在 recipe 后追加 `env.train.total_num_envs=N`。例如使用 4 个环境：

```bash
scripts/train/rl.sh env.train.total_num_envs=4 --cfg job --resolve
```

该命令仅查看配置；实际训练见下方短跑示例。`actor.global_batch_size` 自动解析为 `N × (max_steps_per_rollout_epoch / num_action_chunks)`，默认 rollout 长度下为 `30N`。2、4、8 个环境分别对应 60、120、240，每轮的每个 update epoch 仍只更新一次，学习率不变。

`total_num_envs` 是全局环境总数，在所有 env worker 间平分；多卡时取 `每卡环境数 × GPU 数`，且须被 GPU 数整除；global batch 同样须被 `micro_batch_size × actor GPU 数` 整除，N 为 GPU 数的倍数时通常自然满足。扩展卡数只需调整 `cluster.num_nodes` 与可见 GPU，其余训练参数不变。

容量实测使用 GPU PhysX、seed=1000、官方 checkpoint-20000，目标为每组两轮并在第 2 轮保存。成功组的第 2 轮迭代耗时不含保存，整卡峰值包含保存；余量统一按可分配的 24082 MiB 减去峰值计算。原始数据位于 `logs/num-envs-viser/`。

| 环境数 | global batch | 第 2 轮耗时 / s | 控制步 / s | 整卡峰值 / MiB | 余量 / MiB | 退出码 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 60 | 113.057 | 26.535 | 22935 | 1147 | 0 |
| 4 | 120 | 156.626 | 38.308 | 23427 | 655 | 0 |
| 8 | 240 | — | — | 24074 | 8 | 245（OOM 后受控终止） |

2env 数据来自 `baseline-n2-retry1/`。相对 R3c 约 24.55 步/s 的参考值快 8.10%，超过 5% 的复查线。复查确认 R3c 取首轮，本表取第 2 轮；同口径首轮耗时为本次 124.081 s、R3c 122.216 s，差约 1.53%。这里的 8.10% 不能作为实现提速结论。

8env 在第 2 轮权重同步的 rollout CUDA IPC 接收克隆阶段 OOM：申请 128 MiB 时仅剩 8.88 MiB。记录异常后受控终止，未完成第 2 轮或保存，因此不报告该轮吞吐；表中为退出前采样峰值。

## 两迭代短跑

以下短跑在第 2 次迭代保存一次，保留默认模型、batch 与 rollout 长度：

```bash
RUN_DIR="$PWD/logs/rl-training/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
set -o pipefail
/usr/bin/python3 -m embodied_ai_lab.monitor --output "$RUN_DIR" \
  "$PWD/scripts/train/rl.sh" ppo-offload \
  "runner.logger.log_path=$RUN_DIR" runner.logger.experiment_name=train \
  runner.max_epochs=2 runner.max_steps=2 runner.save_interval=2 \
  2>&1 | tee "$RUN_DIR/run.log"
```

同一时间顺序运行 GPU 任务，避免训练与仿真互抢资源。默认 offload 配方虽只跑两个 epoch，但保存间隔是 125，短跑必须显式覆盖为 2。正常结束时 `process.json` 中 exit_code=0，保存完整权重和恢复状态。

| 产物 | 位置（相对 RUN_DIR） |
| --- | --- |
| 终端日志 | `run.log` |
| 退出状态、耗时、整卡峰值显存 | `process.json`、`gpu-memory.csv` |
| 有效配置与指标 | `tensorboard/config.yaml`、`tensorboard/` |
| 完整模型权重 | `train/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt` |
| DCP 优化器等恢复状态 | `train/checkpoints/global_step_2/actor/dcp_checkpoint/` |

监控统计整卡显存总和，不是单进程显存。CPU 物理使用 `env.train.device=cpu`；默认物理设备为 `cuda:0`。正式训练需显式设置 `runner.max_epochs`、`runner.max_steps` 和 `runner.save_interval`。

## 恢复与指标

在上述训练参数末尾、日志管道之前追加 `"runner.resume_dir=$PREVIOUS_RUN/train/checkpoints/global_step_2"` 恢复；`PREVIOUS_RUN` 指向旧的运行目录，新运行使用不同的 `RUN_DIR`。resume_dir 指到 `global_step_N`，保持模型、训练范围、环境数和 batch 一致，将 max_epochs、max_steps 设置为恢复后的目标总步数。恢复模型、优化器和 scheduler，不保证仿真轨迹逐位续接。

offload 配方启用 `algorithm.log_epoch_metrics`，记录 `actor/epoch_1..N/{clip_fraction,approx_kl,ratio,valid_count}` 和剔除 epoch1 的 `actor/epoch_ge_2/*`。按 PPO 实际广播后的有效 loss_mask 元素数加权，跨 rank 求和再相除；全零 mask 的整组均值为 NaN。`approx_kl` 使用有符号的 `-mean(log_ratio)`，`clip_fraction` 沿用上游考虑 advantage 的裁剪判定。原有跨 epoch 聚合指标保持原口径。

训练后使用 [导出与评测](eval.md) 的公开入口导出 safetensors。checkpoint、精度、actor 与导出一致性检查见 [测试说明](testing.md)；它们验证保存及推理链路，不代表 PPO 已改善策略成功率。
