# Arena + N1.6 PPO

先完成 [环境准备](setup.md)，准备 `models/GR00T-N1.6-G1/checkpoint-20000`，模型下载见 [评测说明](eval.md)。使用 `.venvs/rl`，配置位于 `configs/rl/locomanip/`，入口为 `scripts/train/rl.sh <recipe> [Hydra overrides...]`；省略 recipe 时选择 `ppo-offload`。

需要 VSCode 断点时，按 [RL / PPO 调试流程](debugging.md#rl--ppo) 用 `RL_DEBUG=driver|actor|rollout|env` 选择本次角色，并连接对应 attach 配置；`SFT_DEBUG=1` 仅用于 SFT。

## 配置与实现

`embodied_ai_lab.environments` 复用 Arena 官方任务注册及 G1 IO，RLinf 通过 `Arena-v0` 桥连接任务 `galileo_g1_locomanip_pick_and_place_rl`。成功奖励 +1，掉箱和超时为 0；动作块内部延后 reset，保留终局奖励和观测，供 RLinf 对超时进行 value bootstrap。环境成功率使用 Arena 官方 success 信号。`fall_once` 是骨盆相对默认站立位置下降超过 0.4 m 的诊断指标，终止后屏蔽、reset 时清零。

| 配方 | 模型与更新设置 |
| --- | --- |
| ppo | 冻结 VLM，训练整个 action head 和 value head；无 actor/rollout offload，patch 权重同步 |
| ppo-offload | actor/rollout offload；训练 decoder、最后 1 个 DiT block 和 value head；512 MiB bucket 同步；4 个 update epoch |

冻结参数始终使用 BF16 存储。offload 配方的可训练主权重为 FP32，FSDP 前向 BF16，梯度归约 FP32，Adam 状态 FP32，rollout BF16。`actor.model.train_last_blocks=N` 调整末尾 DiT block 数。`actor.micro_batch_size=1` 和 `actor.model.inference_micro_batch_size=1` 需一并保留，避免 BF16 下重组 batch 改变 chunk log-prob。

`ppo-offload` 保存 checkpoint 时先将 FSDP 张量搬到 CPU；分桶同步通过 FSDP 完整参数上下文，只将需要同步的可训练参数和持久 buffer 复制到 CPU，避免为冻结基座额外创建完整快照。分桶同步保留原 dtype；默认 `ppo` 的 patch 同步保持原有 CUDA 导出路径。

默认 seed=1000、2 个环境、50 步动作块、1500 步 episode。offload 每个 update epoch 累积 60 个动作块后更新。入口默认 `MALLOC_ARENA_MAX=2`，减少主机内存保留，为 CPU checkpoint 保存留出空间；可用同名环境变量覆盖。

只解析配置，不启动训练：

```bash
scripts/train/rl.sh ppo-offload --cfg job --resolve
scripts/train/rl.sh ppo --cfg job --resolve
```

GR00T RL fork 的 letterbox 预处理与 reference 不同，因此实际 RL 使用 GR00T reference；fork 保留作推理对照。reference 与 RLinf 的 attention 使用 SDPA；Euler 去噪更新顺序与 BF16 舍入通过 fork 保持一致。无需再手动安装补丁。

## 物理后端与设备

`env.train.physics_backend` 与 `env.eval.physics_backend` 选择物理引擎，`device` 单独选择物理执行设备：

| 后端 | 执行设备 | Arena 内部 preset |
| --- | --- | --- |
| `isaacsim_physx`（默认） | `cpu`、`cuda:0` | `physx` |
| `newton_mjwarp` | CUDA，例如 `cuda:0` | `newton` |

`ppo` 和 `ppo-offload` 均支持这些字段。eval 的后端默认插值跟随 train，允许单独覆盖；train 与 eval 的设备各自默认 `cuda:0`，修改训练设备不会修改 eval 设备。模型推理和相机仍按原配置使用 GPU。

```bash
# Newton 训练和内置 eval。
scripts/train/rl.sh ppo-offload env.train.physics_backend=newton_mjwarp

# Newton 训练，PhysX 内置 eval。
scripts/train/rl.sh ppo-offload \
  env.train.physics_backend=newton_mjwarp \
  env.eval.physics_backend=isaacsim_physx

# PhysX CPU 训练，Newton GPU 内置 eval。
scripts/train/rl.sh ppo-offload env.train.device=cpu \
  env.eval.physics_backend=newton_mjwarp env.eval.device=cuda:0
```

配置保存在 `configs/rl/locomanip/ppo.yaml`，环境入口为 `embodied_ai_lab/rlinf/env.py`。后端在创建仿真子进程前确定，并在项目边界映射为 Arena 已有的 `physx` / `newton` preset；修改配置后通过新环境进程切换。RL 的 `arena_args` 不允许携带 `--presets` 或 `--physics-backend`，后端统一由上述结构化字段决定；独立 Arena 评测使用 `--presets`。

Isaac Lab 官方的 [`physics=NAME` 选择器](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/environments.html)可选择任务声明的完整 `PresetCfg`。本项目使用 RLinf 入口与 Arena builder，当前通过 Arena preset 接入后端；直接向项目命令追加官方 `physics=` 参数尚未接入该解析流程。

入口先等待 train/eval 仿真全部初始化，再依次加载 rollout 和 actor 模型，避免 Newton 场景构建的主存峰值与模型加载重叠；恢复 checkpoint 的加载仍在全部 worker 就绪后执行。

训练在 `${runner.logger.log_path}/${runner.logger.experiment_name}/physics/` 写入记录：`run.json` 包含来源 checkpoint、来源及当前 train/eval 后端；`train-worker_*-pid_*.json` 和 `eval-worker_*-pid_*.json` 由仿真进程记录实际后端、物理设备、配置和 manager、求解器参数、步长、相机与包版本。`newton_solver_sha256` 用于区分同版本 Newton solver 的源码变化。解析后的训练配置沿用 TensorBoard 配置产物。对应下方短跑示例的物理记录路径为 `$RUN_DIR/train/physics/`。

Newton 的真实训练和跨后端恢复尚未通过整套运行验收。RTX 4090 24 GiB 上，PhysX train/eval 同时使用 GPU 物理的短跑曾在第二轮 rollout 出现 GPU page fault / device lost，采样峰值 23979 MiB；现有证据不能确定是容量还是生命周期问题。可独立设置 PhysX 角色的 `device=cpu`，相机与模型仍使用 GPU。下面已有的容量实测来自未启用内置 eval 的 PhysX 运行，不能外推到两个仿真进程同时驻留。

本机两个 Newton 仿真同时驻留时，分别占用约 6.81/7.06 GiB 显存。串行初始化可跨过场景构建的主存峰值，但 actor 的 FSDP flatten 随后需要额外 3.48 GiB，此时仅余 1.77 GiB，明确超出本机 24 GiB 显存容量。该组合尚未完成训练；可单独选择 PhysX CPU eval 减少驻留开销，或在更大显存机器上验证双 Newton 组合。

Newton GPU train + PhysX CPU eval 的后续短跑虽然将显存峰值降至 20219 MiB，仍在首次权重同步时触发主存保护（约 59.74/62.64 GB），未完成任何 PPO 更新或 checkpoint 保存。因此本机这套配方的 Newton 训练仍未跑通，独立仿真与闭环评测通过不能替代训练验收。

## 并行环境数

默认 `ppo-offload` 仍使用 2 个环境。在训练命令中，用 `scripts/train/rl.sh env.train.total_num_envs=N` 指定环境数，或在 recipe 后追加 `env.train.total_num_envs=N`。例如使用 4 个环境：

```bash
scripts/train/rl.sh env.train.total_num_envs=4 --cfg job --resolve
```

该命令仅查看配置；实际训练见下方短跑示例。`ppo-offload` 的 `actor.global_batch_size` 自动解析为 `N × (max_steps_per_rollout_epoch / num_action_chunks)`，默认 rollout 长度下为 `30N`。2、4、8 个环境分别对应 60、120、240，每轮的每个 update epoch 仍只更新一次，学习率不变。`ppo` 配方的 batch 保持原配置，改变它的环境数时需要自行同步 batch。

容量实测使用 GPU PhysX、seed=1000、官方 checkpoint-20000，目标为每组两轮并在第 2 轮保存。成功组的第 2 轮迭代耗时不含保存，整卡峰值包含保存；余量统一按可分配的 24082 MiB 减去峰值计算。原始数据位于 `logs/num-envs-viser/`。

| PhysX 缓冲 | 环境数 | global batch | 第 2 轮耗时 / s | 控制步 / s | 整卡峰值 / MiB | 余量 / MiB | 退出码 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 原始 | 2 | 60 | 113.057 | 26.535 | 22935 | 1147 | 0 |
| 原始 | 4 | 120 | 156.626 | 38.308 | 23427 | 655 | 0 |
| 原始 | 8 | 240 | — | — | 24074 | 8 | 245（OOM 后受控终止） |
| 候选（已采用配置） | 2 | 60 | 114.871 | 26.116 | 22085 | 1997 | 0 |
| 候选（已采用配置） | 4 | 120 | 150.992 | 39.737 | 22379 | 1703 | 0 |
| 候选（已采用配置） | 8 | 240 | 219.583 | 54.649 | 23001 | 1081 | 0 |
| 生产当前默认 | 2 | 60 | 116.152 | 25.828 | 22103 | 1979 | 0 |

原始缓冲的 2env 数据来自 `baseline-n2-retry1/`。相对 R3c 约 24.55 步/s 的参考值快 8.10%，超过 5% 的复查线。复查确认 R3c 取首轮，本表取第 2 轮；同口径首轮耗时为本次 124.081 s、R3c 122.216 s，差约 1.53%。这里的 8.10% 不能作为实现提速结论。

原始缓冲的 8env 在第 2 轮权重同步的 rollout CUDA IPC 接收克隆阶段 OOM：申请 128 MiB 时仅剩 8.88 MiB。记录异常后受控终止，未完成第 2 轮或保存，因此不报告该轮吞吐；表中为退出前采样峰值。

候选数据来自 `candidate-n2/`、`candidate-n4/`、`candidate-n8/`，三组都完成两轮和保存。它们使用采用前的私有注入，三项缓冲设置与当前生产 `make_sim` 相同。生产当前默认一行来自 `viser-final-n2-off/`，使用最终生产代码、关闭 Viser，无候选注入，同样完成两轮和保存。原始 4env、8env 数据分别来自 `baseline-n4/`、`baseline-n8/`。

### PhysX 缓冲与验收范围

`make_sim` 在 `gym.make` 前，仅对 `PhysxCfg` 将 `env_cfg.sim.physics` 下的 `gpu_found_lost_pairs_capacity`、`gpu_found_lost_aggregate_pairs_capacity`、`gpu_total_aggregate_pairs_capacity` 统一设为 65536，对所有环境数生效；其余 8 个 `gpu_*` 字段保持原值，不增加配置开关。Newton 使用自己的后端参数。

4env 分阶段剖析中，CUDA 初始化和 AppLauncher 后的仿真进程显存没有变化；`gym.make` 后从 7247 降至 6703 MiB，首次 reset/render 后从 7249 降至 6705 MiB，两阶段均节省 544 MiB，整卡采样峰值差为 546 MiB，达到 500 MiB 的采用门槛。对应 `torch.cuda.memory_reserved` 两组均为 22 MiB。候选训练的 2env、4env 整卡峰值分别比原始缓冲低 850、1048 MiB；这是完整训练峰值的另一口径，不能全部归因于缓冲，也不能当作可保证的节省量。

默认 2env 的生产实测整卡峰值由原始缓冲的 22935 降至 22103 MiB，观察到下降 832 MiB，余量由 1147 增至 1979 MiB。这也是完整训练峰值的前后对照，不能将全部差值归因于缓冲；同阶段剖析支持的固定节省量仍为 544 MiB。

4env、seed=1000、300 控制步的站立动作对照中，两次原始缓冲运行及候选运行的关节位置、关节速度、箱子位姿均逐位一致。真实策略的 N=2、4、8 短跑中，stdout 和 Kit 日志的同一套溢出/容量警告检查均为零命中。

站立对照与零溢出通过不代表真实策略轨迹或成功率等价。4env 两轮 success 的原始值为 `[1.0, 0.25]`，候选为 `[0.75, 0.25]`，首轮低 0.25，`task_metrics_equal=false`。原始缓冲的 N=2 复跑也已出现轨迹和训练指标分叉；现有证据不足以确定差异来源，少量短跑不足以证明统计等价、排除小幅系统性影响，或外推到更大的 N 与长时间训练。

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

监控统计整卡显存总和，不是单进程显存。PhysX CPU 物理使用 `env.train.device=cpu`；默认物理设备为 `cuda:0`。正式训练需显式设置 `runner.max_epochs`、`runner.max_steps` 和 `runner.save_interval`。

## 恢复与指标

在上述训练参数末尾、日志管道之前追加 `"runner.resume_dir=$PREVIOUS_RUN/train/checkpoints/global_step_2"` 恢复；`PREVIOUS_RUN` 指向旧的运行目录，新运行使用不同的 `RUN_DIR`。resume_dir 指到 `global_step_N`，保持模型、训练范围、环境数和 batch 一致，将 max_epochs、max_steps 设置为恢复后的目标总步数。恢复模型、优化器和 scheduler，不保证仿真轨迹逐位续接。

恢复时由本次配置决定后端，允许 PhysX 与 Newton 双向切换，保留优化器和训练步数；仿真及 WBC 状态重新初始化。例如在恢复参数后追加 `env.train.physics_backend=newton_mjwarp`，内置 eval 默认随之切换，也可单独覆盖。`physics/run.json` 记录来源与当前后端；来源运行没有 `physics/run.json` 时记为 `unknown`，仍允许恢复。双向跨后端恢复尚未通过实机验收。导出后的模型也可在独立推理和评测中选择本次后端。

offload 配方启用 `algorithm.log_epoch_metrics`，记录 `actor/epoch_1..N/{clip_fraction,approx_kl,ratio,valid_count}` 和剔除 epoch1 的 `actor/epoch_ge_2/*`。按 PPO 实际广播后的有效 loss_mask 元素数加权，跨 rank 求和再相除；全零 mask 的整组均值为 NaN。`approx_kl` 使用有符号的 `-mean(log_ratio)`，`clip_fraction` 沿用上游考虑 advantage 的裁剪判定。原有跨 epoch 聚合指标保持原口径。

训练后使用 [导出与评测](eval.md) 的公开入口导出 safetensors。checkpoint、精度、actor 与导出一致性检查见 [测试说明](testing.md)；它们验证保存及推理链路，不代表 PPO 已改善策略成功率。
