# 闭环评测与 RL 导出

从项目根目录执行，环境准备见 [setup.md](setup.md)。评测需要 Arena 资产、CUDA 相机以及模型 checkpoint；首次运行会加载和编译场景资产。同一时间只运行一个仿真实例，启动前停止占用本地端口 5555 的推理服务器。

## 官方 G1 checkpoint

默认评测模型为 `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation`，revision `dfe74af855007f26093f362cd2d7a2f404b64b93`。只下载推理文件，约 6.2 GB：

```bash
HF_HOME="$PWD/cache/huggingface" uvx --from huggingface_hub hf download \
  nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
  --revision dfe74af855007f26093f362cd2d7a2f404b64b93 \
  --include '*.json' --include '*.yaml' --include '*.safetensors' \
  --local-dir models/GR00T-N1.6-G1/checkpoint-20000
```

`models/`、`logs/` 等大型产物通过项目软链接写入大容量存储；Python 环境保留在本地 `.venvs/`。

## 物理后端与 CPU/GPU 基线

`scripts/eval/baseline.sh OUTPUT_DIR [cpu|gpu|both] [episodes] [num_envs] [model_path] [seed] [--presets physx|newton]` 是完整评测入口。默认使用 `physx`，设备模式默认 `both`，即顺序运行 PhysX CPU 和 GPU；每种设备 100 个 episode、5 个并行环境、G1 checkpoint、seed 42。输出目录中对应的 `cpu/`、`gpu/` 子目录必须不存在。

`--presets` 可出现在任意位置，后续可选位置参数可以省略；`physx` 和 `newton` 分别选择 PhysX 和 Newton MJWarp。Newton 只支持 `gpu`，与 `cpu` 或 `both` 组合会在启动服务器和仿真前报错。每次命令选择一个物理引擎，跨引擎比较时分别运行并使用不同输出目录。

后端选择沿用 [Arena 官方物理配置](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/concepts/environment/physics_configuration.html)与[选择优先级](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/concepts/environment/physics_backend_selection.html)：显式 `--presets` 优先于环境的 `default_physics_backend`。本文 baseline 默认传入 `physx`；底层 Arena 未指定 preset 时才使用环境默认值。

```bash
RUN_DIR="$PWD/logs/baseline/$(date +%Y%m%d-%H%M%S)"
scripts/eval/baseline.sh "$RUN_DIR" both 100 5

# Newton GPU。
NEWTON_DIR="$PWD/logs/baseline-newton/$(date +%Y%m%d-%H%M%S)"
scripts/eval/baseline.sh "$NEWTON_DIR" gpu 100 5 \
  "$PWD/models/GR00T-N1.6-G1/checkpoint-20000" 42 \
  --presets newton
```

例如 `scripts/eval/baseline.sh "$NEWTON_DIR" gpu --presets newton` 使用其余默认值。脚本启动推理服务器、仿真和 GPU 监控，并在每种设备运行结束时清理子进程。`both` 顺序运行 PhysX CPU/GPU，分别重启服务器并设置推理随机种子。CPU/GPU 指物理设备，模型推理与相机仍需 GPU。在启动命令的终端按 Ctrl+C 停止运行。

协议固定为 50 Hz、每个 episode 最多 1500 控制步、动作块 50 步，任务为 `galileo_g1_locomanip_pick_and_place`、物体 `brown_box`、机器人 `g1_wbc_joint`。`protocol.json` 记录请求与实际完整后端名、物理设备、配置类、manager、实际求解器参数、步长、相机、包版本，以及 checkpoint、seed、环境数和推理栈；原有 `physics` 字段继续记录配置类。`newton_solver_sha256` 区分同一 Newton 版本号下的 solver 源码变化，包括安装时自动应用的MuJoCo 导出与初始化兼容修复，见 [环境准备](setup.md)。固定 seed 不能保证仿真逐位复现。

| 产物 | 内容 |
| --- | --- |
| `<device>/episodes.jsonl` | 逐局成功、终止项、进度及摔倒诊断 |
| `<device>/protocol.json` | 实际协议与运行环境 |
| `<device>/complete.json` | 完整结束后写入的 episode 数、成功数、耗时及 CUDA 峰值 |
| `server-<device>.log`、`eval-<device>.log`、`kit-<device>.log` | 模型服务器、评测及仿真日志 |
| `gpu-<device>.csv` | 每 5 秒采样的显存与利用率 |
| `summary.json`、`report.md`、`summary-stdout.log` | PhysX `both` 模式下两种设备均完成后自动生成的汇总 |

`<device>` 为 `cpu` 或 `gpu`；Newton 产物也使用 `gpu/`，以 `protocol.json` 中的后端名区分引擎。只运行单个设备时保留逐局产物，不会生成 CPU/GPU 汇总。命令退出码为 0 和 `complete.json` 共同证明对应运行完整结束；模型加载日志或部分逐局记录不足以证明完成。

Newton 已完成官方 checkpoint、seed 42、2 个环境的 10 局闭环评测：成功 0/10、摔倒 0/10、掉箱 0/10，均在 1500 步超时；共 15000 控制步，rollout 耗时 624.65 秒，约 24.01 步/秒，整卡采样峰值 17553 MiB。该运行使用 reference/SDPA、逐样本推理，未出现接触或约束容量溢出。原始结果位于 `logs/physics-backends/newton-baseline-01/`。这验证了多局运行与 reset，不代表策略已能在 Newton 下完成抓放，也不能据此比较两种引擎的效果。

成功使用 Arena 原生 success 信号。失败分类优先记录官方掉箱终止，再区分已抬起但未放入、接近但未抬起、未接近；接近用底盘距箱子初始 XY ≤0.5 m 作为导航代理。抬起 predicate 仅表示官方进度条件，reset 后箱体回弹也可能触发。`fall_once` 是骨盆相对默认站立位置下降超过 0.4 m 的独立诊断，不改变成功判定。汇总使用 Wilson 95% 区间，记录 CPU/GPU 区间是否重叠；不能把不同协议的结果直接比较。

## 对比 SFT 与 RL checkpoint

`scripts/eval/compare-checkpoints.sh SFT_CHECKPOINT RL_WEIGHTS [gpu|cpu|both] [episodes] [num_envs] [seed] [output_dir]` 依次完成 RL 导出、SFT 评测、RL 评测和指标汇总。SFT 路径必须是 RL 训练的实际起点；导出以该目录为基座，保留它的 processor 和归一化统计量。两次评测统一使用 reference/SDPA、BF16、逐样本推理。

```bash
scripts/eval/compare-checkpoints.sh \
  "$PWD/models/GR00T-N1.6-g1-full/checkpoint-20000" \
  "$PWD/logs/rl-training/your-run/train/checkpoints/global_step_125/actor/model_state_dict/full_weights.pt" \
  gpu 100 5 2000
```

此脚本使用 PhysX，默认 GPU 物理、每个模型 100 局、5 个环境、seed 2000；改成 `both` 会顺序评测 PhysX CPU/GPU，每个模型每种设备各 100 局。需要快速检查流程时可将局数改为 5。任一步失败即停止，使用新的输出目录重新运行。其他物理引擎通过上面的 baseline 入口分别评测模型。

导出模型保存在 `models/rl-export-<timestamp>/`。默认评测目录为 `logs/checkpoint-eval/<timestamp>/`，可用最后一个参数指定不存在的新目录：

| 产物 | 内容 |
| --- | --- |
| `sft/<device>/`、`rl/<device>/` | 各模型的逐局记录、协议与完成凭证 |
| `comparison.json` | 每种设备的两模型统计、失败分类、耗时与成功率差值（RL − SFT，0.1 表示增加 10 个百分点） |
| `report.md` | 成功数、成功率、Wilson 95% 区间、摔倒率、耗时和失败分类 |

统计与报告生成由 `embodied_ai_lab.eval.compare_checkpoints` 模块负责；它要求 SFT/RL 使用同一物理引擎，报告同时显示引擎与 CPU/GPU 设备。旧协议缺少 `physics_backend` 时按 PhysX 读取。已有完整评测记录时，可以单独重新生成汇总，无需再次运行仿真：

```bash
python3 -m embodied_ai_lab.eval.compare_checkpoints \
  logs/checkpoint-eval/你的评测目录 gpu
```

脚本退出码为 0 且生成最终报告，表示两模型均已完成评测与汇总。成功率差值只描述本次结果；各模型的 Wilson 区间不等同于差值置信区间。筛选多个 checkpoint 时使用验证条件，最终比较另用未参与筛选的测试条件。

## 单局观察

`groot-server.sh` 固定加载默认 G1 checkpoint，监听 `127.0.0.1:5555`；`arena.sh` 等待服务器就绪后执行 1 个 episode、1 个环境、seed 42、CPU 物理。在两个终端中分别运行，均从仓库根目录启动。

终端 1 启动服务器并保持运行：

```bash
scripts/eval/groot-server.sh
```

终端 2 启动客户端：

```bash
scripts/eval/arena.sh --record_camera_video

# 同一服务器，改用 Newton GPU 闭环执行。
scripts/eval/arena.sh --presets newton --device cuda:0
```

客户端产物位于 `logs/evaluation/`，每次运行的输出目录含 `protocol_rank0.json`，由实际仿真配置生成后端与设备记录；多 rank 时各 rank 分别记录。两侧日志分别追加到 `logs/groot-server.log` 和 `logs/arena-eval.log`，Kit 日志为 `logs/kit.log`。两个客户端命令顺序执行；完成后在终端 1 按 Ctrl+C 停止服务器并释放模型。其余客户端参数原样传给上游 policy runner。

Arena 统一使用已有的 `--presets physx|newton`；请求后端在协议中记为完整名称 `isaacsim_physx` 或 `newton_mjwarp`。未指定 preset 时请求值为 `null`，实际后端由环境配置生成。Newton 必须显式选择 CUDA 物理设备，脚本原有默认 CPU 不会自动变更。模型服务器无需物理配置；后端选择发生在仿真客户端创建环境时。浏览器查看器、Kit 串流和双端断点见 [debugging.md](debugging.md)。单局用于观察行为，不用于估计成功率。

## 导出 RL 权重

先完成 [RL 训练](rl.md)，取得 `global_step_N/actor/model_state_dict/full_weights.pt`。`scripts/eval/export-rl.sh --help` 可查看参数；`--weights` 和 `--output` 必填，`--base-model` 默认为上述 G1 checkpoint。输出必须是不存在的新目录：

```bash
# RUN_DIR 是 rl.md 中两迭代短跑的运行目录。
WEIGHTS_FILE="$RUN_DIR/train/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
MODEL_DIR="$PWD/models/GR00T-N1.6-PPO-$(date +%Y%m%d-%H%M%S)"
scripts/eval/export-rl.sh --weights "$WEIGHTS_FILE" --output "$MODEL_DIR"
```

示例接续 [RL 短跑](rl.md#两迭代短跑)；新终端中先将 `RUN_DIR` 设为该次运行目录。评测其他实验时改为其实际产物。导出移除 `action_head.value_head.*`，严格核对其余参数名与基座索引，按官方分片索引写入 BF16 safetensors，并复制 config、processor、statistics、embodiment 映射及 experiment_cfg。`rl_export.json` 记录来源权重、基座和推理栈。

确认导出命令退出码为 0 且生成 `rl_export.json` 后，以相同评测入口指定模型和 seed：

```bash
EVAL_DIR="$PWD/logs/rl-eval/$(date +%Y%m%d-%H%M%S)"
scripts/eval/baseline.sh "$EVAL_DIR" gpu 50 5 "$MODEL_DIR" 2000
```

带 `rl_export.json` 的目录自动使用 `.venvs/rl`、`Isaac-GR00T-reference` 的 SDPA 和逐样本 BF16 前向，与本地 RL rollout 一致。两侧使用 480×640 图像和归一化前 FP32 状态。评测官方起点作为对照时，在新的运行中用 `EVAL_RL_SERVER=1` 显式选择相同推理栈：

```bash
EVAL_RL_SERVER=1 scripts/eval/baseline.sh \
  "$PWD/logs/reference-eval/$(date +%Y%m%d-%H%M%S)" gpu 50 5 \
  "$PWD/models/GR00T-N1.6-G1/checkpoint-20000" 2000
```

该对照应在前一个评测命令结束后启动。BF16 batch 形状与 FlashAttention/SDPA 会改变动作，比较时保持推理栈、物理后端、seed、episode 数、环境数和步数协议一致。当前 fork 的 Euler 去噪顺序也影响训练 rollout 中使用 eval 模式的确定性去噪步，见 [rl.md](rl.md)。导出前后动作、初始噪声及真实 server/client 的一致性检查已纳入 GPU 测试，前置产物和运行方式见 [testing.md](testing.md)。
