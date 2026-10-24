# 闭环评测与 RL 导出

从项目根目录执行，环境准备见 [setup.md](setup.md)。评测需要 Arena 资产、CUDA 相机以及模型 checkpoint；首次运行会加载和编译场景资产。同一时间只运行一个仿真实例，启动新的推理服务器前确认本地端口 5555 空闲。

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

## 按官方教程简单评测 SFT checkpoint

沿用 [Arena 官方闭环评测教程](../third_party/IsaacLab-Arena/docs/pages/example_workflows/locomanipulation/step_5_evaluation.rst)的流程：启动 GR00T 推理服务器，再启动 Arena 客户端，读取原生评测指标。两个终端均从项目根目录执行。

终端 1 启动服务器并保持运行：

```bash
scripts/eval/groot-server.sh
```

服务器固定加载上述官方 G1 `checkpoint-20000`，监听 `127.0.0.1:5555`，使用 Arena 内嵌的官方 GR00T 推理入口。

终端 2 按教程运行 1500 个控制步：

```bash
scripts/eval/arena.sh --num_steps 1500
```

需要简单统计多局成功率时，改为完整运行 10 个 episode：

```bash
scripts/eval/arena.sh --num_episodes 10
```

`--num_steps` 限制整次运行的总控制步数，可能在某局结束前停止；`--num_episodes` 按完整局数停止。选择一种即可，统计成功率时使用完整多局。`arena.sh` 默认运行 1 局、1 个环境、seed 42，追加的参数会覆盖对应默认值；设置 `--num_steps` 后，上游 runner 优先使用步数限制。

默认使用 CPU 物理、GPU 模型推理与相机；官方教程说明该策略使用 CPU 物理生成的数据训练。需要录像时追加 `--record_camera_video`，例如：

```bash
scripts/eval/arena.sh --num_episodes 10 --record_camera_video
```

评测结束时查看终端的 `Metrics`，其中 `num_episodes` 是完成局数，`success_rate` 是成功局数占比。产物位于 `logs/evaluation/` 的时间戳子目录，包括 `episode_results_rank0.jsonl`、评测 HTML 报告及启用的视频。两侧日志分别追加到 `logs/groot-server.log` 和 `logs/arena-eval.log`，Kit 日志为 `logs/kit.log`。客户端退出码为 0 后，在终端 1 按 Ctrl+C 停止服务器并释放模型。

### 更换为自己训练的 SFT checkpoint

`groot-server.sh` 的模型路径固定。评测其他 G1 SFT checkpoint 时，在终端 1 改用以下官方 server 命令，将 `CHECKPOINT` 设为实际目录；终端 2 的客户端命令相同：

```bash
source scripts/env.sh
CHECKPOINT="$LAB_ROOT/models/你的SFT运行/checkpoint-20000"
cd "$GROOT_ROOT"
"$GROOT_PY" -m gr00t.eval.run_gr00t_server \
  --modality-config-path "$ARENA_ROOT/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py" \
  --model-path "$CHECKPOINT" \
  --embodiment-tag NEW_EMBODIMENT --device cuda --host 127.0.0.1 --port 5555
```

客户端其他参数原样传给上游 policy runner；浏览器查看器、Kit 串流和双端断点见 [debugging.md](debugging.md)。需要 CPU/GPU 对照、额外失败诊断和置信区间汇总时，使用下面的基线评测入口。

## CPU/GPU 基线

`scripts/eval/baseline.sh OUTPUT_DIR [cpu|gpu|both] [episodes] [num_envs] [model_path] [seed]` 自动编排基线评测与统计。默认后端 `both`，每个后端 100 个 episode、5 个并行环境、G1 checkpoint、seed 42。输出目录中对应的 `cpu/`、`gpu/` 子目录必须不存在。

```bash
RUN_DIR="$PWD/logs/baseline/$(date +%Y%m%d-%H%M%S)"
scripts/eval/baseline.sh "$RUN_DIR" both 100 5
```

脚本启动推理服务器、仿真和 GPU 监控，并在每个后端结束时清理子进程。两个后端顺序运行，分别重启服务器并设置推理随机种子。CPU/GPU 指物理设备，模型推理与相机仍需 GPU。在启动命令的终端按 Ctrl+C 停止运行。

协议固定为 50 Hz、每个 episode 最多 1500 控制步、动作块 50 步，任务为 `galileo_g1_locomanip_pick_and_place`、物体 `brown_box`、机器人 `g1_wbc_joint`。`protocol.json` 记录实际 checkpoint、seed、环境数、物理设备、相机、包版本及推理栈；固定 seed 不能保证 PhysX 逐位复现。

| 产物 | 内容 |
| --- | --- |
| `<backend>/episodes.jsonl` | 逐局成功、终止项、进度及摔倒诊断 |
| `<backend>/protocol.json` | 实际协议与运行环境 |
| `<backend>/complete.json` | 完整结束后写入的 episode 数、成功数、耗时及 CUDA 峰值 |
| `server-<backend>.log`、`eval-<backend>.log`、`kit-<backend>.log` | 模型服务器、评测及仿真日志 |
| `gpu-<backend>.csv` | 每 5 秒采样 `nvidia-smi` 返回的所有 GPU 的显存与利用率 |
| `summary.json`、`report.md`、`summary-stdout.log` | `both` 模式下两个后端均完成后自动生成的汇总 |

只运行单个后端时保留逐局产物，不会生成 CPU/GPU 汇总。命令退出码为 0 和 `complete.json` 共同证明对应运行完整结束；模型加载日志或部分逐局记录不足以证明完成。

汇总字段 `peak_host_single_gpu_memory_mib` 是上述所有 GPU、所有采样时刻中的单卡最大显存占用，包含推理服务器和其他进程；多卡共享主机可能取到与评测无关的卡。它不是多卡用量之和，也不是评测进程独占用量；5 秒采样可能遗漏瞬时峰值。`complete.json` 中的 CUDA 峰值则来自仿真进程内的 PyTorch 统计。

成功使用 Arena 原生 success 信号。失败分类优先记录官方掉箱终止，再区分已抬起但未放入、接近但未抬起、未接近；接近用底盘距箱子初始 XY ≤0.5 m 作为导航代理。抬起 predicate 仅表示官方进度条件，reset 后箱体回弹也可能触发。`fall_once` 是骨盆相对默认站立位置下降超过 0.4 m 的独立诊断，不改变成功判定。汇总使用 Wilson 95% 区间，记录 CPU/GPU 区间是否重叠；不能把不同协议的结果直接比较。

## 对比 SFT 与 RL checkpoint

`scripts/eval/compare-checkpoints.sh SFT_CHECKPOINT RL_WEIGHTS [gpu|cpu|both] [episodes] [num_envs] [seed] [output_dir]` 依次完成 RL 导出、SFT 评测、RL 评测和指标汇总。SFT 路径必须是 RL 训练的实际起点；导出以该目录为基座，保留它的 processor 和归一化统计量。两次评测统一使用 reference/SDPA、BF16、逐样本推理。

```bash
scripts/eval/compare-checkpoints.sh \
  "$PWD/models/GR00T-N1.6-g1-full/checkpoint-20000" \
  "$PWD/logs/rl-training/your-run/train/checkpoints/global_step_125/actor/model_state_dict/full_weights.pt" \
  gpu 100 5 2000
```

默认 GPU 物理、每个模型 100 局、5 个环境、seed 2000；改成 `both` 会顺序评测两个物理后端，每个模型每个后端各 100 局。需要快速检查流程时可将局数改为 5。任一步失败即停止，使用新的输出目录重新运行。

导出模型保存在 `models/rl-export-<timestamp>/`。默认评测目录为 `logs/checkpoint-eval/<timestamp>/`，可用最后一个参数指定不存在的新目录：

| 产物 | 内容 |
| --- | --- |
| `sft/<backend>/`、`rl/<backend>/` | 各模型的逐局记录、协议与完成凭证 |
| `comparison.json` | 每个后端的两模型统计、失败分类、耗时与成功率差值（RL − SFT，0.1 表示增加 10 个百分点） |
| `report.md` | 成功数、成功率、Wilson 95% 区间、摔倒率、耗时和失败分类 |

统计与报告生成由 `embodied_ai_lab.eval.compare_checkpoints` 模块负责。已有完整评测记录时，可以单独重新生成汇总，无需再次运行仿真：

```bash
python3 -m embodied_ai_lab.eval.compare_checkpoints \
  logs/checkpoint-eval/你的评测目录 gpu
```

脚本退出码为 0 且生成最终报告，表示两模型均已完成评测与汇总。成功率差值只描述本次结果；各模型的 Wilson 区间不等同于差值置信区间。筛选多个 checkpoint 时使用验证条件，最终比较另用未参与筛选的测试条件。

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
