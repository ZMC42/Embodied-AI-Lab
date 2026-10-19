# 闭环评测与 RL 导出

从项目根目录执行，环境准备见 [setup.md](setup.md)。评测需要 Arena 资产、CUDA 相机以及模型 checkpoint；首次运行会加载和编译场景资产。同一时间只运行一个仿真实例，启动前停止占用本地端口 5555 的推理服务器。

所有持续运行的下载、推理服务器、仿真与导出任务均由 `systemd --user` 临时服务托管。启动前检查：

```bash
loginctl show-user "$USER" -p Linger
```

必须显示 `Linger=yes`；否则执行 `loginctl enable-linger "$USER"` 并重新确认。不要启用服务开机自动启动。下文命令不使用 `systemctl enable`。

## 官方 G1 checkpoint

默认评测模型为 `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation`，revision `dfe74af855007f26093f362cd2d7a2f404b64b93`。只下载推理文件，约 6.2 GB：

```bash
systemd-run --user --unit="lab-download-g1-$(date +%s)" \
  --property="WorkingDirectory=$PWD" --property=Type=exec --property=RemainAfterExit=yes \
  --setenv="HF_HOME=$PWD/cache/huggingface" \
  uvx --from huggingface_hub hf download \
  nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
  --revision dfe74af855007f26093f362cd2d7a2f404b64b93 \
  --include '*.json' --include '*.yaml' --include '*.safetensors' \
  --local-dir models/GR00T-N1.6-G1/checkpoint-20000
```

`models/`、`logs/` 等大型产物通过项目软链接写入大容量存储；Python 环境保留在本地 `.venvs/`。

## CPU/GPU 基线

`scripts/eval/baseline.sh OUTPUT_DIR [cpu|gpu|both] [episodes] [num_envs] [model_path] [seed]` 是完整评测入口。默认后端 `both`，每个后端 100 个 episode、5 个并行环境、G1 checkpoint、seed 42。输出目录中对应的 `cpu/`、`gpu/` 子目录必须不存在。

```bash
RUN_DIR="$PWD/logs/baseline/$(date +%Y%m%d-%H%M%S)"
UNIT="lab-baseline-$(date +%s)"
mkdir -p "$RUN_DIR"
systemd-run --user --unit="$UNIT" \
  --property="WorkingDirectory=$PWD" --property=Type=exec --property=RemainAfterExit=yes \
  --property=TimeoutStopSec=30 \
  --property="StandardOutput=append:$RUN_DIR/systemd.log" \
  --property=StandardError=inherit \
  "$PWD/scripts/eval/baseline.sh" "$RUN_DIR" both 100 5

systemctl --user show "$UNIT" -p ActiveState -p SubState -p Result -p ExecMainStatus
journalctl --user -u "$UNIT" -f
```

服务器、仿真和 GPU 监控处于同一个服务的 cgroup；脚本在后端结束或服务停止时清理子进程。两个后端顺序运行，分别重启服务器并设置推理随机种子。CPU/GPU 指物理设备，模型推理与相机仍需 GPU。停止运行使用 `systemctl --user stop "$UNIT"`。

协议固定为 50 Hz、每个 episode 最多 1500 控制步、动作块 50 步，任务为 `galileo_g1_locomanip_pick_and_place`、物体 `brown_box`、机器人 `g1_wbc_joint`。`protocol.json` 记录实际 checkpoint、seed、环境数、物理设备、相机、包版本及推理栈；固定 seed 不能保证 PhysX 逐位复现。

| 产物 | 内容 |
| --- | --- |
| `<backend>/episodes.jsonl` | 逐局成功、终止项、进度及摔倒诊断 |
| `<backend>/protocol.json` | 实际协议与运行环境 |
| `<backend>/complete.json` | 完整结束后写入的 episode 数、成功数、耗时及 CUDA 峰值 |
| `server-<backend>.log`、`eval-<backend>.log`、`kit-<backend>.log` | 模型服务器、评测及仿真日志 |
| `gpu-<backend>.csv` | 每 5 秒采样的显存与利用率 |
| `summary.json`、`report.md`、`summary-stdout.log` | `both` 模式下两个后端均完成后自动生成的汇总 |

只运行单个后端时保留逐局产物，不会生成 CPU/GPU 汇总。服务成功退出和 `complete.json` 共同证明对应运行完整结束；模型加载日志或部分逐局记录不足以证明完成。

成功使用 Arena 原生 success 信号。失败分类优先记录官方掉箱终止，再区分已抬起但未放入、接近但未抬起、未接近；接近用底盘距箱子初始 XY ≤0.5 m 作为导航代理。抬起 predicate 仅表示官方进度条件，reset 后箱体回弹也可能触发。`fall_once` 是骨盆相对默认站立位置下降超过 0.4 m 的独立诊断，不改变成功判定。汇总使用 Wilson 95% 区间，记录 CPU/GPU 区间是否重叠；不能把不同协议的结果直接比较。

## 单局观察

`groot-server.sh` 固定加载默认 G1 checkpoint，监听 `127.0.0.1:5555`；`arena.sh` 等待服务器就绪后执行 1 个 episode、1 个环境、seed 42、CPU 物理。分别托管两个服务：

```bash
SERVER_UNIT="lab-groot-server-$(date +%s)"
CLIENT_UNIT="lab-arena-$(date +%s)"
systemd-run --user --unit="$SERVER_UNIT" \
  --property="WorkingDirectory=$PWD" --property=Type=exec --property=RemainAfterExit=yes \
  "$PWD/scripts/eval/groot-server.sh"
systemd-run --user --unit="$CLIENT_UNIT" \
  --property="WorkingDirectory=$PWD" --property=Type=exec --property=RemainAfterExit=yes \
  "$PWD/scripts/eval/arena.sh" --record_camera_video
```

客户端产物位于 `logs/evaluation/`，两侧日志分别追加到 `logs/groot-server.log` 和 `logs/arena-eval.log`，Kit 日志为 `logs/kit.log`。用上文 `systemctl` 命令检查客户端结束，再执行 `systemctl --user stop "$SERVER_UNIT"` 释放模型。其余客户端参数原样传给上游 policy runner；浏览器查看器、Kit 串流和双端断点见 [debugging.md](debugging.md)。单局用于观察行为，不用于估计成功率。

## 导出 RL 权重

先完成 [RL 训练](rl.md)，取得 `global_step_N/actor/model_state_dict/full_weights.pt`。`scripts/eval/export-rl.sh --help` 可查看参数；`--weights` 和 `--output` 必填，`--base-model` 默认为上述 G1 checkpoint。输出必须是不存在的新目录：

```bash
# RUN_DIR 是 rl.md 中两迭代短跑的运行目录。
WEIGHTS_FILE="$RUN_DIR/train/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
MODEL_DIR="$PWD/models/GR00T-N1.6-PPO-$(date +%Y%m%d-%H%M%S)"
UNIT="lab-rl-export-$(date +%s)"
systemd-run --user --unit="$UNIT" \
  --property="WorkingDirectory=$PWD" --property=Type=exec --property=RemainAfterExit=yes \
  "$PWD/scripts/eval/export-rl.sh" \
  --weights "$WEIGHTS_FILE" --output "$MODEL_DIR"

systemctl --user show "$UNIT" -p ActiveState -p SubState -p Result -p ExecMainStatus
journalctl --user -u "$UNIT" -f
```

示例接续 [RL 短跑](rl.md#两迭代短跑)；新终端中先将 `RUN_DIR` 设为该次运行目录。评测其他实验时改为其实际产物。导出移除 `action_head.value_head.*`，严格核对其余参数名与基座索引，按官方分片索引写入 BF16 safetensors，并复制 config、processor、statistics、embodiment 映射及 experiment_cfg。`rl_export.json` 记录来源权重、基座和推理栈。

确认导出服务正常结束后，以相同评测入口指定模型和 seed：

```bash
EVAL_DIR="$PWD/logs/rl-eval/$(date +%Y%m%d-%H%M%S)"
systemd-run --user --unit="lab-rl-eval-$(date +%s)" \
  --property="WorkingDirectory=$PWD" --property=Type=exec --property=RemainAfterExit=yes \
  "$PWD/scripts/eval/baseline.sh" "$EVAL_DIR" gpu 50 5 "$MODEL_DIR" 2000
```

带 `rl_export.json` 的目录自动使用 `.venvs/rl`、`Isaac-GR00T-reference` 的 SDPA 和逐样本 BF16 前向，与本地 RL rollout 一致。两侧使用 480×640 图像和归一化前 FP32 状态。评测官方起点作为对照时，在新的服务中用 `EVAL_RL_SERVER=1` 显式选择相同推理栈：

```bash
systemd-run --user --unit="lab-reference-eval-$(date +%s)" \
  --property="WorkingDirectory=$PWD" --property=Type=exec --property=RemainAfterExit=yes \
  --setenv=EVAL_RL_SERVER=1 \
  "$PWD/scripts/eval/baseline.sh" \
  "$PWD/logs/reference-eval/$(date +%Y%m%d-%H%M%S)" gpu 50 5 \
  "$PWD/models/GR00T-N1.6-G1/checkpoint-20000" 2000
```

该对照应在前一个评测服务结束后启动。BF16 batch 形状与 FlashAttention/SDPA 会改变动作，比较时保持推理栈、物理后端、seed、episode 数、环境数和步数协议一致。当前 fork 的 Euler 去噪顺序也影响训练 rollout 中使用 eval 模式的确定性去噪步，见 [rl.md](rl.md)。导出前后动作、初始噪声及真实 server/client 的一致性检查已纳入 GPU 测试，前置产物和运行方式见 [testing.md](testing.md)。
