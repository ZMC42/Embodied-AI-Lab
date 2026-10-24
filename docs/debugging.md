# 调试与可视化

先完成 [环境准备](setup.md)，在 VSCode（远程机器使用 Remote SSH）中打开仓库根目录，安装 Python 与 Python Debugger 扩展。`.vscode/settings.json` 默认使用 `.venvs/arena`；server、client 和等待任务分别显式选择 `.venvs/groot`、`.venvs/arena`、`.venvs/groot`。

## 双端断点

准备 [G1 checkpoint](eval.md#官方-g1-checkpoint)，先停止占用 5555 端口的推理服务器，再启动 compound `GR00T server + Arena client`。`prepare runtime` 任务执行 `scripts/env.sh` 创建运行目录和资产链接；client 在 ZMQ ping 成功后才启动，等待超时为 600 秒。服务器和客户端的工作目录均保持上游源码目录。

`.vscode/runtime.env` 仅包含 EULA 与无缓冲输出设置，无凭据。launch 配置显式设置 `HF_HOME`、`TMPDIR`，客户端优先加载 Arena venv 内的 CUDA runtime，避免 WebRTC 扩展自带的 CUDA runtime 与 PyTorch 冲突。

建议断点：

- 客户端 `third_party/IsaacLab-Arena/isaaclab_arena_gr00t/policy/gr00t_remote_closedloop_policy.py` 的 `_client.get_action(...)`。
- 服务器 `third_party/IsaacLab-Arena/submodules/Isaac-GR00T/gr00t/policy/gr00t_policy.py` 的 `self.model.get_action(...)`。

两端均为 `justMyCode=false`。服务器停在断点时，客户端等待 ZMQ 返回是预期行为。继续后 action 应返回并推进仿真；compound 的 `stopAll=true` 让停止操作结束两侧调试进程。调试产物位于 `logs/debug/`，Kit 日志为 `logs/debug-kit.log`。该模式用于交互调试；完整评测使用 [评测文档](eval.md) 的基线入口。

## 训练断点调试

### SFT

使用单张可见 GPU、smoke 配方和新输出目录。在终端中启动训练：

```bash
CUDA_VISIBLE_DEVICES=0 SFT_DEBUG=1 scripts/train/sft.sh smoke -- \
  --output-dir "$PWD/models/sft-debug-$(date +%Y%m%d-%H%M%S)" \
  --max-steps 2 --save-steps 2
```

保持启动终端运行，在 VSCode 选择 `GR00T N1.6 SFT training (attach)`，连接远程机器本地 `127.0.0.1:5678`。该端口只绑定 loopback；Remote SSH 中调试器运行在远程机器。smoke 的 loader workers=0，使样本处理留在训练进程。可在数据集的 `extract_step_data()`、`Gr00tN1d6.forward()`、`Gr00tN1d6ActionHead.forward()` 和 Transformers `Trainer.training_step()` 下断点。

断开 attach 后训练继续；停止训练在启动终端按 Ctrl+C。日志为 `logs/train-sft/smoke.log`，也可在另一个终端执行 `tail -f logs/train-sft/smoke.log`。不要同时启动两个等待 5678 的任务。

### RL / PPO

RL 使用 `.venvs/rl`。安装脚本已包含 `debugpy`；已有环境若尚未安装，执行 `uv pip install --python .venvs/rl/bin/python --no-deps 'debugpy==1.8.21'`。`SFT_DEBUG=1` 仍只用于 SFT；RL 用 `RL_DEBUG` 选择本次要调试的一个进程：

| RL_DEBUG | VSCode 配置 | loopback 端口 | 等待位置与可用断点 |
| --- | --- | --- | --- |
| `driver` | `RLinf PPO driver (attach)` | 5679 | 进入上游训练入口前；`EmbodiedRunner.run()`、`init_workers()`、调度与保存 |
| `actor` | `RLinf PPO actor (attach)` | 5680 | rank 0 的模型工厂加载模型前；`build_model()`、`compute_advantages_and_returns()`、`train_micro_batch()`、PPO loss |
| `rollout` | `RLinf PPO rollout (attach)` | 5681 | rank 0 的模型工厂加载模型前；`predict_g1()`、`get_rl_action()`、`predict_micro_batches()` |
| `env` | `RLinf PPO env (attach)` | 5682 | 首个训练仿真子进程启动 AppLauncher 前；`make_sim()`、`ChunkEnv.step()`、G1 动作处理与终局观测 |

单次运行只选一个角色；其他进程会等待该进程继续。模型工厂根据 worker 的实际配置组名选择 actor / rollout，不要求组名固定为默认值。env 只选 `seed_offset=0` 的训练实例，排除其他 pipeline stage 和 eval 实例。这里的入门流程使用默认单卡、每组单 rank、1 个 pipeline stage，不同时启动多个调试训练。

从仓库根目录启动两迭代调试短跑，使用新目录；将 `RL_DEBUG=actor` 改成表中所需角色：

```bash
RUN_DIR="$PWD/logs/rl-training/debug-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
set -o pipefail
CUDA_VISIBLE_DEVICES=0 RL_DEBUG=actor scripts/train/rl.sh ppo-offload \
  "runner.logger.log_path=$RUN_DIR" runner.logger.experiment_name=train \
  runner.max_epochs=2 runner.max_steps=2 runner.save_interval=2 \
  2>&1 | tee "$RUN_DIR/run.log"
```

等日志出现 `[RL debug] actor pid=... waiting on 127.0.0.1:5680`（其他角色对应其名称和端口），保持启动终端运行，再在 VSCode Remote SSH 选择相同角色的 attach 配置并 F5。worker 调度与导入也需要时间，端口尚未监听时不要提前 attach。端口只绑定远程机器 loopback，Remote SSH 中不需要额外转发。

attach 后在相应进程文件中设置断点并继续；`justMyCode=false` 允许进入 RLinf、GR00T reference 和 Isaac Lab。driver 断点不能暂停其他 Ray worker，env worker 的父进程也不执行真实物理步。模型断点应放在 `third_party/Isaac-GR00T-reference` 或 `third_party/RLinf` 对应实现；不要放在 SFT 使用的另一份 GR00T 中。

未设置 `RL_DEBUG` 时，训练不监听端口、不等待客户端，也不导入 debugpy。断开 attach 后训练继续；暂停某一进程时，其他进程等待通信是正常的，但长时间停顿仍可能触发分布式通信超时。只断开调试器不会终止训练，停止任务在启动终端按 Ctrl+C。

正常结束时命令退出码为 0，并生成 `train/checkpoints/global_step_2/`。训练与产物要求仍见 [RL 两迭代短跑](rl.md#两迭代短跑)。

## 远程观察

Arena 和 RL 安装脚本均包含 `viser==1.1.1`。使用 [SFT 简单评测](eval.md#按官方教程简单评测-sft-checkpoint) 的命令，将客户端尾参加上 `--viz viser`。在 VSCode 转发 TCP 8080，再打开 `http://localhost:8080`；有网络直连时可访问机器的 8080 端口。Viser 默认监听 `0.0.0.0:8080`，支持视角、暂停和重置。它同步 PhysX 状态，显示材质与 Kit 有差异；模型所需 RTX 相机仍然运行。

RL 训练时，在 [两迭代短跑](rl.md#两迭代短跑) 的训练参数末尾、日志管道之前追加 `env.train.viser_envs=1`，或用 `env.train.viser_envs=k` 最多显示 k 个环境；默认值 0 关闭查看器。端口固定为 8080，同一时间只运行一个仿真。这个开关仅用于调试短跑，正式长跑保持关闭。

以下开销对照使用 2 个环境，各跑两轮，分别记录首次与稳态 rollout；整卡峰值包含保存。原始数据位于 `logs/num-envs-viser/`。

| viser_envs | 首轮 rollout / s | 第 2 轮 rollout / s | 整卡峰值 / MiB | 相对关闭时峰值增量 / MiB |
| --- | --- | --- | --- | --- |
| 0 | 103.456 | 94.497 | 22935 | 0 |
| 1 | 105.292 | 99.371 | 23389 | +454 |

对应原始目录为关闭时的 `baseline-n2-retry1/` 与开启时的 `viser-n2-on/`，两组均使用原始 PhysX 缓冲。首轮 rollout 增加约 1.77%，第 2 轮增加约 5.16%；这些短跑数字未包含真实浏览器持续连接的负载，正式长跑仍保持关闭。

训练运行期两次 HTTP GET 均返回 200；HTTP 200 只验证服务可达，浏览器中的画面仍需用户在 Mac 上实际打开确认。

需要完整 Kit 界面时，在同一客户端命令增加 `--viz kit --livestream 2`，通过 [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html) 连接机器地址。客户端需能访问 TCP 49100 和 UDP 47998，VSCode 的普通 TCP 转发无法传送 UDP 视频。

多网卡机器可用 `--livestream 1` 显式公布客户端可达地址。在客户端命令前设置 `PUBLIC_IP=<可达地址>`；该值可以是局域网或 VPN 地址。确认相应网络和防火墙允许上述端口。

VSCode compound 中需要查看器时，在 `Arena client (headless)` 的 `args` 增加 `"--viz", "viser"`，或 `"--viz", "kit", "--livestream", "2"`。使用显式公布地址模式则设 livestream=1，并在该配置的 `env` 增加 `PUBLIC_IP`。shell 入口的变量不会自动传进 launch 配置。

查看器画面、双端断点和串流连接应分别验收；能生成视频不能证明实时串流或键盘交互可用。

## 排查入口

| 现象 | 检查 |
| --- | --- |
| 等待服务器超时 | 5555 是否空闲、server 是否成功加载 checkpoint、server 是否停在断点、两端日志 |
| BF16 Beta/Dirichlet 采样或 smoke 显存不足 | 使用已固定的 GR00T submodule 与 smoke 配方，确认 top LLM layers=0、model dtype=BF16、只有一张可见卡 |
| RL 保存阶段主机内存不足 | 检查主机可用内存与并行任务；入口的 `MALLOC_ARENA_MAX=2` 减少分配器保留，不更改 Ray 内存保护阈值 |
| GPU 回归前置产物缺失 | 按 [测试说明](testing.md) 指定实际 config、weights、export 和 smoke checkpoint；测试不会跳过缺失产物 |
| 移动目录后解释器失效 | 在新路径重建 `.venvs/`，核对 VSCode 解释器选择 |

普通回归通过 `scripts/test.sh all` 运行；真实模型与仿真回归见 [testing.md](testing.md)。
