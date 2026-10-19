# 调试与可视化

先完成 [环境准备](setup.md)，在 VSCode（远程机器使用 Remote SSH）中打开仓库根目录，安装 Python 与 Python Debugger 扩展。`.vscode/settings.json` 默认使用 `.venvs/arena`；server、client 和等待任务分别显式选择 `.venvs/groot`、`.venvs/arena`、`.venvs/groot`。

## 双端断点

准备 [G1 checkpoint](eval.md#官方-g1-checkpoint)，先停止占用 5555 端口的手动或托管服务器，再启动 compound `GR00T server + Arena client`。`prepare runtime` 任务执行 `scripts/env.sh` 创建运行目录和资产链接；client 在 ZMQ ping 成功后才启动，等待超时为 600 秒。服务器和客户端的工作目录均保持上游源码目录。

`.vscode/runtime.env` 仅包含 EULA 与无缓冲输出设置，无凭据。launch 配置显式设置 `HF_HOME`、`TMPDIR`，客户端优先加载 Arena venv 内的 CUDA runtime，避免 WebRTC 扩展自带的 CUDA runtime 与 PyTorch 冲突。

建议断点：

- 客户端 `third_party/IsaacLab-Arena/isaaclab_arena_gr00t/policy/gr00t_remote_closedloop_policy.py` 的 `_client.get_action(...)`。
- 服务器 `third_party/IsaacLab-Arena/submodules/Isaac-GR00T/gr00t/policy/gr00t_policy.py` 的 `self.model.get_action(...)`。

两端均为 `justMyCode=false`。服务器停在断点时，客户端等待 ZMQ 返回是预期行为。继续后 action 应返回并推进仿真；compound 的 `stopAll=true` 让停止操作结束两侧调试进程。调试产物位于 `logs/debug/`，Kit 日志为 `logs/debug-kit.log`。该模式用于交互调试；持续评测使用 [评测文档](eval.md) 的托管服务。

## 训练断点调试

使用单张可见 GPU、smoke 配方和新输出目录。先确认 `Linger=yes`，将训练托管到临时 user service：

```bash
UNIT="lab-sft-debug-$(date +%s)"
systemd-run --user --unit="$UNIT" --property="WorkingDirectory=$PWD" \
  --property=Type=exec --property=RemainAfterExit=yes \
  --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=TRAIN_DEBUG=1 \
  "$PWD/scripts/train/sft.sh" smoke -- \
  --output-dir "$PWD/models/sft-debug-$(date +%Y%m%d-%H%M%S)" \
  --max-steps 2 --save-steps 2
```

在 VSCode 选择 `GR00T N1.6 training (attach)`，连接远程机器本地 `127.0.0.1:5678`。该端口只绑定 loopback；Remote SSH 中调试器运行在远程机器。smoke 的 loader workers=0，使样本处理留在训练进程。可在数据集的 `extract_step_data()`、`Gr00tN1d6.forward()`、`Gr00tN1d6ActionHead.forward()` 和 Transformers `Trainer.training_step()` 下断点。

断开 attach 后训练继续；停止训练使用 `systemctl --user stop "$UNIT"`。日志为 `logs/train-sft/smoke.log`，也可 `journalctl --user -u "$UNIT" -f`。不要同时启动两个等待 5678 的任务。

## 远程观察

Arena 安装脚本已安装 `viser==1.1.1`。使用 [单局观察](eval.md#单局观察) 的托管命令，将客户端尾参加上 `--viz viser`。在 VSCode 转发 TCP 8080，再打开 `http://localhost:8080`；有网络直连时可访问机器的 8080 端口。Viser 默认监听 `0.0.0.0:8080`，支持视角、暂停和重置。它同步 PhysX 状态，显示材质与 Kit 有差异；模型所需 RTX 相机仍然运行。

需要完整 Kit 界面时，在同一客户端命令增加 `--viz kit --livestream 2`，通过 [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html) 连接机器地址。客户端需能访问 TCP 49100 和 UDP 47998，VSCode 的普通 TCP 转发无法传送 UDP 视频。

多网卡机器可用 `--livestream 1` 显式公布客户端可达地址。在 `systemd-run` 中增加 `--setenv=PUBLIC_IP=<可达地址>`；该值可以是局域网或 VPN 地址。确认相应网络和防火墙允许上述端口。

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

普通回归通过 `scripts/test.sh all` 运行；真实模型与仿真回归按 [testing.md](testing.md) 的 systemd 方式运行。
