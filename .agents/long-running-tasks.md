# Agent 长程任务操作说明

本说明仅约束 AI Agent。数据生成、下载、训练、评测、模型导出和 GPU 回归等持续任务，必须由 `systemd --user` 临时服务独立托管，保证工具调用、终端或 SSH 断开后继续运行。工具 exec session、终端后台、日志重定向或单独使用 `nohup` 均不能替代托管。禁止设置服务开机自动启动。

## 业务操作与托管

从业务文档取得命令、环境变量、工作目录、资源需求、产物和验收条件，执行时再添加托管包装。人类文档直接展示业务命令，业务脚本和代码只承担自身职责。记录可复用操作时，不要把本说明中的启动、状态查询和服务清理模板复制到业务文档或脚本中。

历史实验、交接、评审记录和一次性验证脚本保留原貌；实际承担服务管理的运维代码可以保留必要实现。

## 启动前

每次启动前检查用户管理器：

```bash
loginctl show-user "$USER" -p Linger
```

必须返回 `Linger=yes`。若未启用，执行后重新检查，确认成功再启动任务：

```bash
loginctl enable-linger "$USER"
loginctl show-user "$USER" -p Linger
```

确认业务前置产物、空闲 GPU 和端口；需要顺序执行的 GPU 任务，在前一项结束并验收后再启动下一项。大文件和运行日志经 `data/models/logs/cache` 软链接写入大容量存储，Python 环境留在本地 `.venvs/`。

## 启动临时服务

从项目根目录执行。以下以 SFT 为例；按业务文档替换命令和参数，使用绝对入口路径，并为每次任务创建独立日志目录：

```bash
RUN_ID="sft-$(date +%Y%m%d-%H%M%S-%N)"
UNIT="lab-$RUN_ID"
RUN_DIR="$PWD/logs/agent-tasks/$RUN_ID"
mkdir -p "$RUN_DIR"
command=("$PWD/scripts/train/sft.sh" smoke -- --output-dir "$PWD/models/$RUN_ID")

systemd-run --user --unit="$UNIT" \
  --property="WorkingDirectory=$PWD" \
  --property=Type=exec --property=RemainAfterExit=yes \
  --property=TimeoutStopSec=30 \
  --property="StandardOutput=append:$RUN_DIR/run.log" \
  --property=StandardError=inherit \
  --setenv=CUDA_VISIBLE_DEVICES=0 \
  "${command[@]}"
```

使用临时 service，不使用 scope，不执行 `systemctl enable`。`Type=exec` 确认入口成功执行，但提交成功不代表业务完成。默认按整个 cgroup 停止子进程，保持服务器、仿真和监控进程随所属任务一起清理；停止超时可按任务需要调整。

服务不会自动继承调用 shell 的环境变量。将业务所需变量逐项通过 `--setenv=NAME=value` 传入，例如 `TRAIN_DEBUG`、`RL_DEBUG`、`LAB_GPU_*`、`EVAL_RL_SERVER`、`PUBLIC_IP`；业务脚本仍负责加载 `scripts/env.sh`。输出、数据、checkpoint 等路径按业务入口要求传入，避免入口切换目录后相对路径指向错误位置。

需要 shell 运算或串行流水线时，以 `/usr/bin/bash -c '命令一 && 命令二'` 为入口；通过环境变量或位置参数传值，保持参数引用。若命令带有 `tee` 等管道，使用 `/usr/bin/bash -o pipefail -c '...'`，避免后续命令成功掩盖业务失败。

把本次 `UNIT`、日志目录、实际命令和业务产物位置记录在任务执行记录中。独立的常驻推理服务器和客户端应分别托管；客户端完成并验收后停止服务器，释放模型和端口。

## 观察与验收

```bash
systemctl --user show "$UNIT" \
  -p ActiveState -p SubState -p Result -p ExecMainStatus
tail -n 80 "$RUN_DIR/run.log"
journalctl --user -u "$UNIT" -n 80 --no-pager
```

使用文件日志时，业务 stdout/stderr 位于 `run.log`，journal 可用于检查服务管理器消息。需要跟随文件日志时使用 `tail -f`，退出日志查看不会停止任务。

`RemainAfterExit=yes` 让成功结束的服务保留 `ActiveState=active`、`SubState=exited`，因此不能只看 active 判断仍在计算。正常完成应同时满足 `SubState=exited`、`Result=success` 及 `ExecMainStatus=0`；失败时检查结果、退出码与日志。提交成功、日志出现模型加载信息或服务消失，都不能作为完成证据。

同时按业务文档验证产物，例如数据报告、SFT 的 checkpoint 与 `DONE`、RL 的完整权重和恢复状态、导出 manifest、评测的 `complete.json` 或测试报告。退出成功和业务验收都通过后，才报告任务完成。

## 保存结果与清理

先保存原始完成状态，再清理服务：

```bash
systemctl --user show "$UNIT" \
  -p ActiveState -p SubState -p Result -p ExecMainStatus \
  > "$RUN_DIR/service-result.txt"
systemctl --user stop "$UNIT"
```

主动 stop 可能改变结果字段，临时服务也可能在停止后被回收；stop 后的状态或 not-found 不能替代原始完成证据。失败服务保存证据后，可用 `systemctl --user reset-failed "$UNIT"` 清理失败状态。

需要提前中止时，同样先记录当前状态，再 stop，并明确记录为中止。调试器断开、停止跟随日志或关闭工具会话都不会停止服务；需要结束任务时必须显式停止对应服务。
