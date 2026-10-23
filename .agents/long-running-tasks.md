# Agent 长程任务操作说明

本说明仅约束 AI Agent。数据生成、下载、训练、评测、模型导出、GPU 回归和模型审核等持续任务，默认使用独立、具名的 tmux session 托管，保证工具调用、终端或 SSH 断开后继续运行，并可随时 attach 接管。工具 exec session、终端后台、日志重定向或单独使用 `nohup` 均不能替代托管。需要自动重启、资源限制或按 cgroup 严格清理全部子进程时，使用 `systemd --user` 临时服务；禁止设置开机自动启动。

## 业务操作与托管

从业务文档取得命令、环境变量、工作目录、资源需求、产物和验收条件，执行时再添加托管包装。人类文档直接展示业务命令，业务脚本和代码只承担自身职责。记录可复用操作时，不要把本说明中的启动、状态查询和清理模板复制到业务文档或脚本中。

历史实验、交接、评审记录和一次性验证脚本保留原貌；实际承担服务管理的运维代码可以保留必要实现。

## 启动前

确认业务前置产物、空闲 GPU 和端口；需要顺序执行的 GPU 任务，在前一项结束并验收后再启动下一项。大文件和运行日志经 `data/models/logs/cache` 软链接写入大容量存储，Python 环境留在本地 `.venvs/`。

每次任务记录 session 或 service 名、工作目录、实际命令、环境变量、日志和产物位置。独立的常驻推理服务器和客户端分别托管；客户端完成并验收后停止服务器，释放模型和端口。GPU 监控等辅助进程也必须记录其所属任务和停止方式。

## 默认：独立 tmux session

从项目根目录执行。以下以下载基础模型为例；按业务文档替换命令和参数，使用绝对入口路径，并为每次任务创建独立日志目录：

```bash
RUN_ID="download-$(date +%Y%m%d-%H%M%S-%N)"
SESSION="lab-$RUN_ID"
RUN_DIR="$PWD/logs/agent-tasks/$RUN_ID"
mkdir -p "$RUN_DIR"
export RUN_DIR

cat > "$RUN_DIR/run.sh" <<'SH'
#!/usr/bin/bash
set -o pipefail
trap 'printf "143\n" > "$RUN_DIR/exit-code"; exit 143' TERM
trap 'printf "130\n" > "$RUN_DIR/exit-code"; exit 130' INT
trap 'printf "129\n" > "$RUN_DIR/exit-code"; exit 129' HUP
ps -o pgid= -p "$$" | tr -d ' ' > "$RUN_DIR/process-group"
"$PWD/scripts/data/download-base-model.sh" 2>&1 | tee "$RUN_DIR/run.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$RUN_DIR/exit-code"
exit "$status"
SH

tmux new-session -d -s "$SESSION" -c "$PWD"
tmux set-option -t "$SESSION" remain-on-exit on
tmux respawn-pane -k -t "$SESSION" \
  "RUN_DIR='$RUN_DIR' /usr/bin/bash '$RUN_DIR/run.sh'"
```

`run.sh` 是本次执行记录，放在日志目录，不进入业务脚本。按任务显式传入所需环境变量，例如 `SFT_DEBUG`、`RL_DEBUG`、`LAB_GPU_*`、`EVAL_RL_SERVER`、`PUBLIC_IP`；业务脚本仍负责加载 `scripts/env.sh`。tmux server 可能早于当前 shell 启动，不要依赖它继承当前 shell 新增的环境变量。输出、数据、checkpoint 等路径按业务入口要求传入，避免入口切换目录后相对路径指向错误位置。

模板保留业务命令的退出码；若替换为多段业务管道，在执行后立即保存完整的 `PIPESTATUS` 并判断各段状态，不能让 `tee` 或后续命令成功掩盖业务失败。需要 shell 串行流水线时使用 `命令一 && 命令二`。`remain-on-exit` 保留已结束的 pane，便于检查退出状态。

```bash
tmux attach -t "$SESSION"
# Ctrl-b d：只断开观察，任务继续运行。
```

需要实时观察模型审核时，使用 CLI 的流式输出，同时保留最终结果。JSON 最终输出模式可能在整个响应结束前没有输出，日志暂时不增长不能证明进程已停。

## 观察与验收

```bash
tmux list-panes -t "$SESSION" \
  -F '#{pane_id} pid=#{pane_pid} dead=#{pane_dead} status=#{pane_dead_status}'
tail -n 80 "$RUN_DIR/run.log"
cat "$RUN_DIR/exit-code"
```

任务仍运行时 `exit-code` 尚不存在。恢复观察时检查 tmux pane 和实际业务进程；session 存在或消失都不能证明任务完成。正常结束必须有 `exit-code` 为 0，并按业务文档验证产物，例如数据报告、SFT 的 checkpoint 与 `DONE`、RL 的完整权重和恢复状态、导出 manifest、评测的 `complete.json` 或测试报告。退出成功和业务验收都通过后，才报告任务完成。

## 停止与清理 tmux 任务

自然结束后保留日志、`exit-code` 和验收记录，再清理 session。若有推理服务器、GPU 监控等辅助进程，先按其记录的停止方式清理，并检查进程和 GPU/端口占用。

需要提前中止时，先记录 pane 状态和当前业务进程，将本次执行标记为中止，再停止其进程组：

```bash
tmux list-panes -t "$SESSION" \
  -F '#{pane_id} pid=#{pane_pid} dead=#{pane_dead} status=#{pane_dead_status}' \
  | tee "$RUN_DIR/state-before-stop.txt"
TASK_PGID=$(cat "$RUN_DIR/process-group")
ps -eo pid,ppid,pgid,args | awk -v group="$TASK_PGID" '$3 == group' \
  >> "$RUN_DIR/state-before-stop.txt"
printf 'interrupted\n' > "$RUN_DIR/outcome"
kill -TERM -- "-$TASK_PGID"
```

留出业务保存和清理时间，再用相同的进程组查询确认没有残留。若仍有进程，检查原因后对该组执行 `kill -KILL -- "-$TASK_PGID"`，并记录强制中止；此时可能没有最终退出码，不能作为正常完成。再核对独立 session 中的服务器、监控进程及 GPU/端口，最后执行 `tmux kill-session -t "$SESSION"`。关闭 pane/session、调试器断开或停止跟随日志，都不能代替子进程清理与验收。

进程组清理只覆盖仍属于该组的进程。业务会自行 `setsid`、守护化或需要严格保证全部后代进程随任务停止时，启动前选择下面的 systemd 路径。本项目 torchrun（包括 SFT）的 worker 会以新 session 启动，RLinf/Ray 的后代也可能脱离原进程组，这两类任务使用 systemd cgroup 托管；不能仅凭原 PGID 无残留就判断 worker 已清理。停止 GPU 任务后，另用 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 核查，并与本次任务的进程记录对应。

## 按需：systemd 临时服务

启动前检查用户管理器：

```bash
loginctl show-user "$USER" -p Linger
```

必须返回 `Linger=yes`。若未启用，执行 `loginctl enable-linger "$USER"` 并重新检查，确认成功再启动。

按上述方式为本次任务创建独立 `RUN_ID`、`RUN_DIR`，以下以 SFT 为例替换业务命令：

```bash
UNIT="lab-$RUN_ID"
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

使用临时 service，不使用 scope，不执行 `systemctl enable`。`Type=exec` 确认入口成功执行，但提交成功不代表业务完成。默认按整个 cgroup 停止子进程；停止超时可按任务需要调整。需要资源限制或自动重启时按任务设置 service 属性；自动重启不能掩盖之前的失败，日志与验收记录须保留各次尝试的结果。

服务不会自动继承调用 shell 的环境变量，将业务所需变量逐项通过 `--setenv=NAME=value` 传入。需要 shell 运算或串行流水线时，以 `/usr/bin/bash -c '命令一 && 命令二'` 为入口；通过环境变量或位置参数传值，保持参数引用。若命令带有 `tee` 等管道，使用 `/usr/bin/bash -o pipefail -c '...'`。

```bash
systemctl --user show "$UNIT" \
  -p ActiveState -p SubState -p Result -p ExecMainStatus
tail -n 80 "$RUN_DIR/run.log"
journalctl --user -u "$UNIT" -n 80 --no-pager
```

`RemainAfterExit=yes` 让成功结束的服务保留 `ActiveState=active`、`SubState=exited`，因此不能只看 active 判断仍在计算。正常完成应同时满足 `SubState=exited`、`Result=success` 及 `ExecMainStatus=0`，并通过相同的业务产物验收。使用文件日志时 stdout/stderr 位于 `run.log`，journal 可用于检查服务管理器消息。

先保存原始完成状态，再清理服务：

```bash
systemctl --user show "$UNIT" \
  -p ActiveState -p SubState -p Result -p ExecMainStatus \
  > "$RUN_DIR/service-result.txt"
systemctl --user stop "$UNIT"
```

主动 stop 可能改变结果字段，临时服务也可能在停止后被回收；stop 后的状态或 not-found 不能替代原始完成证据。失败服务保存证据后，可用 `systemctl --user reset-failed "$UNIT"` 清理失败状态。提前中止也须先记录状态，再 stop，并明确记录为中止。
