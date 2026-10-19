# 环境准备

所有命令在仓库根目录执行。需要 Linux、NVIDIA GPU 与驱动、CUDA toolkit、Git（含 LFS）、[uv](https://docs.astral.sh/uv/getting-started/installation/) 和可访问模型及仿真资产的网络。版本矩阵见 [README](../README.md)。Isaac Sim 运行环境需满足其上游系统要求；脚本会设置接受相关 EULA 的环境变量，使用前应阅读对应许可。

## 存储与安装

`data/models/logs/cache` 存放大文件，链接到大容量存储；三套 venv 留在本地 `.venvs/`。在首次安装的新 checkout 中运行：

```bash
STORAGE=/path/to/storage
mkdir -p "$STORAGE"/{data,models,logs,cache}
for name in data models logs cache; do ln -s "$STORAGE/$name" "$name"; done
scripts/setup/bootstrap.sh
```

bootstrap 以 `GIT_LFS_SKIP_SMUDGE=1` 初始化递归 submodule，检查四个存储入口，然后依次创建 Arena、GR00T 和 RL 环境。缺少存储入口时直接失败，提示对应 `ln -s` 命令。fork 已包含项目修改，不再手工应用 patch；LFS 大文件由各上游运行流程按需获取。

默认 `CUDA_HOME=/usr/local/cuda`；若 toolkit 安装在其他位置，在安装命令前设置 `CUDA_HOME`。uv 缓存使用本机默认位置，不要把 `UV_CACHE_DIR` 指向网络存储。脚本直接调用 venv 的解释器，避免运行阶段 `uv run` 自动同步依赖。

已有子模块和存储链接时，可单独安装某套环境：

```bash
scripts/setup/venv-arena.sh
scripts/setup/venv-groot.sh
scripts/setup/venv-rl.sh
```

| 环境 | 安装来源与附加依赖 |
| --- | --- |
| arena | Arena 的 frozen lockfile、dev extra、gr00t-client group；先装 `cmake<4` 与 `egl-probe==1.0.2`，再装 debugpy、Viser 及已验证版本的附加依赖 |
| groot | Arena 内嵌 GR00T 的 `uv sync --python 3.10`，再装 debugpy；编译扩展使用 CUDA toolkit |
| rl | Arena 的 `uv sync --frozen --inexact`，用 `--no-deps` 安装 `scripts/setup/rl-requirements.txt`（包含 `viser==1.1.1` 及其依赖），再 editable 安装 RLinf 和 GR00T reference |

RL freeze 包含与 GR00T 包元数据不同的 torch、NumPy 等已验证版本，因此安装使用 `--no-deps`。不要在 RL 环境中额外同步 GR00T 的依赖声明。

## 核对布局

```bash
git submodule status --recursive
source scripts/env.sh
"$ARENA_PY" --version
"$GROOT_PY" --version
"$RL_PY" --version
scripts/test.sh all
```

`scripts/env.sh` 从自身位置推导 `LAB_ROOT`，导出三套解释器和第三方目录变量。它将 Hugging Face 缓存放到 `cache/huggingface`，临时运行目录放到 `tmp/runtime`，仿真资产缓存和 Recorder 日志链接到 `data/`。其他脚本共用这一布局，无需按主机修改路径。

`scripts/test.sh all` 会顺序运行三套环境的普通测试，其中包含 CUDA/FSDP 检查；GPU 模型与仿真验收另见 [测试说明](testing.md)。改动或移动工作目录后应重建 venv，因为 venv 中包含绝对路径。

## 持续任务托管

训练、数据生成、评测和 GPU 回归由 `systemd --user` 临时服务托管。先执行：

```bash
loginctl show-user "$USER" -p Linger
```

必须返回 `Linger=yes`；否则先执行 `loginctl enable-linger "$USER"`，然后重新确认。各阶段文档给出具体服务命令，不执行 `systemctl enable`。`--property=RemainAfterExit=yes` 让服务结束后仍可读取真实退出状态；成功时显示 `Result=success`、`ExecMainStatus=0`，即使 `ActiveState=active` 也不代表进程仍在计算，要同时查看 `SubState=exited`。

使用文档中的实际 `UNIT` 查看或停止任务：

```bash
systemctl --user show "$UNIT" -p ActiveState -p SubState -p Result -p ExecMainStatus
journalctl --user -u "$UNIT" -f
systemctl --user stop "$UNIT"
```

先记录退出状态再 stop；停止后 transient unit 可能被回收，不能把 not-found 状态当作成功证据。服务的标准输出进入 journal；需要长期保留时按流程示例写到 `logs/`。不要用终端后台或 `nohup` 替代托管。
