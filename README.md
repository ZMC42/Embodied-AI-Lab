# Embodied-AI-Lab

基于 IsaacLab-Arena、Isaac-GR00T N1.6 和 RLinf 的 Unitree G1 物料箱搬运实验，覆盖 Mimic 数据生成、LeRobot 转换、监督微调（SFT）、PPO 训练和闭环评测。

## 版本矩阵

第三方源码由 Git submodule 固定到包含项目修改的 fork commit，无需另行应用补丁。实际版本以 `git submodule status --recursive` 为准。

| 组件 | 固定基点 / 已验证环境 |
| --- | --- |
| IsaacLab-Arena | 基点 `c7edf3a9` |
| Isaac Lab / Isaac Sim | 基点 `91a0906f` / 6.1.0.0 |
| GR00T（SFT、reference） | N1.6，基点 `e29d8fc` |
| GR00T RL fork（推理对照） | 基点 `f7a25760` |
| RLinf | 基点 `7e90641c` |
| `.venvs/arena` | Python 3.12、PyTorch 2.11.0+cu128 |
| `.venvs/groot` | Python 3.10、PyTorch 2.7.1+cu126、FlashAttention 2.7.4.post1 |
| `.venvs/rl` | Python 3.12、PyTorch 2.11.0+cu128、SDPA |
| NVIDIA 驱动 | 已验证 595 系列 |

单环境官方 checkpoint 评测实测约占 11.7 GiB 显存；SFT smoke 和 PPO offload 在 24 GiB 单卡验证。正式 SFT 的显存需求随每卡 batch 和训练范围变化。

## 目录

```text
configs/{sft,rl}/         实验配方
embodied_ai_lab/          环境桥、数据、训练、评测逻辑
scripts/{setup,data,train,eval}/
                         面向用户的 shell 入口
scripts/{env,test}.sh     共享环境与测试入口
tests/{arena,groot,rl}/   unittest 回归检查
docs/                    分阶段操作说明
third_party/             固定版本的 fork submodule
.vscode/                 双端与训练断点调试
.venvs/{arena,groot,rl}/  本地 Python 环境（不入库）
data models logs cache   指向大容量存储的软链接（不入库）
```

## 快速开始

在 Linux、NVIDIA GPU 和 CUDA toolkit 已就绪的机器上，安装 Git LFS、uv，并从仓库根目录执行：

```bash
git clone https://github.com/ZMC42/Embodied-AI-Lab.git
cd Embodied-AI-Lab
# 设置为已有的大容量存储目录。
STORAGE=/path/to/storage
mkdir -p "$STORAGE"/{data,models,logs,cache}
for name in data models logs cache; do ln -s "$STORAGE/$name" "$name"; done
scripts/setup/bootstrap.sh
scripts/test.sh all
```

bootstrap 初始化递归子模块并安装三套环境；详见 [环境准备](docs/setup.md)。已有环境可先运行测试。默认测试含 CUDA/FSDP 检查。

按流程继续：[数据准备](docs/data.md) → [SFT](docs/sft.md) 或 [PPO](docs/rl.md) → [导出与评测](docs/eval.md)。

[调试与可视化](docs/debugging.md) 说明 VSCode 双端调试、训练 attach 和远程查看器；[测试说明](docs/testing.md) 列出 GPU 回归所需产物与运行方式。
