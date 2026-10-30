# 测试入口

从项目根目录执行：

```bash
scripts/test.sh all
scripts/test.sh rl
```

`arena`、`groot`、`rl` 分别使用 `scripts/env.sh` 导出的对应解释器；`all` 按这个顺序运行三套环境。默认执行各目录的普通 `test_*.py`，排除 `test_gpu_*.py`。默认测试仍包含 CUDA/FSDP 检查，因此并非纯 CPU 测试。`--gpu` 只执行 `test_gpu_*.py`，不重复普通测试。GPU 测试通过 unittest discovery 收集，真实模型、仿真及不同 GR00T 版本各在独立 subprocess 中运行，失败退出码会传回 unittest。

GPU 测试的前置产物和运行示例见下文。

## GPU 测试所需产物

RL 入口默认设置 `MALLOC_ARENA_MAX=2`，减少 glibc 在多线程模型与仿真运行后保留的主机内存，为 CPU checkpoint 保存留出空间；可通过同名环境变量覆盖。它不改变模型、训练配置或 Ray 内存保护阈值。

先完成数据准备、SFT 与 RL 短跑，并导出 RL checkpoint。产物缺失会直接失败，不会 skip。所有相对路径以项目根目录为基准。

| 测试 | 前置产物与运行资源 |
| --- | --- |
| `rl/test_gpu_inference.py` | `models/GR00T-N1.6-G1/checkpoint-20000`、`data/locomanipulation/generated/lerobot` 的 episode 0/49/99 第 100 帧；CUDA。分别运行 reference 与 RL fork，保留 `atol=rtol=1e-5` 的比较报告和源码选择。fork 已知存在 letterbox 预处理差异，原检查允许比较结果为不等价并选择 reference。 |
| `rl/test_gpu_actor.py` | 同上；自动在 reference 子进程生成真实 observation，再验证 actor、value、ratio 与各组梯度，无需依赖其他测试的执行顺序。 |
| `rl/test_gpu_arena.py` | RL 环境、Isaac Sim、CUDA 相机与 `data/assets` 中的 Arena 资产；检查 typed cfg + ArenaEnvBuilder，默认 2 个真实环境、CPU 物理。独立 worker 支持 `--num-envs`（默认 2）与 `--viser-envs`（默认 0）；Viser 开启时在 reset 和 step 后要求 8080 返回 HTTP 200。无需 checkpoint。 |
| `rl/test_gpu_checkpoint.py` | 解析后的 RL 配置、完整训练权重及配置引用的基座模型；Isaac Sim、Arena 资产与 CUDA，重载后闭环执行两个动作块。 |
| `rl/test_gpu_precision.py` | 同一份解析后 RL 配置、完整训练权重及基座模型；足够的主机内存加载模型，逐张量验证 dtype 和数值。 |
| `rl/test_gpu_export.py` | 同一 RL 配置和完整权重、导出目录（含 `rl_export.json`）、G1 基座模型与上述数据；CUDA、空闲本地端口 5556。导出 manifest 必须对应被检查的权重路径；实测 RLinf 与 server 的单条及批量请求，保留 `atol=rtol=1e-5` 和初始噪声逐位相等判据。 |
| `groot/test_gpu_smoke_checkpoint.py` | SFT smoke checkpoint（含优化器、scheduler、RNG、processor 等训练文件）、`models/GR00T-N1.6-3B` 与上述数据；CUDA。验证精确 global step、训练状态、冻结/更新权重及真实推理。 |

GPU checkpoint 测试通过以下环境变量指定本次产物。`LAB_GPU_RL_CONFIG` 须包含 `env.eval.task`、`env.eval.external_environment_class_path`、`env.eval.arena`；旧配置需按原运行的任务和工厂补齐选择字段，并将旧 `arena_args` 中的场景参数迁到 `arena`。四个路径变量没有默认实验目录，运行对应测试时必须设置；未设置或产物缺失都会失败。

| 变量 | 含义 |
| --- | --- |
| `LAB_GPU_RL_CONFIG` | RL 运行目录下的 `tensorboard/config.yaml` |
| `LAB_GPU_RL_WEIGHTS` | 同次运行的 `train/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt` |
| `LAB_GPU_EXPORT` | 由这份完整权重生成的 GR00T 导出目录 |
| `LAB_GPU_SMOKE` | SFT 短跑保存的 `checkpoint-20` 目录 |
| `LAB_GPU_SMOKE_STEPS` | `20`，精确匹配预期保存步数；检查其他 checkpoint 时同时设置路径与步数 |

导出目录通过 `scripts/eval/export-rl.sh` 生成，运行示例见下文；输出必须是尚不存在的新目录，已有导出应先保留。

测试报告位于 `logs/gpu-tests/`。每次运行会重新生成并覆盖自己的 observation 和报告；需保留多次结果时，运行前将上一次报告复制到其他日志目录。上述大型文件及日志应通过项目的 `data/models/logs/cache` 软链接落到外部存储。

## 依次运行导出与 GPU 测试

将下列两个运行目录替换成已完成的 [RL](rl.md#两迭代短跑) 和 [SFT](sft.md#冒烟与正式训练) 短跑目录；在同一个 shell 中依次执行后面的示例：

```bash
RL_RUN="$PWD/logs/rl-training/your-run"
SFT_RUN="$PWD/models/your-sft-run"
EXPORT_DIR="$PWD/models/rl-export-$(date +%Y%m%d-%H%M%S)"
```

先用这份 RL 权重生成导出产物，输出目录必须尚不存在：

```bash
scripts/eval/export-rl.sh \
  --weights "$RL_RUN/train/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt" \
  --output "$EXPORT_DIR"
```

等待导出命令退出，确认退出码为 0 且生成 `rl_export.json`，再运行 RL GPU 测试：

```bash
LAB_GPU_RL_CONFIG="$RL_RUN/tensorboard/config.yaml" \
LAB_GPU_RL_WEIGHTS="$RL_RUN/train/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt" \
LAB_GPU_EXPORT="$EXPORT_DIR" \
  scripts/test.sh rl --gpu
```

等待 RL 测试命令退出，确认退出码为 0、unittest 显示 `OK`，并检查 `logs/gpu-tests/` 中的报告，再运行 GR00T GPU 测试：

```bash
LAB_GPU_SMOKE="$SFT_RUN/checkpoint-20" LAB_GPU_SMOKE_STEPS=20 \
  scripts/test.sh groot --gpu
```

同样确认 GR00T 测试退出码为 0、unittest 显示 `OK`，并检查报告。两套 GPU 测试顺序运行，避免与训练同时占用 GPU。

检查其他 checkpoint 时相应修改 `LAB_GPU_*` 环境变量，SFT 同时指定精确的预期步数 `LAB_GPU_SMOKE_STEPS`。单项测试保留该项所需的产物变量，然后执行 `source scripts/env.sh` 和 `"$RL_PY" -m unittest tests.rl.test_gpu_actor`；GR00T 项使用 `$GROOT_PY`。变量需通过 `export` 或命令前赋值传给测试进程。

Arena worker 可独立指定环境数和查看器，直接执行以下命令，并为每组指定不同的输出文件：

```bash
(
  source scripts/env.sh
  mkdir -p "$LAB_ROOT/logs/gpu-tests"
  export LD_LIBRARY_PATH="$("$RL_PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  cd "$ARENA_ROOT"
  "$RL_PY" -m tests.rl.test_gpu_arena \
    --num-envs 4 --output "$LAB_ROOT/logs/gpu-tests/arena-4.json"
)
```

并行环境与 Viser 的 GPU 回归顺序运行 `--num-envs 2`、`--num-envs 4`、`--num-envs 2 --viser-envs 1` 三组。`--device cuda:0` 可切换为 GPU PhysX。独立 worker 与默认 unittest 入口均检查只在第 0 个环境注入成功、跌倒、超时和掉箱信号，其余环境不应误触发。

更换导出目录时，同时用 `LAB_GPU_EXPORT` 指定新的目录；导出来源权重必须与 `LAB_GPU_RL_WEIGHTS` 一致。完整导出与评测说明见 [eval.md](eval.md)。
