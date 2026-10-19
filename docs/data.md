# 数据准备

在项目根目录执行，先完成 [环境准备](setup.md)。`data/models/logs/cache` 应为指向大容量存储的软链接，`.venvs/` 留在本地磁盘。数据生成不需要 VLA 模型；运行前停止占用 GPU 的推理服务器、训练或其他仿真实例。

## 输入和产物

| 入口 | 输入 | 输出 |
| --- | --- | --- |
| `scripts/data/download-demos.sh` | 固定 revision 的官方数据集 | `data/locomanipulation/official/` 中的已标注 HDF5、数据卡及 `lerobot/meta/` |
| `scripts/data/generate-demos.sh` | 官方已标注 HDF5 | `data/locomanipulation/generated.hdf5` |
| `scripts/data/convert-demos.sh [HDF5]` | 指定 HDF5，默认 `generated.hdf5` | 去掉 `.hdf5` 后缀的同名目录下的 `lerobot/`，包含训练归一化所需 `meta/stats.json` |
| `scripts/data/compare.sh` | 生成 HDF5、转换数据及官方元数据 | 默认 `logs/dataset-comparison/` 中的验证报告与分布比较 |
| `scripts/data/build-dataset.sh` | 已下载的官方数据 | 依次生成 100 条成功轨迹、转换并验证 |
| `scripts/data/download-base-model.sh` | 固定 revision 的 GR00T N1.6 基座 | `models/GR00T-N1.6-3B/`，供 [SFT](sft.md) 使用 |

官方数据源为 `nvidia/Arena-G1-Loco-Manipulation-Task`，revision `07de42fdade76102fc09d921c7b6200d0b31ec01`。下载只包含约 214 MB 的已标注 demo 与官方 LeRobot 统计元数据，不下载约 23 GB 的官方生成 HDF5。固定 revision 没有发布 `recorded.hdf5`，直接使用官方子任务标注。源 HDF5 实际包含 1 条、855 帧；官方生成数据元数据包含 100 条、94,936 帧，以文件而非数据卡中的条数为准。

基座来源为 `nvidia/GR00T-N1.6-3B`，revision `d0814e7ecb19202e7c8468b46098b0b7ef3a6d61`，只下载 JSON 和 safetensors，约 6.6 GB。它与闭环评测使用的 G1 checkpoint 不同，后者见 [评测](eval.md)。

## 下载与完整流水线

先下载官方 demo，再构建数据集；前一步成功后才继续：

```bash
scripts/data/download-demos.sh && scripts/data/build-dataset.sh
```

`build-dataset.sh` 要求 `generated.hdf5` 和 `generated/lerobot` 均不存在，任一步失败立即停止。生成器不支持断点续跑且会覆盖 HDF5，重新运行前先保留旧产物及日志。成功时日志末尾显示 `dataset build completed`，命令退出码为 0。

生成使用上游任务配置：CPU PhysX、`g1_wbc_pink`、1 个环境、seed 1，箱子平面位置随机范围 ±2.5 cm。`generation_guarantee=True`，保存满 100 条成功轨迹后结束；失败尝试不计入条数。首次资产加载和失败重试都会增加耗时。`generate-demos.sh` 末尾参数原样追加给上游生成器，可以覆盖默认选项。

需要训练基座时单独下载：

```bash
scripts/data/download-base-model.sh
```

## 先跑一条

先下载官方 demo。确认下列 smoke HDF5 和转换目录尚不存在，再按顺序生成、转换和验证；任一步失败就停止：

```bash
scripts/data/generate-demos.sh --generation_num_trials 1 \
  --output_file "$PWD/data/locomanipulation/smoke.hdf5" &&
scripts/data/convert-demos.sh data/locomanipulation/smoke.hdf5 &&
scripts/data/compare.sh \
  --generated data/locomanipulation/smoke/lerobot \
  --hdf5 data/locomanipulation/smoke.hdf5 --expected-episodes 1 \
  --output logs/dataset-comparison-smoke
```

确认命令退出码为 0，并检查验证报告后，再运行完整流水线。需要观察生成过程时，可在生成命令末尾加 `--viz viser`，查看器准备见 [调试](debugging.md)。

## 转换、验证和日志

转换按官方 `g1_locomanip_config.yaml` 映射字段，50 fps、chunk size 1000，并按上游规则移除每条轨迹最后一帧。入口修正上游转换器的 chunk 计数和固定训练 split，再用 GR00T 统计函数写入 `meta/stats.json`。生成的转换配置保存在 `logs/conversion/<HDF5 文件名>.yaml`。上游转换器遇到已有输出会询问是否删除，运行前应保留旧结果并选择空输出。

查看验证参数不会启动仿真或扫描数据：

```bash
scripts/data/compare.sh --help
```

已有完整数据可单独运行 `scripts/data/compare.sh`；用 `--generated`、`--hdf5`、`--official`、`--expected-episodes` 和 `--output` 选择其他数据与报告目录。

验证包含 HDF5 成功标记、转换前后条数和长度、43 维状态与动作的有限值、关节顺序、全局索引、时间戳、归一化统计，以及全部视频的帧数、640×480 分辨率和 50 fps。报告目录包含 `report.md`、`comparison.png`、`summary.json`、`state-action-stats.json` 和 `joint-distributions.csv`。官方对照使用发布的统计元数据；分布比较描述逐维边缘分布，不能证明策略成功率或时序相似度。

下载、生成、转换入口将输出追加到 `logs/<入口文件名>.log`；完整流水线另存 `logs/build-dataset.log`。`compare.sh` 自身不保存终端日志，可直接在终端查看；由流水线调用时输出进入流水线日志。
