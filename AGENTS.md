# 核心原则
1. 树立“奥卡姆剃刀”原则（如无必要，勿增实体），明令禁止防御式编程，明令禁止把任务复杂化
2. 大文件（模型权重、训练 checkpoint、数据集、运行日志等）通过 `data/models/logs/cache` 软链接放到大容量存储；Python 虚拟环境留在本地 `.venvs/`。
3. commit message 中禁止添加“Co-Authored-By”或类似签名信息。
4. AI Agent 执行长程任务默认使用独立、具名的 tmux session 托管；需要自动重启、资源限制或 cgroup 清理时使用 `systemd --user`。启动前遵守 [长程任务操作说明](.agents/long-running-tasks.md)。此约束仅针对 Agent；写入业务文档、脚本和代码的内容只描述业务操作，托管细节统一维护在该说明中。
5. 长程或复杂任务尽可能使用 subagent，避免 Context Explosion。
6. 若存在 `private/AGENTS.md`，一并遵守。
