# 核心原则
1. 树立“奥卡姆剃刀”原则（如无必要，勿增实体），明令禁止防御式编程，明令禁止把任务复杂化
2. 大文件（模型权重、训练 checkpoint、数据集、运行日志等）通过 `data/models/logs/cache` 软链接放到大容量存储；Python 虚拟环境留在本地 `.venvs/`。
3. commit message 中禁止添加“Co-Authored-By”或类似签名信息。
4. AI Agent 执行数据生成、训练、评测等需要持续运行的长程任务，必须使用 `systemd --user` 服务独立托管；启动前确认用户管理器启用 `Linger=yes`，禁止设置开机自动启动。工具 exec session、终端后台、日志重定向或单独使用 `nohup` 均不能视为已可靠托管。
5. 长程或复杂任务尽可能使用 subagent，避免 Context Explosion。
6. 若存在 `private/AGENTS.md`，一并遵守。
