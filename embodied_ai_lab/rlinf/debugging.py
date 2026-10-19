"""单卡 RL 调试：只在选中的进程等待 VS Code attach。"""

import os


DEBUG_PORTS = {"driver": 5679, "actor": 5680, "rollout": 5681, "env": 5682}


def wait_for_debugger(role, rank=0):
    target = os.environ.get("RL_DEBUG")
    if target is not None and target not in DEBUG_PORTS:
        raise ValueError("RL_DEBUG must be driver, actor, rollout or env")
    if target != role or rank != 0:
        return

    import debugpy

    debugpy.configure(subProcess=False)
    port = DEBUG_PORTS[role]
    debugpy.listen(("127.0.0.1", port))
    print(f"[RL debug] {role} pid={os.getpid()} waiting on 127.0.0.1:{port}",
          flush=True)
    debugpy.wait_for_client()


def wait_for_model_debugger():
    """模型工厂在 worker 初始化后运行，因此可读取实际组名和 rank。"""
    target = os.environ.get("RL_DEBUG")
    if target not in ("actor", "rollout"):
        return

    from rlinf.scheduler import Worker

    worker = Worker.current_worker
    if worker.worker_address.root_group_name == worker.cfg[target].group_name:
        wait_for_debugger(target, rank=worker._rank)
