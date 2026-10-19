"""先安装本地 checkpoint 扩展，再进入上游训练入口。"""

import os
import runpy
from pathlib import Path

from embodied_ai_lab.rlinf.debugging import wait_for_debugger
from embodied_ai_lab.rlinf.extension import install_checkpoint_cpu_clone, register

wait_for_debugger("driver")
register()
install_checkpoint_cpu_clone()
runpy.run_path(str(Path(os.environ["EMBODIED_PATH"]) / "train_embodied_agent.py"),
               run_name="__main__")
