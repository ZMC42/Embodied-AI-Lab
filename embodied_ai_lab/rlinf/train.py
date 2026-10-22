"""先安装本地 checkpoint 扩展，再进入上游训练入口。"""

import os
import runpy
from pathlib import Path

import hydra

from embodied_ai_lab.rlinf.configuration import prepare_training_config
from embodied_ai_lab.rlinf.debugging import wait_for_debugger
from embodied_ai_lab.rlinf.extension import (
    install_cpu_state_dict_exports,
    install_serial_worker_initialization,
    register,
)

wait_for_debugger("driver")
register()
install_cpu_state_dict_exports()
install_serial_worker_initialization()
upstream = runpy.run_path(str(Path(os.environ["EMBODIED_PATH"]) / "train_embodied_agent.py"))


@hydra.main(version_base="1.1", config_path=str(Path(os.environ["EMBODIED_PATH"]) / "config"))
def main(cfg):
    prepare_training_config(cfg)
    upstream["main"](cfg)


if __name__ == "__main__":
    main()
