from .base import Benchmark

from .cweval.task import CWEvalTask
from .livecodebench.task import LiveCodeBenchTask
from .baxbench.task import BaxBenchTask

name2benchmark = {
    "cweval": CWEvalTask,
    "livecodebench": LiveCodeBenchTask,
    "baxbench": BaxBenchTask,
}


def from_cfg(cfg) -> Benchmark:
    return name2benchmark[cfg.benchmark](cfg)
