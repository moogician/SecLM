from abc import ABC, abstractmethod
from typing import List


class Benchmark(ABC):
    dataset = None
    EVAL_SETTINGS = {"all": lambda _: True}
    EVAL_SUBSETS = {"all": lambda _: True}

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
        self.parallel = cfg.max_concurrent_runs
        self.tag_template = f"{{}}@{self.temperature}"

        self.eval_setting = self.EVAL_SETTINGS.get(cfg.eval_setting, lambda _: True)
        self.eval_subset = self.EVAL_SUBSETS.get(cfg.eval_subset, lambda _: True)

    @abstractmethod
    def generate(self, models):
        """Generate code using LLM inference."""
        raise NotImplementedError

    @abstractmethod
    def test(self, models):
        """Run Docker-based tests on generated code."""
        raise NotImplementedError

    @abstractmethod
    def summary(self):
        """Compute and display metrics from test results."""
        raise NotImplementedError

    def metric(self):
        raise NotImplementedError
