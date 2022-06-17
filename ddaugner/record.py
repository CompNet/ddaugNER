import os, json
from typing import Any, Callable, List

import torch


class ExperimentRecorderContext:
    def __init__(self, recorder: "ExperimentRecorder", context: str) -> None:
        self.recorder = recorder
        self.context = context
        self.old_context = ""

    def __enter__(self):
        self.old_context = self.recorder._context
        self.recorder._context = f"{self.context}.{self.old_context}"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.recorder._context = self.old_context


class ExperimentRecorder:
    """A *simple* experiment recorder"""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

        self._context = ""

        self.values_dict = {}
        self.values_file_path = f"{root_dir}/values.json"
        self._flush_values()

    def _flush_values(self):
        with open(self.values_file_path, "w") as f:
            json.dump(self.values_dict, f, indent=4)

    def record_value_(self, name: str, value: Any):
        """
        :param value: jsonifiable value
        """
        self.values_dict[self._context + name] = value
        self._flush_values()

    def record_series_value_(self, series_name: str, value: Any):
        """Add a value to a series, creating it if it does not exist
        :param series_name:
        :param value: jsonifiable value
        """
        series_name = self._context + series_name
        if not series_name in self.values_dict:
            self.values_dict[series_name] = []
        self.values_dict[series_name].append(value)
        self._flush_values()

    def record_series_reduction_(
        self,
        series_name: str,
        reduction_name: str,
        reduce_fn: Callable[[List[Any]], Any],
    ):
        """"""
        series_name = self._context + series_name
        if not series_name in self.values_dict:
            self.values_dict[series_name] = []
        reduction = reduce_fn(self.values_dict[series_name])
        self.values_dict[f"{series_name}.{reduction_name}"] = reduction
        self._flush_values()

    def record_model_(self, name: str, model: torch.nn.Module):
        name = self._context + name
        torch.save(model, f"{self.root_dir}/{name}.pth")

    def context(self, context_name: str) -> ExperimentRecorderContext:
        return ExperimentRecorderContext(self, context_name)
