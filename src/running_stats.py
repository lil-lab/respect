

from typing import Any, Dict

import torch
from absl import logging

import wandb
from transformers.trainer_callback import TrainerCallback
from utils import XTrainingArguments, clear_all_cache, prefix_dict_keys_with
import globals as g


class RunningStatsCache:
    def __init__(self):
        super().__init__()
        self._cache = []

    def _check_shape(self, x):
        assert x.shape[1] == 1, "RunningStatsCache only supports 1D input"
        # x.shape == (bs, 1)

    def forward(self, x):
        self._check_shape(x)
        self._cache.append(x.clone())
        return None

    def mean_and_var_and_size(self):
        concatenated = torch.concat(self._cache)  # (bs, 1)
        return concatenated.mean(), concatenated.var(), concatenated.size(0)

    def clear_cache(self):
        for i in self._cache:
            i.detach()
        while len(self._cache) > 0:
            self._cache.pop()
        clear_all_cache()

    def is_empty_cache(self):
        return self._cache == []

    def __repr__(self):
        return super().__repr__() + f"({self._cache})"


def track_scalar(running_stats: Dict, scalar: torch.Tensor, cache_key: str):
    if cache_key not in running_stats:
        running_stats[cache_key] = RunningStatsCache()
    clone = scalar.detach().clone()
    if clone.dim() == 0:
        clone = clone.unsqueeze(0)
    if clone.dim() == 1:
        clone = clone.unsqueeze(1)
    running_stats[cache_key].forward(clone)


class BatchLossCallback(TrainerCallback):
    """Custom callback to retrieve, backprop, and log batch-level loss
    """

    def _collect_and_clear_running_stats(self, running_stats: Dict[str, RunningStatsCache], exclude_prefix=None):
        logs = dict()
        for k in running_stats:
            if exclude_prefix and k.startswith(exclude_prefix):
                continue
            if running_stats[k].is_empty_cache():
                continue
            mean, var, size = running_stats[k].mean_and_var_and_size()
            logs[k] = mean.detach().item()
            running_stats[k].clear_cache()
        return logs

    def on_optimizer_step_begin(self, args: XTrainingArguments, state, control, **kwargs):
        """Backprop batch-level loss"""
        pass

    def on_step_end(self, args: XTrainingArguments, state, control, **kwargs):
        running_stats = g.running_stats
        logs = self._collect_and_clear_running_stats(running_stats)
        logs = prefix_dict_keys_with(logs, "train/batch_")
        logs["train/global_step"] = state.global_step
        wandb.run.log(logs, commit=False)

    def on_predict(self, args, state, control, metrics: Dict[str, Any], **kwargs):
        running_stats = g.running_stats
        logs = self._collect_and_clear_running_stats(
            running_stats, exclude_prefix="pre_")
        logs = prefix_dict_keys_with(logs, "all_")
        metrics.update(logs)

    def on_evaluate(self, args, state, control, metrics: Dict[str, Any], **kwargs):
        running_stats = g.running_stats
        logs = self._collect_and_clear_running_stats(
            running_stats, exclude_prefix="pre_")
        logs = prefix_dict_keys_with(logs, "all_")
        metrics.update(logs)
