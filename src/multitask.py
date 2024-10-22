"""
Alternate between minimizing simple seq2seq loss and igl loss:
* same: base model, optimizer, evaluation (both)
* different: forward pass, loss, dataloader, logging, collator
"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from absl import logging
from datasets import Dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

import globals as g
from adapter_idefics import IdeficsAdapter
from base import GEN_KWARGS, pad_left
from base import IGNORE_KEYS_FOR_EVAL as BASE_IGNORE_KEYS_FOR_EVAL
from base import BaseLossTrainer, compute_loss_from_labels
from base import compute_metrics as base_compute_metrics
from base import idefics_data_collator, idefics_transforms
from base import simple_filter as base_simple_filter
from igl import IGNORE_KEYS_FOR_EVAL as IGL_IGNORE_KEYS_FOR_EVAL
from igl import (POLICY_PREFIX, IglLossTrainer, compute_loss_for_igl,
                 igl_data_collator, igl_transform, _route_to_task_type, LookAheadWrapper)
from igl import simple_filter as igl_simple_filter
from reward_decoder_lib import RewardDecoderLib
from running_stats import BatchLossCallback, track_scalar
from transformers import Idefics2ForConditionalGeneration, Seq2SeqTrainer
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import seed_worker
from transformers.utils.peft_utils import ADAPTER_SAFE_WEIGHTS_NAME
from utils import (DatasetConfig, MultiTaskConfig, TaskConfig,
                   XTrainingArguments, clear_all_cache,
                   load_dataset_or_datasets, subdict_with_prefixed_keys)

SharedModel = Idefics2ForConditionalGeneration


class Task:
    def __init__(self, config: TaskConfig, adapter: IdeficsAdapter):
        self.adapter = adapter
        self.name = config.name
        self.training_args = dict()
        self.gen_kwargs = dict()
        self.ignore_keys_for_eval = None
        self.dataset = self.load_dataset(config.dataset_config)
        self.eval_only = config.train_or_eval == "eval"
        self.train_only = config.train_or_eval == "train"
        self.compute_metrics = None

    def load_dataset(self, dataset_config: DatasetConfig) -> DatasetDict:
        raise NotImplementedError

    def get_train_dataset(self):
        raise NotImplementedError

    def get_eval_dataset(self):
        raise NotImplementedError

    def collator(self, features: List[Dict[str, Any]], return_tensors: Optional[str] = None):
        raise NotImplementedError

    def __repr__(self):
        return f"Task({self.name})"


class Seq2SeqTask(Task):

    def __init__(self, config: TaskConfig, adapter: IdeficsAdapter):
        super().__init__(config, adapter)
        self.ignore_keys_for_eval = BASE_IGNORE_KEYS_FOR_EVAL
        self.compute_metrics = lambda eval_preds: base_compute_metrics(
            eval_preds, self.adapter, self.dataset,
            recover_generation_probs=config.seq2seq_recover_generation_probs,
            recover_h_probs=config.seq2seq_recover_h_probs,
            task_name=self.name)
        # eval_by_gen
        self.training_args = {"predict_with_generate": True, }
        self.gen_kwargs.update(GEN_KWARGS)
        self.gen_kwargs["temperature"] = config.gen_temperature
        self.gen_kwargs["do_sample"] = config.gen_do_sample
        self.gen_kwargs["_constrained_decoding"] = config.gen_constrained_decoding

    def load_dataset(self, dataset_config):
        return load_dataset_or_datasets(dataset_config, base_simple_filter)

    def get_train_dataset(self):
        return self._get_dataset("train")

    def get_eval_dataset(self):
        return self._get_dataset("dev")

    def _get_dataset(self, split: str):
        def transform(batch):
            return idefics_transforms(batch, self.adapter,
                                      eval_mode=(split == "dev"))
        return self.dataset.with_transform(transform, output_all_columns=True)

    def collator(self, features, return_tensors="pt"):
        return idefics_data_collator(features, return_tensors) | {"task": self.name}


class IglTask(Task):

    def __init__(self, config: TaskConfig, adapter: IdeficsAdapter):
        super().__init__(config, adapter)
        self.ignore_keys_for_eval = IGL_IGNORE_KEYS_FOR_EVAL
        self.compute_metrics = None

    def load_dataset(self, dataset_config):
        dataset = load_dataset_or_datasets(dataset_config, igl_simple_filter)
        assert dataset_config.igl_pos_only or dataset_config.igl_neg_only
        assert dataset_config.igl_prompt_id is not None
        rd_lib: RewardDecoderLib = g.reward_decoder_lib
        if dataset_config.igl_pos_only:
            dataset = dataset.filter(
                lambda s: rd_lib.get(s['game_turn_id'], dataset_config.igl_prompt_id) == 1)
        if dataset_config.igl_neg_only:
            dataset = dataset.filter(
                lambda s: rd_lib.get(s['game_turn_id'], dataset_config.igl_prompt_id) == -1)
        if dataset_config.igl_max_num:
            size = min(len(dataset), dataset_config.igl_max_num)
            dataset = dataset.shuffle(42).select(range(size))
        return dataset

    def get_train_dataset(self):
        return self._get_dataset("train")

    def get_eval_dataset(self):
        return self._get_dataset("dev")

    def _get_dataset(self, split: str):
        shuffle_context = (split == "train")

        def wrapped_igl_transform(batch): return igl_transform(
            batch, self.adapter, shuffle_context=shuffle_context)
        dataset = self.dataset.with_transform(
            wrapped_igl_transform, output_all_columns=True)
        # move column selection from here to collator
        return dataset

    def collator(self, features, return_tensors="pt"):
        return igl_data_collator(features, return_tensors) | {"task": self.name}


class AutoTask:

    @classmethod
    def build(cls, config: TaskConfig, adapter: IdeficsAdapter) -> Task:
        logging.info(config)
        if config.task == "Seq2SeqTask":
            return Seq2SeqTask(config, adapter)
        if config.task == "IglTask":
            return IglTask(config, adapter)
        raise NotImplementedError


def cycle(iterable):
    # https://github.com/pytorch/pytorch/issues/23900#issuecomment-518858050
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def reweigh_dict(d: Dict[str, float], weights: Dict[str, float],
                 fit="max") -> Dict[str, float]:
    logging.info(f"before removing Nones {d=}")
    assert set(weights.keys()) == set(d.keys())
    # remove keys with None weights
    weights = {k: v for k, v in weights.items() if v is not None}
    ignore_d = {k: v for k, v in d.items() if k not in weights}
    d = {k: v for k, v in d.items() if k in weights}
    assert set(weights.keys()) == set(d.keys())
    if len(weights) == 0:
        logging.info("no weights provided, after excluding Nones")
        logging.info(f"{ignore_d=}")
        return ignore_d

    denom = sum(weights.values())
    weights = {k: v/denom for k, v in weights.items()}

    logging.info(f"{weights=}")
    hy_d = {k: d[k] / weights[k] for k in weights}
    logging.info(f"{hy_d=}")  # number of total batches to support by each task
    if fit == "max":
        ref_exhausted_k = max(hy_d, key=hy_d.get)
    elif fit == "min":
        ref_exhausted_k = min(hy_d, key=hy_d.get)
    else:
        raise ValueError(f"fit={fit} not supported")
    logging.info(f"{ref_exhausted_k=}")
    ref_num_batch = d[ref_exhausted_k]
    ref_weight = weights[ref_exhausted_k]
    new_d = {k: int(ref_num_batch * weights[k] / ref_weight) for k in weights}
    logging.info(f"{new_d=}")

    new_d.update(ignore_d)
    logging.info(f"add in ignore dict {new_d=}")
    return new_d


class MultiTaskDataLoader(DataLoader):
    """
    Adapted from https://github.com/sileod/tasknet
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict: Dict[str, DataLoader], p=1,
                 gradient_accumulation_steps=1,
                 multitask_config: MultiTaskConfig = None):
        self._epoch = 0
        self._multitask_config = multitask_config
        self.dataloader_dict = dataloader_dict
        N = max([len(x)**(1-p) for x in dataloader_dict.values()])
        def f_p(x): return int(N*x**p)

        self.num_batches_dict = {
            task_name: f_p(len(dataloader))
            for task_name, dataloader in self.dataloader_dict.items()
        }
        logging.info(f"{self.num_batches_dict=}")
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            f_p(len(dataloader.dataset)) for dataloader in self.dataloader_dict.values()
        )

        if self._multitask_config.enable_weighted_task_batch:
            logging.info("weigh batches according to weights provided")
            self.num_batches_dict = reweigh_dict(
                self.num_batches_dict,
                self._multitask_config.task_batching_weights,
                fit=self._multitask_config.task_batching_weights_fit)
            # todo: update self.dataset?

        if self._multitask_config.lookahead:
            # drop some so that len(self) is divisible by grad accum steps
            num_batches_to_drop = len(self) % gradient_accumulation_steps
            num_batches_to_drop_each = num_batches_to_drop // len(
                self.num_batches_dict)
            for task_name in self.num_batches_dict:
                self.num_batches_dict[task_name] -= num_batches_to_drop_each
            num_batches_to_drop -= num_batches_to_drop_each * \
                len(self.num_batches_dict)
            for _ in range(num_batches_to_drop):
                task_name = np.random.choice(self.task_name_list)
                self.num_batches_dict[task_name] -= 1
            assert len(self) % gradient_accumulation_steps == 0

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: cycle(dataloader)
            if self._multitask_config.enable_weighted_task_batch
            else iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            batch = next(dataloader_iter_dict[task_name])
            yield batch


class MultiTaskModel(nn.Module):
    def __init__(self, model: SharedModel,  tasks: List[Task]):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.task_name_to_id = {task.name: i
                                for i, task in enumerate(self.tasks)}
        self.generation_config = model.generation_config

    def generate(self, *args, **kwargs):
        task_name = kwargs.pop("task")
        kwargs.pop("game_turn_id", None)
        task = self.tasks[self.task_name_to_id[task_name]]
        gen_kwargs = task.gen_kwargs.copy()
        if isinstance(task, IglTask):
            kwargs = subdict_with_prefixed_keys(kwargs, POLICY_PREFIX)
        constrained_decoding = gen_kwargs.pop("_constrained_decoding")
        if constrained_decoding:
            synced_gpus = kwargs.pop("synced_gpus", None)
            bsz = kwargs["input_ids"].shape[0]
            out = []
            for i in range(bsz):
                kwargs_i = {k: v[[i]] for k, v in kwargs.items()}
                kwargs_i["synced_gpus"] = synced_gpus
                generated_tokens = g.adapter.re_generate(
                    self.model, kwargs_i, gen_kwargs, return_tokens=True)
                out.append(generated_tokens[0])
            return pad_left(out, pad_token_id=g.adapter.PAD_TOKEN_ID)
        return self.model.generate(*args, **gen_kwargs, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def forward(self, task, *args, **kwargs):
        kwargs.pop("game_turn_id", None)
        if _route_to_task_type(task) == "igl":
            kwargs = subdict_with_prefixed_keys(kwargs, POLICY_PREFIX)
        return self.model(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)


class MultiTaskTrainer(Seq2SeqTrainer):
    """Custom trainer to handle logging, saving, and compute_loss
    """

    def __init__(self, tasks: List[Task], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_tasks = [task for task in tasks if not task.eval_only]
        self._eval_tasks = [task for task in tasks if not task.train_only]
        self._dummy_eval_trainers = self._create_eval_trainers(
            self._eval_tasks)

    def _create_eval_trainers(self, tasks: List[Task]) -> List[Seq2SeqTrainer]:
        trainers = []
        silent_args = {
            "report_to": None,
            "logging_strategy": "no",
        }
        for task in tasks:
            assert not task.train_only
            dummy_args = self.args.to_dict()
            dummy_args.update(silent_args)
            dummy_args.update(task.training_args)
            dummy_args = XTrainingArguments(**dummy_args)
            loss_trainer_class = IglLossTrainer if isinstance(  # IglTask does not work with kto etc.
                task, IglTask) else BaseLossTrainer
            dummy_trainer = loss_trainer_class(
                model=self.model,
                args=dummy_args,
                tokenizer=self.tokenizer,
                data_collator=task.collator,
                compute_metrics=task.compute_metrics,
                eval_dataset=task.get_eval_dataset(),
            )
            dummy_trainer._task_name = task.name  # sanity check
            if isinstance(task, IglTask):
                # dummy_trainer has its own running_cache
                # dummy_trainer is silent during eval and reports via metrics
                dummy_trainer.add_callback(BatchLossCallback())
                # okay to share, because evaluate ends after on_step_end. g.running_stats should be empty at this time.
                # guide IGL model to call compute_loss, where we track itemized
                dummy_trainer.can_return_loss = True
            trainers.append(dummy_trainer)
        return trainers

    @property
    def _dataloader_params(self):
        return {
            "batch_size": self._train_batch_size,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": seed_worker,
        }

    def get_train_dataloader(self) -> DataLoader:
        dataloaders = dict()
        for task in self._train_tasks:
            train_dataset = task.get_train_dataset()
            sampler = RandomSampler(train_dataset)
            dataloaders[task.name] = self.accelerator.prepare(
                DataLoader(train_dataset, collate_fn=task.collator,
                           sampler=sampler, **self._dataloader_params))
        args: XTrainingArguments = self.args
        loader = MultiTaskDataLoader(dataloaders,
                                     multitask_config=args.multi_task_config,
                                     gradient_accumulation_steps=args.gradient_accumulation_steps)
        return LookAheadWrapper(loader, cache_size=args.gradient_accumulation_steps, enabled=args.multi_task_config.lookahead)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Evaluate the model on all tasks"""
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        metrics = dict()
        for task, dummy_trainer in zip(self._eval_tasks, self._dummy_eval_trainers):
            assert dummy_trainer._task_name == task.name
            # sync global step so that itemized/igl_eval do not overwrite
            dummy_trainer.state.global_step = self.state.global_step
            task_metrics = dummy_trainer.evaluate(
                ignore_keys=task.ignore_keys_for_eval,
                metric_key_prefix=f"{metric_key_prefix}_{task.name}")
            metrics.update(task_metrics)
            clear_all_cache()
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics)
        # No effects because all caches have been cleared in task-evals
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        task_name = inputs["task"]

        game_turn_id = None
        if "labels" in inputs:
            assert task_name in ("base", "hb")
            game_turn_id = inputs.pop("game_turn_id")
        else:
            assert _route_to_task_type(task_name) == "igl"

        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        train_or_eval = 'train' if self.model.training else 'eval'
        args: XTrainingArguments = self.args
        record_cache = g.record_caches[train_or_eval][task_name]
        adapter = g.adapter
        running_stats = g.running_stats

        track_scalar(
            running_stats, outputs["loss"], f"{task_name}_tr_pre_process_loss")
        if "labels" not in inputs:
            loss = compute_loss_for_igl(
                inputs, outputs, igl_config=args.igl_config,
                global_step=self.state.global_step, adapter=adapter,
                record_cache=record_cache, running_stats=running_stats)
        else:
            inputs['game_turn_id'] = game_turn_id
            kwargs = dict(
                loss_config=args.sft_loss_config,
                legal_token_only=args.local_config.legal_token_only,
                global_step=self.state.global_step,
                adapter=adapter,
                record_cache=record_cache,
            )
            loss = compute_loss_from_labels(
                inputs, outputs, output=False, **kwargs)

        track_scalar(running_stats, loss,
                     f"{task_name}_tr_post_process_loss")

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """ adapted from Trainer._save to accomodate MultiTaskModel """
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir,
                                   safe_serialization=self.args.save_safetensors)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_best_model(self):
        # reduce to the case of safetensors and adapters
        best_adapter_model_path = os.path.join(
            self.state.best_model_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        assert os.path.exists(best_adapter_model_path)
        active_adapter = self.model.model.active_adapter
        self.model.model.load_adapter(
            self.state.best_model_checkpoint, active_adapter)

    def _load_from_checkpoint(self, resume_from_checkpoint: str):
        # reduce to the case of safetensors and adapters
        adapter_path = os.path.join(
            resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        assert os.path.exists(adapter_path)
        active_adapter = self.model.model.active_adapter
        self.model.model.load_adapter(
            resume_from_checkpoint, active_adapter)
