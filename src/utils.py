import dataclasses
import gc
import json
import logging
import os
import re
import sys
from functools import cache
from pathlib import Path
from typing import Any, Dict, List, Union

import datasets
import gin
import numpy as np
import pandas as pd
import torch
from absl import logging as alogging
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image

import wandb
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          Idefics2ForConditionalGeneration,
                          Seq2SeqTrainingArguments)
from transformers.trainer_callback import TrainerCallback

SLURM_JOB_ID = int(os.environ.get("SLURM_JOB_ID", -1))
T_MAX_LENGTH = 2048


@gin.register(module="DatasetConfig")
@dataclasses.dataclass
class DatasetConfig:
    input_dir: str = "data"
    dataset: Union[str, List[str]] = "full"
    interleave_probabilities: List[float] = dataclasses.field(
        default_factory=list)
    pctg_game_id: float = 1.0
    max_num: int = None
    debug: bool = False
    simple_filter: bool = False
    game_id_exclude_filepath_or_dir: str = None
    game_turn_id_exclude_filepath: str = None
    game_turn_id_include_filepath: str = None
    igl_pos_only: bool = False
    igl_neg_only: bool = False
    igl_prompt_id: str = None
    igl_max_num: int = None


@gin.register(module="LocalConfig")
@dataclasses.dataclass
class LocalConfig:
    debug: bool = False
    input_dir: str = "data"
    checkpoint: str = "HuggingFaceM4/idefics2-8b"
    checkpoint_path: str = None
    resume_from_checkpoint_path: str = None
    slurm_job_id: int = SLURM_JOB_ID
    legal_token_only: bool = False
    keep_max_turns: int = None
    eval_overwrite: bool = False
    attn_implementation: str = "flash_attention_2"  # or None
    run_notes: str = ""
    project_name: str = "multiref_multitask"


@gin.register(module="IglConfig")
@dataclasses.dataclass
class IglConfig:
    reward_remap: Dict[float, float] = dataclasses.field(default_factory=dict)
    kl_coeff: float = 0.0
    prob_poor_clamp_min: float = 0.0
    prob_clamp_min: float = 0.0
    label_smoothing: float = 0.0
    kto_only: bool = False
    kto_beta: float = 0.1
    kto_desirable_coeff: float = 1.0
    kto_undesirable_coeff: float = 1.0
    prompt_id: str = "nothing"  # match to nothing if unused (say just base)


@gin.register(module="TaskConfig")
@dataclasses.dataclass
class TaskConfig:
    task: str = None
    name: str = None
    dataset_config: DatasetConfig = None
    train_or_eval: str = None
    seq2seq_recover_generation_probs: bool = False
    seq2seq_recover_h_probs: bool = False
    gen_temperature: float = 1.0
    gen_do_sample: bool = False
    gen_constrained_decoding: bool = False


@gin.register(module="MultiTaskConfig")
@dataclasses.dataclass
class MultiTaskConfig:
    enable_weighted_task_batch: bool = False
    task_batching_weights: Dict[str, float] = dataclasses.field(
        default_factory=dict)
    task_batching_weights_fit: str = "max"
    lookahead: bool = True
    # round batches to multiple of grad accum steps, also dictates lookahead behavior


@gin.register(module="SftLossConfig")
@dataclasses.dataclass
class SftLossConfig:
    label_smoothing: float = 0.0
    temperature_scaling: float = 1.0
    ignore_index: int = 32001  # Adapter.LABEL_MASK_ID


@gin.configurable(module="XTrainingArguments")
@dataclasses.dataclass
class XTrainingArguments(Seq2SeqTrainingArguments):
    local_config: LocalConfig = None
    bnb_config: BitsAndBytesConfig = None
    lora_config: LoraConfig = None
    igl_config: IglConfig = None
    task_configs: Dict[str, TaskConfig] = dataclasses.field(
        default_factory=dict)
    multi_task_config: MultiTaskConfig = None
    sft_loss_config: SftLossConfig = None


def device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_logger(name, level=logging.WARNING):
    logger = logging.getLogger(name)

    format = "[%(asctime)s:%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(format, datefmt)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(level)

    logger.setLevel(level)
    logger.addHandler(stdout_handler)
    return logger


def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


def load_idefics_model(checkpoint: str,
                       checkpoint_adapter: str = None,
                       is_quantized: bool = True,
                       is_trainable: bool = False,
                       bnb_config: BitsAndBytesConfig = None,
                       lora_config: LoraConfig = None,
                       attn_implementation: str = "flash_attention_2",
                       revision: str = None
                       ):
    if lora_config is None:
        # Keep consistent with gin config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
            init_lora_weights="gaussian",
            use_dora=True
        )

    alogging.info("loading pretrained model")
    alogging.info(f"{is_quantized=}")
    alogging.info(f"{bnb_config=}")
    model = Idefics2ForConditionalGeneration.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=bnb_config,
        attn_implementation=attn_implementation,
    )
    alogging.info(type(model))

    if checkpoint_adapter is None or checkpoint_adapter == "":
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        alogging.info(f"done: loading peft {checkpoint} model")
        return model

    peft_model = PeftModel.from_pretrained(
        model, checkpoint_adapter, is_trainable=is_trainable, revision=revision)
    alogging.info(
        f"done: loading peft idefics model from {checkpoint_adapter}")
    return peft_model


def get_idefics_processor(checkpoint="HuggingFaceM4/idefics2-8b"):
    # keep multiples of 14
    size = {"longest_edge": 224, "shortest_edge": 224}
    processor = AutoProcessor.from_pretrained(
        checkpoint, do_image_splitting=False, size=size)
    return processor


def nested_apply(h, s):
    # h is an unary function, s is one of N, tuple of N, list of N, or set of N
    if isinstance(s, str):
        return h(s)
    ret = [nested_apply(h, i) for i in s]
    if isinstance(s, tuple):
        return tuple(ret)
    if isinstance(s, set):
        return set(ret)
    return ret


def sorted_list(s):
    return sorted(list(s))


def exist_checkpoint(output_dir: str) -> bool:
    output_dir = Path(output_dir)
    ret = any(
        f.name.startswith("checkpoint") for f in output_dir.iterdir())
    return ret


def exist_run_id(output_dir: str) -> bool:
    output_dir = Path(output_dir)
    ret = any(
        f.name == "wandb_run_id.txt" for f in output_dir.iterdir())
    return ret


def save_run_id(run_id: str, output_dir: str):
    with open(os.path.join(output_dir, "wandb_run_id.txt"), "w") as f:
        f.write(run_id)


def read_run_id(output_dir: str):
    with open(os.path.join(output_dir, "wandb_run_id.txt"), "r") as f:
        ret = f.read().strip()
    return ret


def track_extra_wandb_config(xargs: XTrainingArguments, tasks: List[Any]):
    train_task_names = [name for (name, task) in xargs.task_configs.items()
                        if task.train_or_eval == "train"]
    task_type = "+".join(train_task_names)
    task_dataset_len = {
        task.name + '.num': len(task.dataset) for task in tasks}

    wandb.config.update(
        {"batch_size": xargs.per_device_train_batch_size * xargs.gradient_accumulation_steps,
         "task_type": task_type,
         **task_dataset_len, })
    wandb.run.notes = xargs.local_config.run_notes


def save_game_turn_ids(tasks: List[Any], output_dir: str):
    dir = os.path.join(output_dir, "game_turn_ids")
    os.makedirs(dir, exist_ok=True)
    for task in tasks:
        dataset = task.dataset
        game_turn_ids = sorted(dataset["game_turn_id"])
        filepath = os.path.join(
            output_dir, "game_turn_ids", f"{task.name}.txt")
        pd.DataFrame(game_turn_ids).to_csv(filepath, index=False, header=False)


@ cache
def extract_similarity(dataset: Dataset) -> List[Dict[str, List]]:
    similarities = dataset[:]["similarity"]
    similarities = [json.loads(s) for s in similarities]
    return similarities


def pctg_subset_by_game_id(dataset: Dataset, pctg: float) -> Dataset:
    _prev_num_utterances = len(dataset)
    game_ids = np.array(list(dict.fromkeys(dataset["game_id"])))
    alogging.info(f"{game_ids.shape=}")
    all_num_game_id = game_ids.shape[0]
    cap_num_game_id = int(all_num_game_id * pctg)
    selected_index = np.random.choice(
        all_num_game_id, size=cap_num_game_id, replace=False)
    selected_game_ids = set(game_ids[selected_index])
    dataset = dataset.filter(
        lambda x: maybe_strip_aug_suffix(x["game_id"]) in selected_game_ids)
    _new_num_utterances = len(dataset)
    alogging.info(
        f"{all_num_game_id=}, {cap_num_game_id=}, {_prev_num_utterances=}, {_new_num_utterances=}")
    alogging.info(
        f"percentage of game_id training: {pctg}")
    alogging.info(
        f"percentage of utterance training: {_new_num_utterances / _prev_num_utterances}")
    return dataset


def read_txt_as_list(filepath: str):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return [l.strip() for l in lines]


def maybe_strip_aug_suffix(game_turn_id: str):
    return game_turn_id.split("_aug")[0]


def load_dataset_or_datasets(config: DatasetConfig, simple_filter=None):
    input_dir = Path(config.input_dir)
    if isinstance(config.dataset, str):
        dataset = datasets.load_from_disk(
            input_dir.joinpath(config.dataset))
    else:
        dsets = [datasets.load_from_disk(input_dir.joinpath(p))
                 for p in config.dataset]
        if config.interleave_probabilities == []:
            dataset = datasets.concatenate_datasets(dsets)
        else:
            assert len(config.dataset) == len(
                config.interleave_probabilities)
            dataset = datasets.interleave_datasets(
                dsets,
                probabilities=config.interleave_probabilities,
                seed=42, stopping_strategy="first_exhausted")
            # Make sure the first exhausted is not the augmented.

    if config.simple_filter:
        alogging.info("applying simple filter")
        dataset = simple_filter(dataset)

    if 0.0 < config.pctg_game_id < 1.0:
        dataset = pctg_subset_by_game_id(dataset, config.pctg_game_id)

    assert not (
        config.game_turn_id_exclude_filepath
        and config.game_turn_id_include_filepath), \
        "cannot use both exclude and include"
    if config.game_turn_id_exclude_filepath is not None:
        filepath = input_dir/config.game_turn_id_exclude_filepath
        alogging.info(f"excluding game_turn_ids from {filepath}")
        game_turn_ids_exclude = set(read_txt_as_list(filepath))
        dataset = dataset.filter(
            lambda x: maybe_strip_aug_suffix(x['game_turn_id']) not in
            game_turn_ids_exclude)
    if config.game_turn_id_include_filepath is not None:
        filepath = input_dir/config.game_turn_id_include_filepath
        alogging.info(f"only including game_turn_ids from {filepath}")
        game_turn_ids_include = set(read_txt_as_list(filepath))
        dataset = dataset.filter(
            lambda x: maybe_strip_aug_suffix(x['game_turn_id']) in
            game_turn_ids_include)
    if config.game_id_exclude_filepath_or_dir is not None:
        filepath = input_dir/config.game_id_exclude_filepath_or_dir
        alogging.info(f"excluding game_ids from {filepath}")
        if str(filepath).endswith(".txt"):
            filepaths = [filepath]
        else:
            filepaths = [filepath/f for f in os.listdir(filepath)
                         if f.endswith(".txt")]
        game_ids_exclude = set().union(
            *[set(read_txt_as_list(f)) for f in filepaths])
        dataset = dataset.filter(
            lambda x: x['game_id'] not in game_ids_exclude)

    if config.max_num is not None:
        size = min(len(dataset), config.max_num)
        dataset = dataset.shuffle(42).select(range(size))

    if config.debug:
        dataset = dataset.select(range(32))
        return dataset

    return dataset


def clear_all_cache():
    torch.cuda.empty_cache()
    gc.collect()


def subdict_with_prefixed_keys(d, prefix: str):
    return {k.removeprefix(prefix): d[k] for k in d if k.startswith(prefix)}


def prefix_dict_keys_with(d, prefix: str):
    return {prefix+k: d[k] for k in d}


def suffix_dict_keys_with(d, suffix: str):
    return {k+suffix: d[k] for k in d}


def nested_to(d, device):
    for k in d:
        d[k] = d[k].to(device)


def load_human_bot_annotation(ann):
    ANN_PATTERN = r'^page[a-zA-Z0-9_-]+\.png(?:, page[a-zA-Z0-9_-]+\.png)*$'
    df_annotation = pd.read_csv(f'data/human_bot_annotation/{ann}.csv')
    df_annotation['turn_id'] = df_annotation['turn_on_vis'].map(lambda t: t-1)
    well_formatted = df_annotation['labels'].map(
        lambda x: bool(re.fullmatch(ANN_PATTERN, x)))
    assert well_formatted.all()
    df_annotation['labels'] = df_annotation['labels'].map(
        lambda xs: xs.split(', '))

    return df_annotation[['turn_id', 'game_id', 'labels']]


class OverflowGradientCallback(TrainerCallback):
    """Custom callback to detect gradient overflow

    Patch until they enable `error_if_nonfinite` by default in torch.utils.clip_grad_norm_
    """

    def __init__(self):
        super().__init__()

    def on_optimizer_step_begin(self, args, state, control, **kwargs):
        """Examine gradient norm before optimizer step"""
        model = kwargs["model"]
        assert model.training
        grads = [param.grad for _, param in model.named_parameters()
                 if param.requires_grad and param.grad is not None]
        has_nan_gradient = any(torch.isnan(grad).any() for grad in grads)
        has_inf_gradient = any(torch.isinf(grad).any() for grad in grads)

        logs = {"train/grad_has_nan": float(has_nan_gradient),
                "train/grad_has_inf": float(has_inf_gradient),
                "train/global_step": state.global_step}
        wandb.run.log(logs, commit=False)
        return


class BestMetricsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.best_metrics_max = {
            "eval_base_ood_raw_accuracy": 0.0,
            "eval_base_ood_accuracy": 0.0,
        }
        self.best_metrics_min = {
            "eval_base_ood_loss": float("inf"),
        }

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        for k, v in self.best_metrics_max.items():
            if k in metrics and metrics[k] > v:
                self.best_metrics_max[k] = metrics[k]
        for k, v in self.best_metrics_min.items():
            if k in metrics and metrics[k] < v:
                self.best_metrics_min[k] = metrics[k]

        logs = {**suffix_dict_keys_with(self.best_metrics_max, "_best"),
                **suffix_dict_keys_with(self.best_metrics_min, "_best"),
                "train/global_step": state.global_step}
        logs = subdict_with_prefixed_keys(logs, "eval_")
        logs = prefix_dict_keys_with(logs, "eval/")
        wandb.run.log(logs, commit=False)

        return control
