"""
functions for base seq2seq
* idefics_data_collator
* idefics_transforms
* compute_metrics
* simple_filter
"""

import json
import os
import random

import gin
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging
from datasets import Dataset

import globals as g
import wandb
from adapter_idefics import IdeficsAdapter
from io_utils import EvalPredictionIO, RecordCache, write_df_to_csv
from transformers import Seq2SeqTrainer
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import EvalPrediction
from utils import (T_MAX_LENGTH, XTrainingArguments, device, nested_apply,
                   nested_to, prefix_dict_keys_with)

IGNORE_KEYS_FOR_EVAL = ["logits", "past_key_values", "hidden_states", "attentions",
                        "image_hidden_states"]

# NOTE: keep consistent with gin
GEN_KWARGS = {"max_new_tokens": 10,
              "num_beams": 1,
              "suppress_tokens": IdeficsAdapter.SUPPRESS_TOKEN_IDS,
              "remove_invalid_values": True,
              "renormalize_logits": True,
              }


def simple_filter(dataset: Dataset) -> Dataset:
    return dataset \
        .filter(lambda sample:
                sample["is_good_deselect"] or sample["is_good_select"])


# == idefics_transforms and idefics_data_collator ==

@gin.configurable(module="idefics_transforms", allowlist=["shuffle_context"])
def idefics_transforms(example_batch, adapter: IdeficsAdapter, eval_mode: bool, shuffle_context: bool = False):
    t_max_length = T_MAX_LENGTH
    # Not configuring with gin because we want arrow to properly hash
    batch_size = len(example_batch["chats"])

    if eval_mode and shuffle_context:
        logging.log_first_n(
            logging.INFO, "shuffle_context is ignored in eval mode. Setting to shuffle_context = False", 10)
        shuffle_context = False

    if shuffle_context:
        logging.log_first_n(
            logging.INFO, "shuffle_context is enabled", 10)

    def build_processor_input(i, omit_last_answer):
        context = example_batch["context"][i]
        if shuffle_context:
            # shuffle out of place
            context = random.sample(context, len(context))
        return adapter.build_processor_input(
            context, example_batch["chats"][i],
            select_accum=example_batch["select_accum"][i],
            deselect_accum=example_batch["deselect_accum"][i],
            pre_click_selected_accum=example_batch["pre_click_selected_accum"][i],
            hash_image=True, omit_last_answer=omit_last_answer,
            sort_names=True, omit_context=False, chat_feedback=None)

    # somewhat like adapter.compose
    omit_last_answer = False
    messages_and_images = [
        build_processor_input(i, omit_last_answer=omit_last_answer)
        for i in range(batch_size)]
    messages = [m[0] for m in messages_and_images]
    prompts = adapter.processor.apply_chat_template(
        messages, add_generation_prompt=omit_last_answer)
    # removes a trailing \n (token_id=13)
    prompts = [p.strip() for p in prompts]
    images = [m[1] for m in messages_and_images]
    inputs = adapter.processor(text=prompts, images=images,
                               padding="longest", truncation=True,
                               max_length=t_max_length,
                               return_tensors="pt")

    # fine tune on the target only e.g. "A B C D"
    labels = inputs["input_ids"].detach().clone()
    if not eval_mode:
        for b in range(batch_size):
            pivot = adapter.start_index_of_last_answer(
                inputs["input_ids"][b, :])
            labels[b, :pivot] = adapter.LABEL_MASK_ID

    if eval_mode is True:
        omit_last_answer = True
        # guaranteed shorter than non eval_mode
        t_eval_max_length = labels.shape[1]
        messages_and_images = [
            build_processor_input(i, omit_last_answer=omit_last_answer)
            for i in range(batch_size)]
        messages = [m[0] for m in messages_and_images]
        prompts = adapter.processor.apply_chat_template(
            messages, add_generation_prompt=omit_last_answer)
        prompts = [p.strip() for p in prompts]
        images = [m[1] for m in messages_and_images]
        inputs = adapter.processor(text=prompts, images=images,
                                   padding="max_length", truncation=True,
                                   max_length=t_eval_max_length,
                                   return_tensors="pt")

    inputs["labels"] = labels
    inputs["game_turn_id"] = example_batch["game_turn_id"]
    return inputs


def pad_left(input_ids, pad_token_id=0):
    # pad along first dim

    # by default pad_sequence pads to the right (end of sequence),
    # here idefics are padded on the left. so we reverse the input_ids
    if isinstance(input_ids[0], list):
        # some issue in ann dataset processing, it does not return tensors
        # input_id[0] might be 1D or 2D lists. We will pad along its first dim.
        input_ids = [torch.tensor(i) for i in input_ids]
    list_of_reversed_input_ids = [i.flip(0) for i in input_ids]
    ret = torch.nn.utils.rnn.pad_sequence(
        list_of_reversed_input_ids, batch_first=True,
        padding_value=pad_token_id).flip(1)
    return ret


def idefics_data_collator(features, return_tensors=None):
    """ we need this because minibatch max sequence lengths very likely different from they were processed in transform steps with their cohort.
    """
    if return_tensors is None:
        return_tensors = "pt"
    keys_no_padding = ["pixel_values", "pixel_attention_mask"]
    keys_padding = [
        "input_ids", "attention_mask"]
    if "labels" in features[0]:  # exists for base, missing for igl
        keys_padding.append("labels")
    padding_ids = {
        "input_ids": IdeficsAdapter.PAD_TOKEN_ID,
        "attention_mask": 0,
        "labels": IdeficsAdapter.LABEL_MASK_ID,
    }

    features_no_padding = [
        {k: v for k, v in f.items() if k in keys_no_padding} for f in features]
    features_padding = [
        {k: v for k, v in f.items() if k in keys_padding} for f in features]
    ret_no_padding = default_data_collator(
        features_no_padding, return_tensors=return_tensors)
    ret_padding = {
        k: pad_left([f[k] for f in features_padding],
                    pad_token_id=padding_ids[k])
        for k in keys_padding}

    ret = ret_padding | ret_no_padding
    if "game_turn_id" in features[0]:
        ret['game_turn_id'] = [f["game_turn_id"] for f in features]
    return ret

# == compute_loss_from_labels ==


class BaseLossTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        args: XTrainingArguments = self.args
        adapter = g.adapter
        assert not model.training
        task_name = inputs["task"]
        record_cache = g.record_caches["eval"][task_name]

        game_turn_id = inputs.pop("game_turn_id")

        outputs = model(**inputs)

        inputs['game_turn_id'] = game_turn_id

        loss = compute_loss_from_labels(
            inputs, outputs,
            loss_config=args.sft_loss_config,
            legal_token_only=args.local_config.legal_token_only,
            global_step=self.state.global_step,
            adapter=adapter,
            record_cache=record_cache,
            tag="eval",
            output=False)

        return (loss, outputs) if return_outputs else loss


def compute_loss_from_labels(inputs, outputs,
                             loss_config=None,
                             legal_token_only=None,
                             global_step=None,
                             adapter: IdeficsAdapter = None,
                             record_cache: RecordCache = None,
                             tag: str = "train",
                             output=False):
    labels = inputs['labels']
    input_ids = inputs['input_ids']
    game_turn_id = inputs['game_turn_id']

    ignore_index = loss_config.ignore_index
    label_smoothing = loss_config.label_smoothing
    temperature = loss_config.temperature_scaling
    batch_size = labels.shape[0]

    logits = outputs.logits[..., :-1, :]
    labels = labels[..., 1:]
    input_ids = input_ids[..., 1:]  # SFT labels useful for extract probs

    # CE with label smoothing
    logits = logits / temperature
    if not legal_token_only:
        loss_fn = nn.CrossEntropyLoss(
            reduction="none", ignore_index=ignore_index, label_smoothing=label_smoothing)
        loss = loss_fn(
            logits.transpose(2, 1),  # (bs, vocab, seq_len-1)
            labels  # (bs, seq_len-1)
        )  # (bs, seq_len-1)
    else:
        logits = logits.masked_fill(
            ~IdeficsAdapter.LEGAL_TOKEN_MASK, -torch.inf)
        loss_fn = nn.CrossEntropyLoss(
            reduction="none", ignore_index=ignore_index)
        # nn.CrossEntropyLoss label smoothing does not work with -torch.inf
        loss = loss_fn(
            logits.transpose(2, 1),  # (bs, vocab, seq_len-1)
            labels  # (bs, seq_len-1)
        )  # (bs, seq_len-1)
        if label_smoothing > 0:
            # manually smooth over legal tokens only
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.masked_fill(
                ~IdeficsAdapter.LEGAL_TOKEN_MASK, torch.nan)
            # (bs, seq_len-1)
            smoothed_loss = - log_probs.nansum(dim=-1)
            loss = (1 - label_smoothing) * loss + \
                label_smoothing * smoothed_loss

    loss_mask = (labels == ignore_index)
    loss = loss.masked_fill(loss_mask, torch.nan)
    loss = loss.nanmean(dim=-1)  # avg per seq, then per batch

    probs_action, out = adapter.extract_policy_action(
        logits, input_ids, from_generation=False, output=True)

    out = {
        "game_turn_id": game_turn_id,
        'global_step': [global_step] * batch_size,
        'prob': probs_action,
        'loss': loss,
        **prefix_dict_keys_with(out, 'extract_'),
        # distinguish h_extract_ and gen_extract_ for eval_base
        'tag': [tag] * batch_size,

    }
    # distinguish h_extract_ and gen_extract_ for eval_base
    record_cache.batch_record(out)
    loss = loss.mean()  # (1,)
    return (loss, out) if output else loss


# == compute_metrics ==

# decode: token ids to strings e.g. select A B deselect C D
# unhash: strings to image names e.g. {pageA-1.png}

def compute_similarity_accuracy(preds, targets, similarity):
    if len(targets) == 0:
        return 1.0 if len(preds) == 0 else 0.0
    ret = []
    blocks = [set(v) for v in similarity.values()]
    flat_blocks = set().union(*blocks)
    for target in targets:
        if target in flat_blocks:
            is_sim_accurate = any(
                {target, pred} <= block for pred in preds for block in blocks)
        else:
            is_sim_accurate = target in preds
        ret.append(1 if is_sim_accurate else 0)
    return np.average(ret)


def safe_decode(preds, adapter: IdeficsAdapter):
    # short generation batch pad right with -100
    preds[preds == -100] = adapter.PAD_TOKEN_ID
    decoded_preds = adapter.processor.batch_decode(
        preds, skip_special_tokens=True)
    decoded_array = [d[-30:] for d in decoded_preds]
    return decoded_array


def nonempty(x):
    return len(x) > 0


def compute_instance_record(instance, adapter: IdeficsAdapter):
    decoded_preds, decoded_targets, dp = instance

    # parse
    h_preds_select, h_preds_deselect = adapter.parse_raw(decoded_preds)
    empty_pred = len(h_preds_select) == 0 and len(h_preds_deselect) == 0
    h_targets_select, h_targets_deselect = adapter.parse_raw(decoded_targets)

    # unhash
    def unhash(x): return adapter.unhash(dp["context"], x)
    preds_select, preds_deselect, targets_select, targets_deselect = nested_apply(
        unhash,
        (h_preds_select, h_preds_deselect, h_targets_select, h_targets_deselect)
    )

    # all entries below are unhashed. i.e. ends with .png
    # all sets

    game_context = set(dp["context"])
    currently_selected = set(dp["currently_selected"])
    currently_not_selected = game_context - currently_selected
    inconsistent_selection = nonempty(preds_select & currently_selected)
    inconsistent_deselection = nonempty(
        preds_deselect & currently_not_selected)
    inconsistent_preds = inconsistent_selection or inconsistent_deselection
    prev_deselected = set().union(*[set(s)for s in dp["deselect_accum"][:-1]])
    maybe_inefficient_preds_select_prev_deselect = nonempty(
        prev_deselected & preds_select)
    maybe_inefficient_targets_select_prev_deselect = nonempty(
        prev_deselected & targets_select)

    preds = preds_select | preds_deselect
    targets = targets_select | targets_deselect

    # sets
    hash_out = dict(
        preds_select=preds_select,
        preds_deselect=preds_deselect,
        targets_select=targets_select,
        targets_deselect=targets_deselect,
        preds=preds,
        targets=targets,
    )
    # bool
    hash_bool_out = dict(
        empty_pred=empty_pred,
        inconsistent_selection=inconsistent_selection,
        inconsistent_deselection=inconsistent_deselection,
        inconsistent_preds=inconsistent_preds,
        maybe_inefficient_preds_select_prev_deselect=maybe_inefficient_preds_select_prev_deselect,
        maybe_inefficient_targets_select_prev_deselect=maybe_inefficient_targets_select_prev_deselect,
    )
    hash_binary_out = {k: 1 if v else 0 for k, v in hash_bool_out.items()}

    similarity = json.loads(dp["similarity"])
    game_targets = set(dp["targets"])

    assert len(targets) != 0, "listener must have selected something"
    game_non_targets = game_context - game_targets

    bool_out = dict(
        has_select_target=nonempty(targets_select),
        has_deselect_target=nonempty(targets_deselect),
        has_mixed_turns=nonempty(
            targets_select) and nonempty(targets_deselect),
        has_mixed_preds=nonempty(preds_select) and nonempty(preds_deselect),
        accuracy=preds == targets,
        raw_accuracy=decoded_preds == decoded_targets,
        select_accuracy=preds_select == targets_select,
        deselect_accuracy=preds_deselect == targets_deselect,
        unintended_select_positive=nonempty(
            preds_select & (game_targets - targets_select)),
        # not a turn target but a game target
        gt_select_target=nonempty(preds_select & game_targets),
        gt_deselect_non_target=nonempty(preds_deselect & game_non_targets),
        gt_select_non_target=nonempty(preds_select & game_non_targets),
        gt_deselect_target=nonempty(preds_deselect & game_targets),
    )
    binary_out = {k: 1 if v else 0 for k, v in bool_out.items()}

    num_out = dict(
        len_preds=len(preds),
        len_preds_len_clicks_ratio=len(preds) / len(targets),
        sim_accuracy=compute_similarity_accuracy(preds, targets, similarity),
        sim_select_accuracy=compute_similarity_accuracy(
            preds_select, targets_select, similarity),
        true_positive=len(preds & targets) / len(targets),
        false_positive=len(preds - targets) / len(targets),
        false_negative=len(targets - preds) / len(targets),
    )

    maybe_na_out = dict(
        select_accuracy_on_select_turns=np.nan if len(
            targets_select) == 0 else int(preds_select == targets_select),
        deselect_accuracy_on_deselect_turns=np.nan if len(
            targets_deselect) == 0 else int(preds_deselect == targets_deselect),
        accuracy_on_later_turns=np.nan if adapter.keep_max_turns is None or dp[
            "turn_id"] < adapter.keep_max_turns else binary_out["accuracy"],
    )

    maybe_na_out.update({
        f"accuracy_after_{i}_turns": np.nan if dp["turn_id"] < i else binary_out["accuracy"] for i in range(3, 10)
    })
    maybe_na_out.update({
        f"accuracy_until_{i}_turns": np.nan if dp["turn_id"] > i else binary_out["accuracy"] for i in range(0, 10)
    })

    str_out = dict(
        game_turn_id=dp["game_turn_id"],
        decoded_preds=decoded_preds,
        decoded_targets=decoded_targets,
        similarity=dp['similarity'],
    )

    # these will be collected by wandb
    numeric_out = hash_binary_out | binary_out | num_out | maybe_na_out

    return hash_out | str_out | numeric_out


def save_local(df: pd.DataFrame, trainer, task_name):
    df = df.copy()
    df["global_step"] = trainer.state.global_step
    # keep consistent with RecordCacheCallback.on_evaluate
    epoch = round(trainer.state.epoch) if trainer.state.epoch else 0
    file_name = f"global_step_{trainer.state.global_step}_epoch_{epoch}_task_{task_name}.csv"
    csv_path = os.path.join(trainer.args.output_dir,
                            "itemized", f"{task_name}_gen", file_name)
    write_df_to_csv(df, csv_path)


def save_eval_preds(eval_preds: EvalPrediction, trainer, task_name):
    eval_preds_npz_path = os.path.join(
        trainer.args.output_dir, "itemized", task_name+"_gen",
        f"global_step_{trainer.state.global_step}_task_{task_name}_eval_preds.npz")
    EvalPredictionIO.save_npz(eval_preds_npz_path, eval_preds)


def upload_artifact_to_wandb(df, trainer, task_name):
    artifacts = {
        "train/global_step": trainer.state.global_step,
        f"eval/itemized_{task_name}": wandb.Table(dataframe=df.dropna()),
    }
    if 'h_prob' in df.columns and 'h_loss' in df.columns:
        artifacts[f"eval/itemized_{task_name}_prob_h"] = wandb.Histogram(
            df['h_prob'].dropna())
        artifacts[f"eval/itemized_{task_name}_loss_h"] = wandb.Histogram(
            df['h_loss'].dropna())
    if 'gen_prob' in df.columns:
        artifacts[f"eval/itemized_{task_name}_prob_gen"] = wandb.Histogram(
            df['gen_prob'].dropna())
    wandb.log(artifacts, commit=False)


def annotate_prob_and_loss(example_batch, adapter: IdeficsAdapter, record_cache: RecordCache, tag: str):
    # use back doors
    trainer = adapter.trainer
    policy_model_inference_only = adapter.policy_model

    inputs = example_batch
    game_turn_id = inputs.pop("game_turn_id")
    nested_to(inputs, device())
    kwargs = dict(
        loss_config=trainer.args.sft_loss_config,
        legal_token_only=trainer.args.local_config.legal_token_only,
        global_step=trainer.state.global_step,
        record_cache=record_cache,
        tag=tag,
        adapter=adapter,
    )
    with torch.inference_mode():
        outputs = policy_model_inference_only(**inputs)
        inputs['game_turn_id'] = game_turn_id
        _, out = compute_loss_from_labels(
            inputs, outputs, output=True, **kwargs)

    # compute_loss_from_labels will be write to record_cache,
    # and multitask_trainer.on_evaluate will collect from record_cache
    # here we just need to extract prob and loss for uploading gradients to wandb
    keys = ("game_turn_id", "prob", "loss")
    return {k: out[k] for k in keys}


def midway_transform_and_collate(example_batch, adapter: IdeficsAdapter):
    input_ids: torch.Tensor = example_batch['_midway_input_ids']
    attention_mask = input_ids.ne(adapter.PAD_TOKEN_ID).long()
    # hack to suppress "perform batched generation with padding_side='right'"
    # last 10 tokens are always attended (max new tokens) as if they were generated
    attention_mask[:, -10:] = 1
    images = nested_apply(adapter.get_image, example_batch['context'])
    inputs = adapter.processor(images=images, return_tensors="pt")

    labels = input_ids.detach().clone()
    for b in range(input_ids.shape[0]):
        pivot = adapter.start_index_of_last_answer(input_ids[b, :])
        labels[b, :pivot] = adapter.LABEL_MASK_ID
        pivot = adapter.end_index_of_last_answer(input_ids[b, :])
        labels[b, pivot:] = adapter.LABEL_MASK_ID

    return dict(
        game_turn_id=example_batch['game_turn_id'],
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=inputs["pixel_values"],
        pixel_attention_mask=inputs["pixel_attention_mask"],
        labels=labels,
    )


def recover_generation_stats(input_ids: torch.Tensor,  # (bs, seq_len)
                             adapter: IdeficsAdapter,
                             dataset: Dataset,
                             record_cache: RecordCache,
                             tag: str):
    """
    given input_ids, reconstruct idefics_transform inputs
    (1) construct mask
    (2) build pixel_values

    return: probs, and extract_*, loss from forward pass etc. in itemized df view, indexed by game_turn_id
    """
    map_kwargs = dict(
        load_from_cache_file=False,
        batch_size=8,
        batched=True,
        keep_in_memory=True,
    )
    dataset = dataset.add_column("_midway_input_ids", [i for i in input_ids])
    dataset.set_format(type="torch", columns=["_midway_input_ids"])

    def t(x): return midway_transform_and_collate(x, adapter)
    ds = dataset.map(t, remove_columns=dataset.column_names,
                     desc="midway_transform_and_collate",
                     **map_kwargs)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask",
                                         "pixel_values", "pixel_attention_mask", "labels"])

    def a(x): return annotate_prob_and_loss(x, adapter, record_cache, tag)
    ds = ds.map(a, remove_columns=ds.column_names,
                desc="annotate_prob_and_loss", **map_kwargs)
    assert set(ds.column_names) == set(["game_turn_id", "prob", "loss"])
    df = ds.to_pandas()
    del ds
    return df


def merge_df(df: pd.DataFrame, df_gen, df_h):
    for d, prefix in [(df_gen, "gen_"), (df_h, "h_")]:
        if d is None:
            continue
        assert len(df) == len(d)
        assert (df['game_turn_id'] == d['game_turn_id']).all()
        d: pd.DataFrame = d.copy()
        d.drop(columns=["game_turn_id"], inplace=True)
        d.columns = [f"{prefix}{c}" for c in d.columns]
        df = df.merge(d, left_index=True, right_index=True)
    return df


def compute_metrics(eval_preds: EvalPrediction, adapter: IdeficsAdapter,
                    dataset: Dataset, task_name: str,
                    recover_generation_probs=False,
                    recover_h_probs=False):
    num_samples = len(dataset)
    trainer: Seq2SeqTrainer = adapter.trainer
    save_eval_preds(eval_preds, trainer, task_name)
    preds, labels = eval_preds  # N x t_max_length, N x t_max_length
    assert num_samples == preds.shape[0], \
        f"mismatching length: {num_samples=} vs. {preds.shape[0]}"
    decoded_preds_array = safe_decode(preds, adapter)
    decoded_targets_array = safe_decode(labels, adapter)

    df_gen, df_h = None, None
    record_cache = g.record_caches["eval"][task_name]
    if recover_generation_probs:
        df_gen = recover_generation_stats(
            preds, adapter, dataset, record_cache, tag="gen")
    if recover_h_probs:
        df_h = recover_generation_stats(
            labels, adapter, dataset, record_cache, tag="h")

    def _compute_instance_record(x): return compute_instance_record(x, adapter)
    instance_records = list(map(_compute_instance_record, zip(
        decoded_preds_array, decoded_targets_array, dataset)))
    df = pd.DataFrame(instance_records)
    # with numeric and nonnumeric columns

    df = merge_df(df, df_gen, df_h)
    save_local(df, trainer, task_name)
    if wandb.run:
        upload_artifact_to_wandb(df, trainer, task_name)

    # aggregate
    numeric_df = df.select_dtypes(include=['number'])
    metrics = numeric_df.mean().to_dict()
    metrics['size'] = num_samples
    return metrics
