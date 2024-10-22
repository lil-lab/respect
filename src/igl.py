"""
functions for igl

* igl_transform and igl_data_collator
* simple_filter
"""

from typing import Any, Dict, List

import gin
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader

import globals as g
from adapter_idefics import IdeficsAdapter
from base import idefics_data_collator, idefics_transforms
from dataset.augmentation import SingleClick
from io_utils import RecordCache
from reward_decoder_lib import RewardDecoderLib
from running_stats import track_scalar
from transformers import Seq2SeqTrainer
from transformers.data.data_collator import default_data_collator
from transformers.models.idefics2.modeling_idefics2 import \
    Idefics2CausalLMOutputWithPast
from transformers.trainer_callback import TrainerCallback
from utils import IglConfig, clear_all_cache, device, prefix_dict_keys_with

POLICY_PREFIX = "policy."
KTO_PREFIX = "kto."


def igl_keep_columns(c: str):
    return (c in ["prob_action_poor",
                  "game_turn_id", "game_id", "turn_id", "policy_id"]
            or c.startswith(POLICY_PREFIX)
            or c.startswith(KTO_PREFIX))


@gin.configurable(module="igl_data_collator", allowlist=["prompt_id"])
def igl_data_collator(features, return_tensors=None, prompt_id=None):
    if return_tensors is None:
        return_tensors = "pt"

    idefics_features = [
        {k.removeprefix(POLICY_PREFIX): v
         for k, v in f.items() if k.startswith(POLICY_PREFIX)}
        for f in features]
    ret_idefics = idefics_data_collator(idefics_features, return_tensors)
    ret_idefics = prefix_dict_keys_with(ret_idefics, POLICY_PREFIX)

    kto_features = [
        {k.removeprefix(KTO_PREFIX): v
            for k, v in f.items() if k.startswith(KTO_PREFIX)}
        for f in features]
    ret_kto = idefics_data_collator(kto_features, return_tensors)
    ret_kto = prefix_dict_keys_with(ret_kto, KTO_PREFIX)

    other_features = [{k: v for k, v in f.items()
                       if igl_keep_columns(k)
                       and (k not in ret_idefics)
                       and (k not in ret_kto)}
                      for f in features]
    ret_other = default_data_collator(other_features, return_tensors)
    # return_tensors = "pt" omits string features like "game_turn_id"
    ret_other["game_turn_id"] = [f["game_turn_id"] for f in features]
    ret_other["game_id"] = [f["game_id"] for f in features]
    ret_other["policy_id"] = [f["policy_id"] for f in features]

    rd_lib: RewardDecoderLib = g.reward_decoder_lib
    ret_other["reward"] = torch.tensor(
        [float(rd_lib.get(gt_id, prompt_id)) for gt_id in ret_other["game_turn_id"]])

    return ret_idefics | ret_other | ret_kto


def col_view_to_row_view(d: Dict[str, Any]):
    num_rows = len(d["game_turn_id"])
    return [{k: v[i] for k, v in d.items()} for i in range(num_rows)]


def row_view_to_col_view(l: List[Dict[str, Any]]):
    return {k: [d[k] for d in l] for k in l[0]}


def igl_transform(example_batch, adapter_idefics: IdeficsAdapter,
                  shuffle_context=False):
    idefics_inputs = idefics_transforms(
        example_batch, adapter_idefics, eval_mode=False, shuffle_context=shuffle_context)
    idefics_inputs.pop("game_turn_id")

    prefixed_idefics_tensor_inputs = prefix_dict_keys_with(
        idefics_inputs, POLICY_PREFIX)

    kto_augment = SingleClick().augment
    kto_example_batch = row_view_to_col_view(
        [kto_augment(example)
         for example in col_view_to_row_view(example_batch)]
    )
    kto_inputs = idefics_transforms(
        kto_example_batch, adapter_idefics, eval_mode=False, shuffle_context=shuffle_context)
    kto_inputs = {"input_ids": kto_inputs["input_ids"],
                  "attention_mask": kto_inputs["attention_mask"]}
    prefixed_kto_tensor_inputs = prefix_dict_keys_with(kto_inputs, KTO_PREFIX)

    misc_keys = ("game_id", "turn_id", "game_turn_id",
                 "prob_action_poor", "policy_id")
    misc = {k: example_batch[k] for k in misc_keys}
    return prefixed_idefics_tensor_inputs | misc | prefixed_kto_tensor_inputs


def simple_filter(dataset: Dataset) -> Dataset:
    """
    1. remove the last in an interaction because there is no feedback, roughly 10%
    """
    return dataset.filter(lambda s: s["chat_feedback"] != "")


# ignore all of them to avoid concat in trainer for compute_metrics. will lead to memory blow up
IGNORE_KEYS_FOR_EVAL = ["logits", "past_key_values", "hidden_states",
                        "attentions", "image_hidden_states"]


class IglLossTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        adapter = g.adapter
        running_stats = g.running_stats
        record_caches = g.record_caches
        task_name = inputs["task"]
        assert not model.training
        record_cache = record_caches["eval"][task_name]
        loss = compute_loss_for_igl(inputs, outputs, self.args.igl_config,
                                    self.state.global_step, adapter, record_cache, running_stats)
        return (loss, outputs) if return_outputs else loss


def compute_loss_for_igl(inputs: Dict[str, Any],
                         outputs: Idefics2CausalLMOutputWithPast,
                         igl_config: IglConfig = None,
                         global_step: int = None,
                         adapter: IdeficsAdapter = None,
                         record_cache: RecordCache = None,
                         running_stats: Dict = None):
    batch_size = len(inputs["game_turn_id"])
    xs = inputs

    shifted_labels = inputs[POLICY_PREFIX + "labels"][:, 1:]
    shifted_logits = outputs.logits[:, :-1, :]  # (bs, seq_len-1, vocab)

    lprobs_action, out = adapter.extract_policy_action(
        shifted_logits, shifted_labels, output=True, return_logits=True)
    lprobs_action = lprobs_action.clamp_min(
        torch.tensor(igl_config.prob_clamp_min).log())
    lprobs_action = lprobs_action.to(device())  # (bs,)

    # The following variables have the shape of (bs,)
    probs_action_poor = xs["prob_action_poor"]
    heur_reward = xs["reward"]
    reward = torch.tensor([igl_config.reward_remap[h.item()] for h in heur_reward],
                          dtype=heur_reward.dtype).to(heur_reward.device)
    default_neu_reward = igl_config.reward_remap.get(0, 0)

    probs_action_poor = probs_action_poor.clamp_min(
        igl_config.prob_poor_clamp_min)
    lprobs_action_poor = probs_action_poor.log()

    # bandit
    q = (lprobs_action.detach().clone() - lprobs_action_poor).exp()
    debias = torch.where(reward > default_neu_reward,
                         torch.ones_like(reward), q)
    bandit_loss = - debias * reward * lprobs_action

    # kl
    kl_loss_approx = lprobs_action - lprobs_action_poor

    # kto
    # moving average per batch, see LookAheadBatchCallback
    if igl_config.kto_only:
        z = running_stats["pre_kl_batch"].mean_and_var_and_size()[
            0].to(device())
    else:
        z = torch.zeros_like(reward).to(device())
    kto_beta = igl_config.kto_beta
    logp_ratios = lprobs_action - lprobs_action_poor
    chosen_losses = 1 - F.sigmoid(kto_beta * (logp_ratios - z))
    chosen_rewards = kto_beta * logp_ratios.detach()
    rejected_losses = 1 - F.sigmoid(kto_beta * (z - logp_ratios))
    rejected_rewards = kto_beta * logp_ratios.detach()
    kto_loss = torch.where(reward > default_neu_reward,
                           igl_config.kto_desirable_coeff * chosen_losses,
                           igl_config.kto_undesirable_coeff * rejected_losses)

    itemized_loss_action = bandit_loss + igl_config.kl_coeff * kl_loss_approx

    num_tokens = [len(l) for l in out['target_ids']]
    itemized_loss_len_averaged = itemized_loss_action / \
        torch.tensor(num_tokens, dtype=torch.float32).to(device())

    # smoothed loss
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    smoothed_loss = - log_probs.mean(dim=-1)  # (bs, seq_len-1)
    # mask shifted_labels
    ignore_index = adapter.LABEL_MASK_ID
    for b in range(batch_size):
        piv = adapter.start_index_of_last_answer(shifted_labels[b, :])
        shifted_labels[b, :piv] = ignore_index
    # shifted_labels should be about the same as out['target_ids']
    smoothed_loss = smoothed_loss.masked_fill(
        shifted_labels == ignore_index, torch.nan).nanmean(-1)
    # avg over seq len, (bs,)

    # final loss per sequence, averaged over sequence length
    label_smoothing = igl_config.label_smoothing
    itemized_loss = (1 - label_smoothing) * itemized_loss_len_averaged \
        + label_smoothing * reward.abs() * smoothed_loss

    # overwrite if kto
    if igl_config.kto_only:
        itemized_loss = (1 - label_smoothing) * kto_loss \
            + label_smoothing * smoothed_loss

    loss = torch.mean(itemized_loss)  # (1,)

    def only_heur_reward_positive(x): return torch.masked_select(
        x.detach().clone(), heur_reward > default_neu_reward)

    def only_heur_reward_negative(x): return torch.masked_select(
        x.detach().clone(), heur_reward < default_neu_reward)

    probs_action = lprobs_action.detach().clone().exp()
    stats = {
        "loss": itemized_loss,
        "heur_reward": heur_reward,
        "reward": reward,
        "probs_action": probs_action,
        "probs_action_positive": only_heur_reward_positive(probs_action),
        "probs_action_negative": only_heur_reward_negative(probs_action),
        "bandit_loss": bandit_loss,
        "bandit_loss_positive": only_heur_reward_positive(bandit_loss),
        "bandit_loss_negative": only_heur_reward_negative(bandit_loss),
        "q": q,
        "debias": debias,
        "debias_positive": only_heur_reward_positive(debias),
        "debias_negative": only_heur_reward_negative(debias),
        "kl_loss_approx": kl_loss_approx,
        "smoothed_loss": smoothed_loss,
        "itemized_loss_action": itemized_loss_action,
        "itemized_loss_len_averaged": itemized_loss_len_averaged,
        "kto_loss": kto_loss,
        "kto_loss_positive": only_heur_reward_positive(kto_loss),
        "kto_loss_negative": only_heur_reward_negative(kto_loss),
        "kto_z": z * torch.ones_like(reward),
        "kto_logp_ratios": logp_ratios,
        "kto_logp_ratios_positive": only_heur_reward_positive(logp_ratios),
        "kto_logp_ratios_negative": only_heur_reward_negative(logp_ratios),
        "kto_chosen_rewards": only_heur_reward_positive(chosen_rewards),
        "kto_rejected_rewards": only_heur_reward_negative(rejected_rewards),
    }
    for k, v in stats.items():
        track_scalar(running_stats, v, k)

    record_cache.batch_record({
        "game_turn_id": xs['game_turn_id'],
        "global_step": [global_step] * batch_size,
        "loss": itemized_loss,
        "probs_action": probs_action,
        "probs_action_poor": probs_action_poor,
        "heur_reward": heur_reward,
        "reward": reward,
        "bandit_loss": bandit_loss,
        "q": q,
        "debias": debias,
        "kl_loss_approx": kl_loss_approx,
        "smoothed_loss": smoothed_loss,
        "itemized_loss_action": itemized_loss_action,
        "itemized_loss_len_averaged": itemized_loss_len_averaged,
        "kto_loss": kto_loss,
        "kto_z": z * torch.ones_like(reward),
        "kto_logp_ratios": logp_ratios,
        "kto_chosen_losses": chosen_losses,
        "kto_chosen_rewards": chosen_rewards,
        "kto_rejected_losses": rejected_losses,
        "kto_rejected_rewards": rejected_rewards,
        **prefix_dict_keys_with(out, 'extract_')
    })
    return loss


class LookAheadWrapper(DataLoader):
    def __init__(self, dataloader: DataLoader, cache_size: int, enabled: bool):
        self.dataloader = dataloader
        self.cache = []
        self.cache_size = cache_size
        self.enabled = enabled

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self):
        if not self.enabled:
            yield from self.dataloader
            return
        loader = iter(self.dataloader)
        for i in range(len(self.dataloader)):
            if i % self.cache_size == 0:
                self.cache = []
                for _ in range(self.cache_size):
                    self.cache.append(next(loader))
            yield self.cache[i % self.cache_size]


def _route_to_task_type(task_name: str) -> str:
    if task_name.startswith("igl"):
        return "igl"
    return "seq2seq"


class LookAheadBatchCallback(TrainerCallback):
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def on_step_begin(self, args, state, control, model, train_dataloader,
                      **kwargs):
        if not self.enabled:
            return control
        cache_batch = train_dataloader.cache
        assert cache_batch is not None
        lps = []  # kl examples on policy
        ps_ref = []  # kl examples on reference policy
        # remove sft tasks
        cache_batch = [batch for batch in cache_batch if _route_to_task_type(
            batch['task']) == "igl"]

        for batch in cache_batch:
            with torch.inference_mode():
                # access kto_ref from g.policy_lib
                bsz = len(batch['game_turn_id'])
                p_ref = torch.tensor([
                    g.policy_lib.get(batch['game_id'][i], batch['turn_id'][i],
                                     batch["policy_id"][i])for i in range(bsz)])
                ps_ref.append(p_ref)
                # todo: make sure kto is consistent with that stored in policy_lib
                input_ids = batch[KTO_PREFIX + "input_ids"].detach().clone()
                new_batch = dict(
                    task="base",
                    input_ids=batch[KTO_PREFIX + "input_ids"],
                    attention_mask=batch[KTO_PREFIX + "attention_mask"],
                    pixel_values=batch[POLICY_PREFIX + "pixel_values"],
                    pixel_attention_mask=batch[POLICY_PREFIX +
                                               "pixel_attention_mask"]
                )
                outputs = model(**new_batch)
                input_ids = input_ids[..., 1:]
                logits = outputs.logits[..., :-1, :]
                lp, out = g.adapter.extract_policy_action(
                    logits, input_ids, output=True, return_logits=True)
                lps.append(lp.detach())
                del outputs, logits, input_ids
        # (bs * grad_accum - none-igl mini batches,)
        if len(lps) == 0:
            return control
        lps = torch.cat(lps).to('cpu')
        ps_ref = torch.cat(ps_ref).to('cpu')
        logp_ratio = lps - ps_ref.log()
        stats = {
            'pre_probs_actions': lps.exp(),
            'pre_probs_actions_ref': ps_ref,
            'pre_logp_ratio': logp_ratio,
            'pre_kl_batch': logp_ratio.mean().clamp_min(0) * torch.ones_like(logp_ratio),
        }
        for k, v in stats.items():
            if k in g.running_stats:
                assert g.running_stats[k].is_empty_cache()
            track_scalar(g.running_stats, v, k)
        global_step = state.global_step
        batch_size = lps.size(0)
        game_turn_id = [i for b in cache_batch for i in b['game_turn_id']]
        g.record_caches["train"]["lookahead"].batch_record({
            'game_turn_id': game_turn_id,
            'global_step': [global_step] * batch_size,
            **stats,
        })
        clear_all_cache()
        return control
