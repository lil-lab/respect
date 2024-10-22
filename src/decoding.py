# inspired by https://github.com/r2d4/rellm

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Union
from typing import Dict, Union, List

import regex
import torch

from transformers import LogitsProcessor, PreTrainedModel, PreTrainedTokenizer

from absl import logging


class LogitsMask(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.ones_like(scores) * (- torch.inf)
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0
        scores = scores + mask
        return scores

# # MINI_DECODER = {
# #     384: 'D', 2: '</s>', 32002: '<end_of_utterance>', 420: 'G', 17960: 'elect',
# #     330: 'A', 365: 'B', 334: 'C', 5339: 'select', 401: 'F', 475: 'J',
# #     634: 'des', 315: 'I', 413: 'E', 382: 'H'}


class ReTokenFilter:
    def __init__(self, decoder: Dict):
        self.decoder = decoder

    def is_valid_token(self, token_id: int, partial_completion: str, patterns: List[regex.Pattern]) -> bool:
        if partial_completion == "":
            return token_id in (5339, 634)  # select, des
        if partial_completion == "des":  # deselect always occurs before select
            return token_id == 17960  # elect
        decoded_token = self.decoder[token_id]
        if partial_completion.split(" ")[-1] in ("select", "deselect"):
            return decoded_token in "ABCDEFGHIJ"  # can't be verbs or eos
        new_partial = partial_completion
        if decoded_token not in ("</s>", "<end_of_utterance>"):
            new_partial += " " + decoded_token
            logging.debug(f"{new_partial=}")
        return any(pattern.fullmatch(new_partial, partial=True) for pattern in patterns) and decoded_token not in partial_completion.split(" ")

    def filter_tokens(self, partial_completion: str, patterns: Union[regex.Pattern, List[regex.Pattern]]) -> Set[int]:
        if isinstance(patterns, regex.Pattern):
            patterns = [patterns]

        with ThreadPoolExecutor():
            valid_token_ids = set(
                filter(
                    lambda token_id: self.is_valid_token(
                        token_id, partial_completion, patterns),
                    self.decoder.keys()
                )
            )
        return valid_token_ids


def complete_re(pattern: Union[regex.Pattern, List[regex.Pattern]],
                tokenizer: PreTrainedTokenizer,
                decoder: Dict[int, str],  # a mini tokenizer
                model: PreTrainedModel,
                max_new_tokens: int = 10,
                return_tokens: bool = False,
                **model_kwargs):

    if isinstance(pattern, regex.Pattern):
        pattern = [pattern]

    gen_tokens = 0
    partial_completion = ""

    assert model_kwargs['input_ids'].shape[0] == 1
    input_ids = model_kwargs.pop('input_ids')  # running variable
    original_input_ids = input_ids.detach().clone()
    attention_mask = model_kwargs.pop('attention_mask')
    output_ids = None

    prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    prompt_plus_completion = prompt + partial_completion

    token_filter = ReTokenFilter(decoder)
    while gen_tokens < max_new_tokens:
        prompt_length = input_ids.shape[1]

        allowed_token_ids = token_filter.filter_tokens(
            partial_completion, pattern)
        logging.debug(f"step={gen_tokens} {allowed_token_ids=}")
        mask_processor = LogitsMask(allowed_token_ids)

        output_ids = model.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_new_tokens=1,
                                    pad_token_id=tokenizer.eos_token_id,
                                    logits_processor=[mask_processor,],
                                    **model_kwargs)
        if output_ids.shape[1] == prompt_length:
            logging.debug("no new tokens generated, stopping")
            break
        prompt_plus_completion = tokenizer.decode(
            output_ids[0], skip_special_tokens=True)
        partial_completion = prompt_plus_completion[len(prompt)+1:]  # space
        logging.debug(f"step={gen_tokens} completion={partial_completion}")
        new_token_id = output_ids[:, prompt_length:].item()
        gen_tokens += 1
        if new_token_id in (2, 32002):  # idefics2 EOS token
            logging.debug(f"EOS token {new_token_id} generated, stopping")
            break
        input_ids = output_ids
        attention_mask = torch.nn.functional.pad(
            attention_mask, (0, 1), value=1)
    if gen_tokens == max_new_tokens and output_ids[:, -1] not in (2, 32002):
        logging.warning(
            "max_new_tokens reached without <end_of_utterance>, illegal")
        partial_completion = ""
        output_ids = original_input_ids
    logging.debug(f"{partial_completion=}")
    if return_tokens:
        return output_ids
    return partial_completion
