import logging
import re
from functools import cache
from pathlib import Path
from typing import List, Set, Tuple, TypeVar
import regex
import torch
import torch.nn.functional as F
from PIL import Image

from decoding import complete_re
from utils import (T_MAX_LENGTH, LocalConfig, get_idefics_processor,
                   get_logger, nested_apply, sorted_list, device)

RE_PATTERN = r'^(deselect\s[A-Z](?:\s[A-Z])*(?:\sselect\s[A-Z](?:\s[A-Z])*)?|select\s[A-Z](?:\s[A-Z])*)$'  # noqa


# Name type, newtype of str. e.g. "page4-249.png"
N = TypeVar('N')

ALPHABET = 'ABCDEFGHIJ'  # we only have 10 images
LEGAL_TOKEN_IDS = [2, 315, 330, 334, 365, 382, 384, 401, 413,
                   420, 475, 5339, 634, 17960, 32002]  # A - J and <end_of_utterance> and <\s> and 'select' and 'deselect'

MINI_DECODER = {
    384: 'D', 2: '</s>', 32002: '<end_of_utterance>',
    420: 'G', 17960: 'elect', 330: 'A', 365: 'B', 334: 'C', 5339: 'select',
    401: 'F', 475: 'J', 634: 'des', 315: 'I', 413: 'E', 382: 'H'}

class AlphabeticNameHash:

    @cache
    def __init__(self, context: List[N]) -> None:
        self._forward_map = {im: ALPHABET[i] for i, im in enumerate(context)}
        self._backward_map = {ALPHABET[i]: im for i, im in enumerate(context)}

    def hash(self, im: N) -> str:
        return self._forward_map[im]

    def unhash(self, i: str) -> N:
        return self._backward_map[i]

    def valid_hash(self, i: str) -> bool:
        return i in self._backward_map


class IdeficsAdapter:

    PAD_TOKEN_ID = 0
    LABEL_MASK_ID = 32001  # idefics2: image_token_id
    LEGAL_TOKEN_IDS = LEGAL_TOKEN_IDS
    LEGAL_TOKEN_MASK = torch.zeros(32003, requires_grad=False)\
        .index_fill_(0, torch.tensor(LEGAL_TOKEN_IDS), 1).to(device=device(), dtype=torch.bool)
    SUPPRESS_TOKEN_IDS = list(set(range(32003)) - set(LEGAL_TOKEN_IDS))

    def __init__(self, image_folder: Path = None, checkpoint=None,
                 legal_token_only=False, logger=None, keep_max_turns=None) -> None:
        self.t_max_length = 2048
        self.logger = get_logger(
            __name__, level=logging.DEBUG) if logger is None else logger
        self.image_folder = image_folder if image_folder else Path(
            "data/tangram_pngs")
        self.image_cache = {}
        self.processor = get_idefics_processor(
            checkpoint) if checkpoint else get_idefics_processor()
        self.tokenizer = self.processor.tokenizer
        self.legal_token_only = legal_token_only
        self.keep_max_turns = keep_max_turns

    def get_image(self, im_name: N) -> Image.Image:
        if im_name not in self.image_cache:
            self.image_cache[im_name] = Image.open(
                self.image_folder.joinpath(im_name))
        return self.image_cache[im_name]

    @staticmethod
    def build(local_config: LocalConfig):
        adapter = IdeficsAdapter(Path(local_config.input_dir).joinpath(
            "tangram_pngs"), checkpoint=local_config.checkpoint,
            legal_token_only=local_config.legal_token_only,
            keep_max_turns=local_config.keep_max_turns)
        adapter.t_max_length = T_MAX_LENGTH
        return adapter

    def unhash(self, context: List[N], c: str):
        return AlphabeticNameHash(tuple(context)).unhash(c)

    def valid_hash(self, context: List[N], c: str):
        return AlphabeticNameHash(tuple(context)).valid_hash(c)

    def re_generate(self, model, model_inputs, gen_kwargs, return_tokens=False):
        model_kwargs = {**model_inputs, **gen_kwargs}
        pattern = regex.compile(RE_PATTERN)
        tokenizer = self.tokenizer
        decoder = MINI_DECODER
        max_new_tokens = model_kwargs.pop("max_new_tokens")
        partial_completion = complete_re(
            pattern, tokenizer, decoder, model, max_new_tokens, return_tokens=return_tokens, **model_kwargs)
        return partial_completion

    def parse(self, context: List[N], decoded_out: str, currently_selected: List[N], hash_images=True) -> List[str]:
        assert hash_images, "only supports hashed images"
        h = AlphabeticNameHash(tuple(context))
        self.logger.debug(f"{context=}")
        # do inference
        self.logger.debug(f"{decoded_out=}")
        selection, deselection = self.parse_raw(decoded_out)

        hashed_currently_selected = {h.hash(n) for n in currently_selected}
        desel_to_remove = deselection - hashed_currently_selected
        if len(desel_to_remove) > 0:
            self.logger.debug(f"warn! {desel_to_remove=}")
            deselection = deselection - desel_to_remove

        sel_to_remove = selection & hashed_currently_selected
        if len(sel_to_remove) > 0:
            self.logger.debug(f"warn! {sel_to_remove=}")
            selection = selection - sel_to_remove

        self.logger.debug("post strict cleaning")
        self.logger.debug(f"{selection=}")
        self.logger.debug(f"{deselection=}")

        model_clicks = selection | deselection
        self.logger.debug(f"{model_clicks=}")
        model_clicks_png = [h.unhash(n)
                            for n in model_clicks if h.valid_hash(n)]
        self.logger.debug(f"{model_clicks_png=}")
        return model_clicks_png

    @staticmethod
    def parse_raw(text: str) -> Tuple[Set[N], Set[N]]:
        last_answer = text.strip()
        if ":" in text:
            last_answer_pattern = r":.*$"
            xs = re.findall(last_answer_pattern, text)
            last_answer = xs[0].removeprefix(":").strip()
        xs = re.search(RE_PATTERN, last_answer)
        if xs is None:
            print(f"{last_answer=}")
            print("did not pass regex")
            return set(), set()

        select_pattern = r"(?<!de)select( [A-J])+$"
        xs = re.search(select_pattern, last_answer)
        if xs is not None:
            xs = xs.group()
        selections = set(xs.split(" ")[1:]) if xs else set()

        deselect_pattern = r"^deselect( [A-J])+"
        xs = re.search(deselect_pattern, last_answer)
        if xs is not None:
            xs = xs.group()
        deselections = set(xs.split(" ")[1:]) if xs else set()

        return selections, deselections

    def compose(self, context, chats, previous_selected, hash_images, padding):
        select_accum, deselect_accum, clickss = self.unfold_select_deselect(
            previous_selected)

        select_accum = select_accum + [[]]
        deselect_accum = deselect_accum + [[]]
        previous_selected = [[]] + previous_selected  # old states pre click
        assert len(chats) == len(select_accum) == len(
            deselect_accum) == len(previous_selected)

        messages, images = self.build_processor_input(
            context, chats, select_accum, deselect_accum, previous_selected, hash_images, omit_last_answer=True, sort_names=True, omit_context=False, chat_feedback=None)
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True)
        prompt = prompt.strip()
        self.logger.debug(prompt)
        # Keep consistent with base
        inputs = self.processor(
            text=prompt, images=images,
            padding=padding, truncation=True, max_length=self.t_max_length,
            return_tensors="pt")
        return inputs

    def build_processor_input(self, image_pngs: List[N], chats: List[str],
                              select_accum: List[List[N]],
                              deselect_accum: List[List[N]],
                              pre_click_selected_accum: List[List[N]],
                              hash_image: bool, omit_last_answer: bool, sort_names: bool, omit_context: bool, chat_feedback: str, ):
        def _text_content(text): return {"type": "text", "text": text}

        def _image_content(): return {"type": "image"}

        def _user_prompt(content): return {"role": "user", "content": content}

        def _assistant_prompt(content): return {
            "role": "assistant", "content": content}

        def _system_prompt(content): return {
            "role": "system", "content": content}

        def _current_state(selected: List[N]):
            if len(selected) == 0:
                return 'none is selected'
            return f'{" ".join(selected)} currently selected'

        def _listener_action(select: List[N], deselect: List[N]):
            if len(select) == 0 and len(deselect) == 0:
                return 'nothing'
            if len(select) == 0:
                return f'deselect {" ".join(deselect)}'
            if len(deselect) == 0:
                return f'select {" ".join(select)}'
            return f'deselect {" ".join(deselect)} select {" ".join(select)}'

        func = AlphabeticNameHash(tuple(image_pngs)).hash if hash_image else id
        context, select_accum, deselect_accum,  pre_click_selected_accum = nested_apply(
            func, (image_pngs, select_accum, deselect_accum, pre_click_selected_accum))

        prompt = []
        images = []
        if not omit_context:
            images = [self.get_image(im) for im in image_pngs]
            images_and_names_content = []
            for im_name in context:
                images_and_names_content.append(_image_content())
                images_and_names_content.append(_text_content(im_name))
            prompt.append(_system_prompt(images_and_names_content))
        if not len(chats) == len(select_accum) == len(deselect_accum) == len(pre_click_selected_accum):
            self.logger.error(f"{chats=}")
            self.logger.error(f"{select_accum=}")
            self.logger.error(f"{deselect_accum=}")
            self.logger.error(f"{pre_click_selected_accum=}")
            assert False
        total_turns = len(chats)
        for i, (chat, select, deselect, pre_click_selected) in enumerate(
            zip(chats, select_accum, deselect_accum, pre_click_selected_accum)):
            if self.keep_max_turns and i + self.keep_max_turns < total_turns:
                continue
            if sort_names:
                select = sorted(select)
                deselect = sorted(deselect)
                pre_click_selected = sorted(pre_click_selected)

            prompt.append(_system_prompt(
                [_text_content(_current_state(pre_click_selected))]))
            prompt.append(_user_prompt([_text_content(chat)]))
            prompt.append(_assistant_prompt(
                [_text_content(_listener_action(select, deselect))]))
        if omit_last_answer:
            prompt.pop(-1)
        if chat_feedback is not None:
            prompt.append(_user_prompt([_text_content(chat_feedback)]))
        return prompt, images

    def extract_policy_action(self, logits: torch.Tensor, collapsed_ids: torch.Tensor = None, from_generation=False, output=False, return_logits=False):
        """
        logits: what the model outputs batch_size x t_max_length x vocab
        collapsed_ids: referenced a_i. batch_size x t_max_length. optional
        return batch_size(1)
        """
        assert logits.ndim == 3
        if self.legal_token_only:
            logits.masked_fill_(~self.LEGAL_TOKEN_MASK, -torch.inf)
        batch_size = logits.shape[0]
        # collapsed_ids batch_size x t_max_length
        if collapsed_ids is None:
            collapsed_ids = torch.argmax(logits, dim=-1)
        else:
            assert collapsed_ids.shape == (batch_size, logits.shape[1])
        action_logits = torch.zeros(batch_size)
        normalized_logits = F.log_softmax(logits, dim=-1)

        if output:
            out = {
                "target_ids": [],
                "target_logits": [],
                "gen_range": []
            }

        # gather?
        for b in range(batch_size):
            if from_generation:
                i = 0  # during deployment, the first token is the one generated
            else:
                i = self.start_index_of_last_answer(collapsed_ids[b])
            j = self.end_index_of_last_answer(collapsed_ids[b])
            if not i < j <= logits.shape[1]:
                action_logits[b] = - torch.inf
                self.logger.warning(
                    "ill formatted output, annealing action probs")
                self.logger.warning(f"{(i, j)=}")
                self.logger.warning(f"{collapsed_ids[b, i:]=}")
                self.logger.warning(
                    f"{self.tokenizer.decode(collapsed_ids[b, i:])=}")
                if output:
                    out["target_ids"].append(collapsed_ids[b, i:].tolist())
                    out["target_logits"].append(None)
                    out["gen_range"].append([i, -1, j])
                continue
            gen_range = torch.arange(i, j)
            generated_logits = normalized_logits[b, gen_range,
                                                 collapsed_ids[b, gen_range]]
            action_logits[b] = generated_logits.sum()
            if output:
                out["target_ids"].append(collapsed_ids[b, i:j].tolist())
                out["target_logits"].append(generated_logits.tolist())
                out["gen_range"].append(gen_range.tolist())
        action_probs = action_logits.exp()
        if output:
            out['action_logits'] = action_logits.tolist()
            out['action_probs'] = action_probs.tolist()
            if return_logits:
                return action_logits, out
            return action_probs, out
        return action_logits if return_logits else action_probs

    def end_index_of_last_answer(self, input_ids: torch.Tensor) -> int:
        """ similar to start_index_of_last_answer
        operates on token level """
        assert input_ids.ndim == 1
        # note: idefics2 adapter.tokenizer.additional_special_tokens_ids[adapter.tokenizer.additional_special_tokens.index("<end_of_utterance>")], 32002
        pivot_token_id = 32002
        # assume input_ids has to end with <end_of_utterance>, or RuntimeError
        pivot_index = (input_ids == pivot_token_id).nonzero().max().item()
        return pivot_index + 1

    def start_index_of_last_answer(self, input_ids: torch.Tensor) -> int:
        """ input_ids has shape (t_max_length, )"""
        assert input_ids.ndim == 1
        if input_ids[0] == self.LABEL_MASK_ID:
            # extracting from labels. thye typically look like:
            # [32001, 32001, ..., 32001, 330, 32002] --> decoded as "<image> A<end_of_utterance>"
            # if no clicks / illegal clicks were made from free form generation
            # [32001, 32001, ..., 32001, 28705, 32002] -->
            # decoded as "<image> <end_of_utterance>"
            pivot_token_id = self.LABEL_MASK_ID
        else:
            # : # self.tokenizer("Assistant: ")
            pivot_token_id = 28747
        pivot_index = (input_ids == pivot_token_id).nonzero().max().item()
        return pivot_index + 1

    def unfold_select_deselect(self, previous_selected: List[List[N]]) -> Tuple[List[N], List[N], List[N]]:
        # currently selected AFTER i-th turn
        num_turns = len(previous_selected)
        selected: List[List[str]] = []  # turn-wise selection
        deselected: List[List[str]] = []  # turn-wise deselection
        clicks: List[List[str]] = []
        # combining turn-wise newly selected and newly deselected
        prev_selected = set()
        for turn in range(num_turns):
            curr_selected = set(previous_selected[turn])
            newly_selected = curr_selected - prev_selected
            newly_deselected = prev_selected - curr_selected
            selected.append(sorted_list(newly_selected))
            deselected.append(sorted_list(newly_deselected))
            clicks.append(sorted_list(newly_selected | newly_deselected))
            prev_selected = curr_selected.copy()
        return selected, deselected, clicks
