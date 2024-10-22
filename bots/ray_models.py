# File: ray_models
# ----------------
# Ray Actor wrappers around models

import logging
from pathlib import Path

import ray
import torch
from src.adapter_idefics import IdeficsAdapter
from src.utils import device, get_logger, load_idefics_model, nested_apply, sorted_list
import torch.amp


def nested_replace(s, _from, _to):
    # s is one of str, tuple or list. _from and _to are both str
    func = lambda x: x.replace(_from, _to)
    return nested_apply(func, s)


def nested_to_device(s):
    # s is either a tensor or a dictionary
    if isinstance(s, torch.Tensor):
        return s.to(device())
    return {k: v.to(device()) for k, v in s.items()}


# CLASSES


@ray.remote(num_gpus=0.5)
class RayIdefics:
    def __init__(self, config):
        super().__init__()
        self.image_folder = Path("../data/tangram_pngs")
        self.logger = get_logger("ray.application", level=logging.INFO)
        self.logger.info(f"loading model from config {config}")
        self.hash_images = True

        model = load_idefics_model(
            config["checkpoint"],
            config["checkpoint_adapter_path"],
            is_trainable=False,
            bnb_config=None,
            is_quantized=True,
            revision=config.get("revision", None),
        )
        model.eval()
        model.to(device())
        self.model = model
        self.logger.info(f"initialized RayIdefics: on device {device()}")

        self.adapter = IdeficsAdapter(self.image_folder,
                                      config["checkpoint"],
                                      legal_token_only=False,
                                      logger=self.logger,)
        self.adapter.t_max_length = 2048
        self.padding = False
        self.constrained_decoding = config.get("constrained_decoding", False)

        assert config["do_sample"] in (True, False)
        assert not config["do_sample"] or config["temperature"] > 0

        self.gen_kwargs = {
            "max_new_tokens": 10,
            "do_sample": True,
            "temperature": 1.0,
            "output_logits": True,
            "return_dict_in_generate": True,
            "remove_invalid_values": True,  # just to be safe
            "renormalize_logits": True,
            "suppress_tokens": IdeficsAdapter.SUPPRESS_TOKEN_IDS
        }
        if self.constrained_decoding:
            self.gen_kwargs.pop("output_logits")
            self.gen_kwargs.pop("return_dict_in_generate")

        self.logger.info("done initializing ray Idefics worker")

    async def predict(self, image_paths, chats, previous_selected):
        image_paths = [n["path"] for n in image_paths]
        image_paths, previous_selected = nested_replace(
            (image_paths, previous_selected), "svg", "png"
        )

        currently_selected = previous_selected[-1] if len(previous_selected) > 0 else []

        model_input = self.adapter.compose(
            image_paths, chats, previous_selected, self.hash_images, self.padding
        )
        model_input = nested_to_device(model_input)
        model_output = None
        if self.constrained_decoding:
            with torch.inference_mode(), torch.autocast(
                device_type=device().type, dtype=torch.bfloat16
            ):
                synced_gpus = model_input.pop("synced_gpus", None)
                model_input["synced_gpus"] = synced_gpus
                decoded_out = self.adapter.re_generate(
                    self.model, model_input, self.gen_kwargs, return_tokens=False)
        else:
            with torch.inference_mode(), torch.autocast(
                device_type=device().type, dtype=torch.bfloat16
            ):
                model_output = self.model.generate(**model_input, **self.gen_kwargs)
            decoded_out = self.adapter.tokenizer.decode(
                model_output.sequences[0], skip_special_tokens=True)
        model_selection = self.adapter.parse(
            image_paths, decoded_out, currently_selected, self.hash_images
        )

        if len(model_selection) == 0:
            self.logger.warning("empty clicks by model")
            model_selection = [image_paths[0]]
            self.logger.debug(f"{image_paths=}")
            self.logger.debug(f"selecting {model_selection}")
            prob = -1
        elif self.constrained_decoding:
            prob = -3
        else:
            logits = torch.stack(model_output.logits, dim=-2)[[0]]
            generated_seqs = model_output.sequences[[0], -logits.shape[1] :]
            assert logits.shape[1] == generated_seqs.shape[1]
            try:
                prob = self.adapter.extract_policy_action(
                    logits, generated_seqs, from_generation=True
                ).item()
            except RuntimeError as e:
                self.logger.error(f"Runtime error in extract_policy_action: {e}")
                prob = -2
            self.logger.debug(f"{prob=}")

        curr_selected = previous_selected[-1] if len(previous_selected) > 0 else []
        new_selected = self.apply_clicks_to_current_selected(
            model_selection, curr_selected
        )
        self.logger.debug(f"{new_selected=}")

        new_selected = nested_replace(new_selected, "png", "svg")
        return {"path": new_selected, "prob": prob, "decoded_out": decoded_out}

    def apply_clicks_to_current_selected(self, model_clicks, curr_selected):
        # apply model selection/deselection to what is currently selected
        curr_selected = set(curr_selected)
        new_selected = curr_selected.copy()
        for c in model_clicks:
            if c in curr_selected:
                new_selected.remove(c)
            else:
                new_selected.add(c)
        new_selected = sorted_list(new_selected)
        return new_selected
