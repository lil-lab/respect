import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from absl import logging

from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
import wandb
import globals as g


def write_df_to_csv(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if Path(path).exists():
        logging.warning(f"overwriting {path}")
    df.index.rename('index', inplace=True)
    df.to_csv(path)
    logging.info(f"saved to {path}")


def batch_trim_left(array, token_id):
    # assume left padded
    i = np.argmax(array != token_id, axis=1).min()
    return array[:, i:]


class EvalPredictionIO:

    @staticmethod
    def save_csv(path, eval_prediction: EvalPrediction):
        preds = eval_prediction.predictions
        preds = batch_trim_left(preds, 0)  # Adapter.PAD_TOKEN_ID
        labels = eval_prediction.label_ids
        labels = batch_trim_left(labels, 32001)  # Adapter.LABEL_MASK_ID
        df = pd.DataFrame({
            "preds": preds.tolist(),
            "labels": labels.tolist(),
        }, index=range(len(preds)))
        write_df_to_csv(df, path)

    @staticmethod
    def load_csv(path):
        df = pd.read_csv(path)
        preds = np.array(df["preds"].apply(eval).tolist())
        labels = np.array(df["labels"].apply(eval).tolist())
        return EvalPrediction(predictions=preds, label_ids=labels)

    @staticmethod
    def save_npz(path, eval_prediction: EvalPrediction):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if Path(path).exists():
            logging.warning(f"overwriting {path}")
        np.savez_compressed(path, preds=eval_prediction.predictions,
                            labels=eval_prediction.label_ids)
        logging.info(f"saved to {path}")

    @staticmethod
    def load_npz(path):
        data = np.load(path)
        return EvalPrediction(predictions=data["preds"], label_ids=data["labels"])


class RecordCache:

    def __init__(self):
        self._schema = None
        self._cache = []

    def is_empty(self):
        return len(self._cache) == 0

    def _check(self, record):
        assert "game_turn_id" in record, "game_turn_id is required"
        assert set(record.keys()) == self._schema

    @staticmethod
    def serialize(list_like):
        if isinstance(list_like, np.ndarray):
            return list_like.tolist()
        if isinstance(list_like, torch.Tensor):
            return list_like.detach().cpu().numpy().tolist()
        if isinstance(list_like[0], (set, dict)):
            return [str(i) for i in list_like]
        if isinstance(list_like, list):
            return list_like
        raise ValueError(f"unsupported type {type(list_like)}")

    def single_record(self, record: Dict[str, List]):
        """cache {'a': 1, 'b': 2}"""
        if self._schema is None:
            self._schema = set(k for k in record.keys())
        self._check(record)
        self._cache.append(record)

    def batch_record(self, records: Dict[str, List]):
        """cache {'a': [1,2], 'b': [3,4]}"""
        records = {k: self.serialize(v) for k, v in records.items()}
        batch_size = len(list(records.values())[0])
        assert all(len(v) == batch_size for v in records.values())
        for i in range(batch_size):
            self.single_record({k: v[i] for k, v in records.items()})

    def to_df(self):
        return pd.DataFrame(self._cache)

    def clear_cache(self):
        self._cache = []
        self._schema = None


class RecordCacheCallback(TrainerCallback):
    """save itemized base and igl train signal on epoch end and evaluate end"""

    @staticmethod
    def _write(record_cache: RecordCache, path: str, global_step, task):
        if record_cache is None or record_cache.is_empty():
            return
        df = record_cache.to_df()
        write_df_to_csv(df, path)
        record_cache.clear_cache()  # reaccumulate for another epoch / eval
        if wandb.run is None:
            logging.warning("wandb is not initialized")
            return
        prefix = "eval" if "eval" in task else "train"

        artifact = {
            "train/global_step": global_step,
            f"{prefix}/itemized_{task}": wandb.Table(dataframe=df.dropna()),
        }
        wandb.log(artifact, commit=False)

    def on_epoch_end(self, args, state, control, **kwargs):
        caches = g.record_caches['train']
        for task, cache in caches.items():
            csv_path = os.path.join(
                args.output_dir, "itemized", f"{task}_train", f"global_step_{state.global_step}_epoch_{round(state.epoch)}_task_{task}.csv")
            RecordCacheCallback._write(
                cache, csv_path, state.global_step, f"{task}_train")

    def on_evaluate(self, args, state, control, **kwargs):
        caches = g.record_caches['eval']
        for task, cache in caches.items():
            epoch = round(state.epoch) if state.epoch else 0
            csv_path = os.path.join(
                args.output_dir, "itemized", f"{task}_eval", f"global_step_{state.global_step}_epoch_{epoch}_task_{task}.csv")
            RecordCacheCallback._write(
                cache, csv_path, state.global_step, f"{task}_eval")
