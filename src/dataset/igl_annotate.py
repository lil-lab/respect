import sys  # noqa
sys.path.append("./src")  # noqa

from utils import (LocalConfig, device, load_idefics_model, nested_to)
from transformers import PreTrainedModel
from base import idefics_transforms
from policy_lib import PolicyLib
from adapter_idefics import IdeficsAdapter
from absl import app, flags, logging
import torch
import datasets
from pathlib import Path

"""

python src/dataset/igl_annotate.py --checkpoint_path results/1252536-1380 --dataset dec_8 --save_as dec_8_a

python src/dataset/igl_annotate.py --checkpoint_path results/2642336 --dataset full6_10 --save_as full6_10_a --policy_id 2642336 --policy_lib_file_path data/policy_lib/full6_10.csv

python src/dataset/igl_annotate.py --human_ann --dataset dec_6 --save_as dec_6_ann

Annotate human-bot interactions with inputs necessary for IGL (IGL_ANNOTATION_KEYS)

"""


flags.DEFINE_string("checkpoint", default="HuggingFaceM4/idefics2-8b",
                    help="the model type to deployed/base policy model")
flags.DEFINE_string("checkpoint_path", default="",
                    help="the adapter path to deployed/base policy model")
flags.DEFINE_string("dataset", default=None,
                    help="dataset to be annotated", required=True)
flags.DEFINE_string("save_as", default=None,
                    help="annotated dataset for training", required=True)
flags.DEFINE_string("policy_lib_file_path", default="", required=False,
                    help="optional path to store poor policies")
flags.DEFINE_string("policy_id", default="", required=False,
                    help="prob action predicted by policy")
flags.DEFINE_list("include_treatments", default=[],
                  help="include treatment, include all if empty")
FLAGS = flags.FLAGS

BATCH_SIZE = 24


def get_local_config():
    local_config = LocalConfig(
        checkpoint=FLAGS.checkpoint,
        checkpoint_path=FLAGS.checkpoint_path,
        legal_token_only=False,
    )
    return local_config


def annotate_igl(example_batch, policy_model_inference_only: PreTrainedModel, adapter: IdeficsAdapter):

    policy_inputs = example_batch
    nested_to(policy_inputs, device())
    labels = policy_inputs["input_ids"].clone().detach()
    with torch.inference_mode():
        # (bs, seq_len, vocab)
        logits = policy_model_inference_only(**policy_inputs).logits
        shifted_logits = logits[:, :-1, :]  # (bs, seq_len, vocab)
        shifted_labels = labels[:, 1:]  # (bs, seq_len)
        probs_action = adapter.extract_policy_action(
            shifted_logits, shifted_labels)  # (bs,)

    return {
        "prob_action_poor": probs_action,
        "policy_id": [FLAGS.policy_id] * len(probs_action)
    }


def add_prob_action_poor(dataset: datasets.DatasetDict, local_config
                         ) -> datasets.DatasetDict:

    logging.info("load adapter")
    adapter = IdeficsAdapter.build(local_config)

    def train_transform(batch):
        idefics_inputs = idefics_transforms(batch, adapter, eval_mode=False)
        idefics_inputs.pop("labels")
        idefics_inputs.pop("game_turn_id")
        return idefics_inputs

    dataset = dataset.map(train_transform, batched=True,
                          batch_size=BATCH_SIZE, desc="prepare train dataset", remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=dataset.column_names)

    logging.info("load model")
    poor_policy = load_idefics_model(local_config.checkpoint,
                                     local_config.checkpoint_path, is_trainable=False)
    poor_policy.eval()

    def annotate_igl_transform(batch): return annotate_igl(
        batch, poor_policy, adapter)

    dataset = dataset.map(annotate_igl_transform, batched=True,
                          batch_size=BATCH_SIZE,
                          desc="annotate prob_action_poor",
                          remove_columns=dataset.column_names, load_from_cache_file=False)

    return dataset


IGL_ANNOTATION_KEYS = ["prob_action_poor", "policy_id"]


def main(_):
    local_config = get_local_config()
    logging.set_verbosity(logging.INFO)

    assert FLAGS.policy_lib_file_path == "" or FLAGS.policy_id != "", "policy id must be provided if policy_lib_file_path is provided."

    logging.info("load dataset")
    dataset = datasets.load_from_disk(
        Path(local_config.input_dir).joinpath(FLAGS.dataset))
    logging.info(dataset)

    if FLAGS.include_treatments:
        dataset = dataset.filter(
            lambda x: x["treatment"] in FLAGS.include_treatments)
        logging.info(
            f"filtered dataset with treatment {FLAGS.include_treatments}")
        logging.info(dataset)

    raw_columns = dataset.column_names
    additional_columns = []

    if "prob_action_poor" in raw_columns:
        dataset = dataset.remove_columns(["prob_action_poor", "policy_id"])
        raw_columns = dataset.column_names
        logging.info("removed existing prob_action_poor")
    new_dataset = add_prob_action_poor(dataset, local_config)

    dataset = dataset.add_column(
        "prob_action_poor", new_dataset["prob_action_poor"].tolist())
    dataset = dataset.add_column("policy_id", new_dataset["policy_id"])
    additional_columns += IGL_ANNOTATION_KEYS

    logging.info(dataset)
    print(dataset.to_pandas()["prob_action_poor"].describe())

    dataset = dataset.select_columns(raw_columns + additional_columns)
    logging.info(dataset)
    dataset.save_to_disk(Path(local_config.input_dir).joinpath(FLAGS.save_as))
    dataset.to_json(Path(local_config.input_dir).joinpath(
        FLAGS.save_as)/"plain.json")

    if FLAGS.policy_lib_file_path != "":
        poor_lib = PolicyLib()
        _ = dataset.map(lambda x: poor_lib.populate(x, FLAGS.policy_id))
        poor_lib.save(FLAGS.policy_lib_file_path)
    logging.info("done")


if __name__ == "__main__":
    app.run(main)
