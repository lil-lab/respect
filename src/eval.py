"""
python src/eval.py --gin_file src/config/multitask.gin --gin_param 'XTrainingArguments.task_configs={"base_ood": @base_ood/TaskConfig()}' --gin_param TaskConfig.seq2seq_recover_generation_probs=False --gin_param TaskConfig.seq2seq_recover_h_probs=False --gin_param 'LocalConfig.checkpoint_path="dt_results/4353471/checkpoint-180"'
"""

import os
from pprint import pprint

import gin
from absl import app, logging
from datasets.utils.logging import set_verbosity_error

import globals
import wandb
from adapter_idefics import IdeficsAdapter
from config.gin_template import parse_config
from io_utils import RecordCache, RecordCacheCallback
from multitask import AutoTask, MultiTaskModel, MultiTaskTrainer
from running_stats import BatchLossCallback
from transformers.trainer_utils import set_seed
from utils import (SLURM_JOB_ID, XTrainingArguments, exist_run_id,
                   load_idefics_model, read_run_id)

def main(_):
    logging.info("house keeping")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_PROJECT"] = "multiref_multitask"
    set_verbosity_error()
    # Issue
    gin.bind_parameter('XTrainingArguments.run_name', str(SLURM_JOB_ID))
    parse_config()
    xargs = XTrainingArguments()

    set_seed(xargs.seed)

    logging.info("load adapter")
    adapter = IdeficsAdapter.build(xargs.local_config)

    logging.info("load datasets")
    task_configs = [c for c in xargs.task_configs.values()
                    if c.train_or_eval == "eval"]
    # only Seq2Seq tasks are allowed
    assert all(c.task == "Seq2SeqTask" for c in task_configs)
    tasks = [AutoTask.build(c, adapter) for c in task_configs]

    if (not xargs.local_config.debug) and exist_run_id(xargs.output_dir) and xargs.local_config.eval_overwrite:
        # https: // github.com/huggingface/transformers/issues/25032  # issuecomment-1648022501
        wandb.init(project=os.environ["WANDB_PROJECT"],
                   resume="must", id=read_run_id(xargs.output_dir))
    else:
        wandb.disabled = True

    logging.info("load model")
    model = load_idefics_model(
        os.path.expanduser(xargs.local_config.checkpoint),
        os.path.expanduser(xargs.local_config.checkpoint_path),
        is_quantized=(xargs.bnb_config is not None), is_trainable=False,
        bnb_config=xargs.bnb_config, lora_config=xargs.lora_config,
        attn_implementation=xargs.local_config.attn_implementation,)

    trainer = MultiTaskTrainer(
        tasks,
        model=MultiTaskModel(model, tasks),
        args=xargs,
        tokenizer=adapter.tokenizer,
    )

    globals.adapter = adapter
    globals.running_stats = dict()
    globals.record_caches = {
        "eval": {
            task.name: RecordCache() for task in tasks if not task.train_only
        },
    }

    trainer.add_callback(BatchLossCallback())
    trainer.add_callback(RecordCacheCallback())

    # back door for multitask compute_metrics
    adapter.policy_model = model
    adapter.trainer = trainer
    metrics = trainer.evaluate()
    pprint(metrics)


if __name__ == "__main__":
    app.run(main)
