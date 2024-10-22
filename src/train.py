
import os

import datasets
import gin
from absl import app, logging

import globals
import wandb
from adapter_idefics import IdeficsAdapter
from config.gin_template import parse_config
from dataset.policy_lib import PolicyLib
from igl import LookAheadBatchCallback
from io_utils import RecordCache, RecordCacheCallback
from multitask import AutoTask, MultiTaskModel, MultiTaskTrainer
from reward_decoder_lib import RewardDecoderLib
from running_stats import BatchLossCallback
from transformers.trainer_utils import set_seed
from utils import (SLURM_JOB_ID, BestMetricsCallback, OverflowGradientCallback,
                   XTrainingArguments, exist_checkpoint, exist_run_id,
                   load_idefics_model, read_run_id, save_game_turn_ids,
                   save_run_id, track_extra_wandb_config)


def main(_):
    logging.info("house keeping")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.disable_progress_bar()
    datasets.disable_caching()
    datasets.utils.logging.set_verbosity_error()
    gin.bind_parameter('XTrainingArguments.run_name',
                       str(SLURM_JOB_ID))
    parse_config()
    xargs = XTrainingArguments()
    os.environ["WANDB_PROJECT"] = xargs.local_config.project_name
    wandb_path = os.path.join(xargs.output_dir, "wandb")
    os.environ["WANDB_DIR"] = wandb_path
    os.makedirs(wandb_path, exist_ok=True)

    set_seed(xargs.seed)

    logging.info("load adapter")
    adapter = IdeficsAdapter.build(xargs.local_config)
    globals.adapter = adapter

    logging.info("load reward decoder lib")
    reward_decoder_lib = RewardDecoderLib(
        [xargs.igl_config.prompt_id],)
    reward_decoder_lib.load(os.path.join(
        xargs.local_config.input_dir, "reward_decoder_outputs"))
    globals.reward_decoder_lib = reward_decoder_lib

    logging.info("load policy lib")
    policy_lib = PolicyLib()
    policy_lib.load(os.path.join(
        xargs.local_config.input_dir, "policy_lib", "sc"))
    globals.policy_lib = policy_lib

    logging.info("load datasets")
    tasks = [AutoTask.build(task_config, adapter)
             for task_config in xargs.task_configs.values()]

    if (not xargs.local_config.debug) and exist_run_id(xargs.output_dir):
        # https: // github.com/huggingface/transformers/issues/25032  # issuecomment-1648022501
        wandb.init(project=os.environ["WANDB_PROJECT"],
                   resume="must", id=read_run_id(xargs.output_dir))

    logging.info("load model")
    model = load_idefics_model(
        os.path.expanduser(xargs.local_config.checkpoint),
        os.path.expanduser(xargs.local_config.checkpoint_path),
        is_quantized=(xargs.bnb_config is not None), is_trainable=True,
        bnb_config=xargs.bnb_config, lora_config=xargs.lora_config,
        attn_implementation=xargs.local_config.attn_implementation,)

    trainer = MultiTaskTrainer(
        tasks,
        model=MultiTaskModel(model, tasks),
        args=xargs,
        tokenizer=adapter.tokenizer,
    )

    globals.running_stats = dict()
    globals.record_caches = {
        "train": {
            task.name: RecordCache() for task in tasks if not task.eval_only
        },
        "eval": {
            task.name: RecordCache() for task in tasks if not task.train_only
        },
    }
    globals.record_caches["train"]["lookahead"] = RecordCache()

    trainer.add_callback(BatchLossCallback())
    trainer.add_callback(RecordCacheCallback())
    trainer.add_callback(OverflowGradientCallback())
    trainer.add_callback(BestMetricsCallback())
    trainer.add_callback(LookAheadBatchCallback(
        enabled=xargs.multi_task_config.lookahead))

    # back door for multitask compute_metrics
    adapter.policy_model = model
    adapter.trainer = trainer

    trainer.log({"dummy": 0})  # trigger wandb setup in trainer
    assert wandb.run is not None
    save_run_id(wandb.run.id, xargs.output_dir)
    track_extra_wandb_config(xargs, tasks)
    save_game_turn_ids(tasks, xargs.output_dir)

    logging.info("watch for checkpoints")
    if xargs.local_config.resume_from_checkpoint_path:
        # if checkpoint path does not have optimizer or scheduler .pth
        # then it only loads model weights
        resume_from_checkpoint = os.path.expanduser(
            xargs.local_config.resume_from_checkpoint_path)
    else:
        resume_from_checkpoint = exist_checkpoint(xargs.output_dir)
    logging.info(f"{resume_from_checkpoint=}")

    if (not xargs.local_config.debug) and resume_from_checkpoint is False:
        trainer.evaluate()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.evaluate()
    trainer.save_model()


if __name__ == "__main__":
    app.run(main)
