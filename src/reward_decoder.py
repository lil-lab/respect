"""
python src/reward_decoder.py --dataset data/aug121314_bp --prompt_name 02_one_history_binary
"""

import json
import os
import pprint
from functools import partial
from pathlib import Path

import datasets
import evaluate
import pandas as pd
import torch
from absl import app, flags

from src.adapter_idefics import IdeficsAdapter
from transformers import (Idefics2ForConditionalGeneration, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

flags.DEFINE_string('dataset', 'data/may2829jun4', 'Directory of the dataset')
flags.DEFINE_integer('num', -1, 'Number of examples to evaluate, -1 for all')
flags.DEFINE_string('prompt_name', None,
                    'Name of the prompt / suffix', required=True)

FLAGS = flags.FLAGS
REWARD_DECODER_OUTPUTS_DIR = Path('data/reward_decoder_outputs')
PROMPT_NAMES = [
    '02_one_history_binary',
    '03_one_history_trinary'
]



def reward_heuristics_trimodal_ground_truth(example_batch, default_reward_value):
    batch_size = len(example_batch["chat_feedback"])
    rewards = [-1] * batch_size
    available = [True] * batch_size
    for b in range(batch_size):
        if example_batch["is_good_select"][b] and \
                len(example_batch["deselected"][b]) == 0:
            # pure gt good select
            rewards[b] = 1
        elif example_batch["is_good_deselect"][b] and \
                len(example_batch["selected"][b]) == 0:
            # pure gt good deselect
            rewards[b] = 1
        elif example_batch["is_good_select"][b] and \
                example_batch["is_good_deselect"][b]:
            # both gt good select and good deselect
            rewards[b] = 1
        elif example_batch["is_good_select"][b] and \
                len(example_batch["deselected"][b]) > 0:
            rewards[b] = default_reward_value
        elif example_batch["is_good_deselect"][b] and \
                len(example_batch["selected"][b]) > 0:
            rewards[b] = default_reward_value
        else:
            rewards[b] = -1
    return {"reward": torch.tensor(rewards, dtype=torch.float),
            "available": torch.tensor(available, dtype=torch.float)}



def build_processor_input(example, adapter: IdeficsAdapter):
    def _user_prompt(text): return {"role": "user", "content": [{
        "type": "text", "text": text}]}

    full_trajectory, _ = adapter.build_processor_input(
        example['context'], example["chats"],
        select_accum=example["select_accum"],
        deselect_accum=example["deselect_accum"],
        pre_click_selected_accum=example["pre_click_selected_accum"],
        hash_image=True, omit_last_answer=False,
        sort_names=True, omit_context=True, chat_feedback=example['chat_feedback'])

    trajectory = [t for t in full_trajectory if t["role"] != "system"]
    trajectory = trajectory[-4:]

    trajectory_text = adapter.processor.apply_chat_template(
        trajectory, add_generation_prompt=False)

    trajectory_text = trajectory_text.replace(
        "User", "\tSpeaker").replace("Assistant", "\tListener")

    if FLAGS.prompt_name == '02_one_history_binary':
        messages = [
            _user_prompt("Please carefully read the following conversation and answer: Is the very last utterance from the speaker a positive or negative feedback? Often negative feedbacks include corrections and keywords like no, not, undo, don't, with generally negative sentiment, while positive feedbacks often include good, yes, correct, okay, or simply move on to the next stage. Lean towards negative if it sounds neutral.\n(start of the conversation)\n" +
                         trajectory_text + "(end of the conversation)\nAnswer a single word, Positive, or Negative."),
        ]
    elif FLAGS.prompt_name == '03_one_history_trinary':
        messages = [_user_prompt("Please carefully read the following conversation and answer: Is the very last utterance from the speaker a positive, neutral, or negative feedback? Often negative feedbacks include corrections and keywords like no, not, undo, don't, with generally negative sentiment, while positive feedbacks often include good, yes, correct, okay, or simply move on to the next stage. \n(start of the conversation)\n" +
                                 trajectory_text + "(end of the conversation)\nAnswer a single word, Positive, Neutral or Negative."),
                    ]
    else:
        raise ValueError(f"prompt_name must be one of {PROMPT_NAMES}")
    return messages, []


def transpose_dict_of_lists(data):
    keys = list(data.keys())
    values = list(zip(*data.values()))
    return [dict(zip(keys, value)) for value in values]


def transform(example_batch,
              adapter: IdeficsAdapter):
    example_batch_t = transpose_dict_of_lists(example_batch)
    messages_and_images = [build_processor_input(
        e, adapter) for e in example_batch_t]
    messages, _ = zip(*messages_and_images)
    prompts = adapter.processor.apply_chat_template(
        messages, add_generation_prompt=True)
    prompts = [p.strip() for p in prompts]
    inputs = adapter.processor(text=prompts,
                               padding="max_length", truncation=True,
                               max_length=256, return_tensors="pt")
    return inputs


def run(ds, adapter, trainer):
    # preprocess
    ds2 = ds.map(partial(transform, adapter=adapter), batched=True,
                 batch_size=16, remove_columns=ds.column_names)

    num = FLAGS.num if FLAGS.num > 0 else len(ds)
    dataset_raw = ds.select(range(num))
    dataset = ds2.select(range(num))

    # generate
    predictions = trainer.predict(dataset, max_new_tokens=3)

    # decode
    generated_texts = adapter.processor.batch_decode(
        predictions.predictions, skip_special_tokens=True)

    trimmed_generated_texts = [text.split(
        "\n")[-1] for text in generated_texts]
    print(f"{set(trimmed_generated_texts)=}")
    generated_text_to_reward = {
        'Assistant: Negative.': -1.0,
        'Assistant: Neutral': 0.0,
        'Assistant: Positive.': 1.0,
    }

    predictions = [generated_text_to_reward[text]
                   for text in trimmed_generated_texts]
    labels = dataset_raw['reward_trimodal_ground_truth']
    print(f"{predictions[:5]=}")
    print(f"{labels[:5]=}")

    # pack into dataframe

    game_turn_ids = dataset_raw['game_turn_id']
    df = pd.DataFrame({'predictions': predictions,
                       'labels': labels,
                       'game_turn_id': game_turn_ids,
                       'text': generated_texts})
    df.sort_values(by='game_turn_id', inplace=True)
    return df


def print_and_save(df):
    prompt_name = FLAGS.prompt_name
    dataset_name = FLAGS.dataset.split('/')[-1]
    output_dir = REWARD_DECODER_OUTPUTS_DIR / \
        f'{dataset_name}_len_{len(df)}_{prompt_name}'

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_json(output_dir / 'main.json', orient='records', indent=4)

    # write counts
    out_counts = [
        "== counts == ",
        pd.crosstab(df['predictions'], df['labels']).to_string(),
        "== normalized by row == ",
        pd.crosstab(df['predictions'], df['labels'],
                    normalize='index').to_string(),
        "== normalized by column == ",
        pd.crosstab(df['predictions'], df['labels'],
                    normalize='columns').to_string(),
        "== normalized by all == ",
        pd.crosstab(df['predictions'], df['labels'],
                    normalize='all').to_string(),
        "== summed by row == ",
        df.text.apply(lambda x: x.split('\n')[-1]).value_counts().to_string(),
    ]
    with open(output_dir / 'counts.txt', 'w') as f:
        f.write('\n\n'.join(out_counts))
    print('\n\n'.join(out_counts[2:4] + out_counts[8:10]))

    # write metrics
    predictions = df['predictions'].tolist()
    references = df['labels'].tolist()
    metrics = {
        **evaluate.combine(["f1", "precision", "recall"]).compute(predictions=predictions, references=references, average='weighted'),
        **evaluate.combine(["accuracy",]).compute(predictions=references, references=predictions),
        "positive": (df['predictions'] == 1).sum().item(),
        "neutral": (df['predictions'] == 0).sum().item(),
        "negative": (df['predictions'] == -1).sum().item(),
    }
    print()
    print()
    pprint.pprint(metrics)

    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write(json.dumps(metrics, indent=4))

    df[df.predictions == 1.0]['game_turn_id'].to_csv(
        output_dir / 'pos.txt', index=False, header=False)
    df[df.predictions == 0.0]['game_turn_id'].to_csv(
        output_dir / 'neu.txt', index=False, header=False)
    df[df.predictions == -1.0]['game_turn_id'].to_csv(
        output_dir / 'neg.txt', index=False, header=False)

    df[df.labels == 1.0]['game_turn_id'].to_csv(
        output_dir / 'pos_gt.txt', index=False, header=False)
    df[df.labels == 0.0]['game_turn_id'].to_csv(
        output_dir / 'neu_gt.txt', index=False, header=False)
    df[df.labels == -1.0]['game_turn_id'].to_csv(
        output_dir / 'neg_gt.txt', index=False, header=False)

    print(f"save to {output_dir}")


def gt_reward(x):
    out = reward_heuristics_trimodal_ground_truth(x, default_reward_value=0)
    return {'reward_trimodal_ground_truth': out['reward'], }


def main(_):
    assert FLAGS.prompt_name in PROMPT_NAMES, f"prompt_name must be one of {PROMPT_NAMES}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b", torch_dtype=torch.bfloat16, device_map="auto",
        cache_dir="/scratch/zc478/cache/huggingface",
    ).to('cuda')
    adapter = IdeficsAdapter('data/tangram_pngs')

    ds = datasets.load_from_disk(FLAGS.dataset)
    ds = ds.map(gt_reward, batched=True, batch_size=16)
    ds = ds.shuffle(seed=42)
    ds = ds.filter(lambda x: len(x['chat_feedback']) > 0)

    training_args = Seq2SeqTrainingArguments(
        output_dir="../local_results",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=adapter.processor,
        args=training_args,
    )

    df = run(ds, adapter, trainer)
    print_and_save(df)


if __name__ == '__main__':
    app.run(main)
