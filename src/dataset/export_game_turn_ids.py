"""
python src/dataset/export_game_turn_ids.py --dataset data/apr_23_dev --shuffle --num_game 25 --save_as data/special_game_turn_ids/apr_23_dev.txt
"""

import random

import datasets
import pandas as pd
from absl import app, flags, logging

flags.DEFINE_string('dataset', 'data/apr_23_dev', 'dataset')
flags.DEFINE_string('save_as', None,
                    'Path to save the aggregated reward decoder outputs', required=True)
flags.DEFINE_bool('shuffle', False, 'shuffle')
flags.DEFINE_integer('seed', 42, 'seed')
flags.DEFINE_integer('num_game', None, 'num')

FLAGS = flags.FLAGS


def main(_):
    random.seed(FLAGS.seed)
    dataset = datasets.load_from_disk(FLAGS.dataset)
    df = dataset.to_pandas()
    game_id = df['game_id'].unique().tolist()
    logging.info(f'Number of game ids: {len(game_id)}')
    if FLAGS.shuffle:
        random.shuffle(game_id)
    if FLAGS.num_game:
        game_id = game_id[:FLAGS.num_game]

    df_filtered = df[df.game_id.isin(game_id)]['game_turn_id']
    df_filtered.to_csv(FLAGS.save_as, index=False, header=False)
    logging.info(
        f'{len(df_filtered)} game_turn_id from {len(game_id)} games saved to {FLAGS.save_as}')


if __name__ == '__main__':
    app.run(main)
