
"""
python src/dataset/overwrite_clicks_with_ann.py --ann jun_5_train_and_test --dataset data/jun_5_train_and_test --save_as data/jun_5_train_and_test_ann
"""

from src.utils import (load_human_bot_annotation, sorted_list)
from absl import app, flags, logging
import pandas as pd
import datasets

flags.DEFINE_string("ann", default="dec_6", help="ann")
flags.DEFINE_string("dataset", default="data/dec_6",
                    help="dataset to be annotated")
flags.DEFINE_string("save_as", default="data/dec_6_ann",
                    help="annotated dataset for training")

FLAGS = flags.FLAGS


def replace_clicks_with_human_ann(dataset: datasets.Dataset, dataset_name: str) -> datasets.DatasetDict:
    prev_features = dataset.features
    df = dataset.to_pandas()
    ann_df = load_human_bot_annotation(dataset_name)
    joined_df = pd.merge(ann_df, df,  how='left', left_on=[
        'game_id', 'turn_id'], right_on=['game_id', 'turn_id'])
    dataset = datasets.Dataset.from_pandas(joined_df)

    def replace_last_clicks(sample):
        sample["clicks"][-1] = sample["labels"]
        pre_click_selected = sample["pre_click_selected_accum"][-1]
        select = set(sample['labels']) - set(pre_click_selected)
        deselect = set(sample['labels']) - select
        sample["select_accum"][-1] = sorted_list(select)
        sample["deselect_accum"][-1] = sorted_list(deselect)
        sample['selected'] = sorted_list(select)
        sample['deselected'] = sorted_list(deselect)
        return sample

    dataset = dataset.map(replace_last_clicks, features=prev_features,
                          remove_columns=["labels"])
    # Overwrite is_good_select to True because I am the oracle.
    # This should be enough to pass simple_filter.
    # So I left is_good_deselect, overturn_* outdated.
    dataset = dataset.map(lambda _: {"is_good_select": True, })
    return dataset


def main(_):
    dataset = datasets.load_from_disk(FLAGS.dataset)
    dataset = replace_clicks_with_human_ann(dataset, FLAGS.ann)
    logging.info(dataset)
    dataset.save_to_disk(FLAGS.save_as)
    dataset.to_json(f"{FLAGS.save_as}/plain.json")
    logging.info(f"saved to {FLAGS.save_as}")


if __name__ == "__main__":
    app.run(main)
