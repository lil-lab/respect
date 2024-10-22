"""
python src/dataset/verify_split.py --dataset apr_24 --split train --belongs_split train_and_test
"""

import pickle
from absl import flags, app
import datasets
from pathlib import Path

flags.DEFINE_string("dataset", default=None, required=True, help="")
flags.DEFINE_string("split", default=None, required=True,
                    help="target_split_name")
flags.DEFINE_string("belongs_split", default=None, required=True,
                    help="one of train, test, dev, and train_and_test")
FLAGS = flags.FLAGS


def main(_):
    input_dir = Path("data")
    split_dir = Path("data/dataset_splits")
    dataset_dict = datasets.load_from_disk(input_dir.joinpath(FLAGS.dataset))
    dataset = dataset_dict[FLAGS.split]
    splits = FLAGS.belongs_split.split("_and_")
    print(f"{splits=}")
    all_tangrams = set()
    for split in splits:
        with open(split_dir.joinpath(split + "_imgs.pkl"), "rb") as f:
            all_tangrams.update(pickle.load(f))
    all_tangrams = {s.split('.')[0] for s in all_tangrams}
    print(f"{len(all_tangrams)=}")
    belongs_dataset = dataset.filter(
        lambda x: set(s.split('.')[0] for s in x["context"]) < all_tangrams)
    assert len(belongs_dataset) == len(dataset)
    print("All tangrams in the dataset are in the split.")


if __name__ == "__main__":
    app.run(main)
