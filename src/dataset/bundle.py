"""
This script is used to bundle multiple datasets into one

python src/dataset/bundle.py --to_combine "jul_31_train_ba","aug_1_train_ba","aug_2_train_ba" --save_as data/jul31aug12_ba
"""

from absl import flags, app, logging
import datasets
from pathlib import Path

flags.DEFINE_list("to_combine",
                  default=["may_28_train_and_test",
                           "may_29_train_and_test",
                           "jun_4_train_and_test"],
                  help="names of the datasets to be combined")
flags.DEFINE_string("save_as", default="data/may2829jun4",
                    help="target dataset for igl training")

FLAGS = flags.FLAGS


def main(_):
    input_dir = Path("data")
    dsets = [datasets.load_from_disk(input_dir / name)
             for name in FLAGS.to_combine]
    combined = datasets.concatenate_datasets(dsets)
    combined.save_to_disk(FLAGS.save_as)
    combined.to_json(f"{FLAGS.save_as}/plain.json")
    print(combined)
    logging.info(f"Combined dataset saved to {FLAGS.save_as}")


if __name__ == "__main__":
    app.run(main)
