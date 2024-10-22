"""
python src/dataset/backfill_policy_lib.py --policy_lib_file_path data/policy_lib/dec_8.csv --dataset data/dec_8_a --policy_id 1252536-1380

python src/dataset/backfill_policy_lib.py --policy_lib_file_path data/policy_lib/dec_8.csv --dataset data/dec_8_a2 --policy_id 2642336
"""

import datasets
from absl import app
from absl import flags
from policy_lib import PolicyLib

flags.DEFINE_string("policy_lib_file_path", default=None, required=True,
                    help="csv path to store poor policies")
flags.DEFINE_string("policy_id", default=None, required=True,
                    help="prob action predicted by policy")
flags.DEFINE_string("dataset", default=None, required=True,
                    help="dataset (with an _a) that will back fill the csv")

FLAGS = flags.FLAGS


def main(_):
    dataset = datasets.load_from_disk(FLAGS.dataset)
    print(dataset)
    policy_lib = PolicyLib()
    _ = dataset.map(lambda x: policy_lib.populate(x, FLAGS.policy_id))
    policy_lib.save(FLAGS.policy_lib_file_path)


if __name__ == "__main__":
    app.run(main)
