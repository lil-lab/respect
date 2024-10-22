
"""
python src/dataset/summary.py
"""

import sys  # isort:skip
sys.path.append('./src')  # isort:skip # noqa: E402

import datasets
from absl import app

from src.base import simple_filter

flags = app.flags

flags.DEFINE_bool(
    'base_simple_filter', False, help='use simple filter for base dataset'
)

FLAGS = flags.FLAGS

def main(argv):
    dataset = datasets.load_from_disk(argv[1])
    print(f"{argv[1]} dataset summary")
    print("== schema ==")
    print(dataset)
    print("== unique games ==")
    print(f"{len(set(dataset['game_id']))}")
    print("== train game endings ==")
    df = dataset.to_pandas()
    print(df["end"].value_counts())
    print("== treatment == ")
    print(df['treatment'].value_counts())
    if "prob_action_poor" in dataset.column_names:
        print("== train prob_action_poor ==")
        print(df["prob_action_poor"].describe().apply(
            lambda x: format(x, 'f')))
    if FLAGS.base_simple_filter:
        print("== base simple filter ==")
        print(simple_filter(dataset))
    print(f"== end of summary of {argv[1]} ==")

if __name__ == "__main__":
    app.run(main)
