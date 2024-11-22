"""
    python src/dataset/generate_dataset.py --save_as data/nov_27 --anns nov_27
"""
from absl import logging
import json
from pathlib import Path
from typing import List
import pandas as pd
from absl import app, flags, logging
import datasets
from datasets import Dataset
from datasets.features import Features, Sequence, Value


flags.DEFINE_list("anns", default=['jun_5'],
                  help="annotations in the game_folder")
flags.DEFINE_string("save_as", default="data/jun_5_train_and_test",
                    help="target dataset for training")
flags.DEFINE_string("treatment", default="", help="treatment to filter on")

FLAGS = flags.FLAGS

games_folder = Path("visualization/pickle_files/games/")  # raw pickle files
image_folder = Path("data/tangram_pngs")
# config_folder = Path("server/refgame/")  # multiref does not have similarity blocks

games_ids_to_ignore = []


def sorted_list(s):
    return sorted(list(s))


def load_raw(pickle_file: Path) -> Dataset:
    with open(pickle_file, mode="rb") as f:
        df = pd.DataFrame.from_dict(pd.read_pickle(f), orient="index")
    df.end = df.end.map({'successful': 1, 'unsuccessful': 0})
    df2 = df[["chat", "clicks", "context_listener",
              "targets", "split", "config", "treatment", "end"]]

    logging.info(f"checking image exists in {image_folder}")
    images = df2.loc[:, "context_listener"].explode().unique()
    all_exists = all(image_folder.joinpath(im).exists() for im in images)
    if not all_exists:
        raise FileNotFoundError("Not all images exit.")

    # def load_similarity_block(config_path):
    #     with open(config_folder.joinpath(config_path)) as json_file:
    #         c = json.load(json_file)
    #     assert len(c['blocks']) == 1
    #     blocks = c["blocks"][0]['base']
    #     png_blocks = dict()
    #     for k, vs in blocks.items():
    #         png_blocks[k+'.png'] = [v+'.png' for v in vs]
    #     return {"similarity": json.dumps(png_blocks)}

    ds = Dataset.from_pandas(df2).rename_columns({
        "chat": "chats",
        "context_listener": "context",
        "__index_level_0__": "game_id"
    })
    # ds = ds.map(load_similarity_block, input_columns="config",
    #             remove_columns=["config"])
    return ds


def determine_overturned(selected: List[List[str]], deselected:  List[List[str]],
                         strict: bool, immediately: bool) -> List[bool]:
    assert len(selected) == len(deselected)
    num_turns = len(selected)
    ret = [None] * num_turns
    reducer = all if strict else any
    for turn in range(num_turns):
        end = turn + 2 if immediately else num_turns
        is_overturned = False
        if selected[turn]:
            is_overturned = reducer(
                any(new_select in future_deselects
                    for future_deselects in deselected[turn+1:end])
                for new_select in selected[turn]
            )

        if is_overturned:
            ret[turn] = is_overturned
            continue

        if deselected[turn]:
            is_overturned = reducer(
                any(new_deselect in future_selects
                    for future_selects in selected[turn+1:end])
                for new_deselect in deselected[turn]
            )
        ret[turn] = is_overturned
    return ret


def add_turn_labels(sample):
    """
    The pickle files clickss refer to currently selected.
    I add newly selected and newly deselected for each turn.
    Sets are used but they are sorted when serialized.
    """
    assert len(sample["chats"]) == len(sample["clicks"])
    num_turns = len(sample["clicks"])
    targets = set(sample["targets"])
    chat_feedback = sample["chats"][1:] + [""]

    # currently selected AFTER i-th turn
    currently_selected: List[List[str]] = []
    selected: List[List[str]] = []  # turn-wise selection
    deselected: List[List[str]] = []  # turn-wise deselection
    # combining turn-wise newly selected and newly deselected
    clicks: List[List[str]] = []
    good_select: List[bool] = []  # all new selections are among targets
    good_deselect: List[bool] = []  # all new de-selections are non-targets
    prev_selected = set()
    for turn in range(num_turns):
        curr_selected = set(sample["clicks"][turn])
        currently_selected.append(sorted_list(sample["clicks"][turn]))
        newly_selected = curr_selected - prev_selected
        newly_deselected = prev_selected - curr_selected
        selected.append(sorted_list(newly_selected))
        deselected.append(sorted_list(newly_deselected))
        clicks.append(sorted_list(newly_selected | newly_deselected))
        is_good_select = len(newly_selected) > 0 and \
            all(s in targets for s in newly_selected)
        is_good_deselect = len(newly_deselected) > 0 and \
            all(s not in targets for s in newly_deselected)
        good_select.append(is_good_select)
        good_deselect.append(is_good_deselect)
        prev_selected = curr_selected.copy()

    # now shift currently_selected.
    # currently_selected[i] now denotes selected BEFORE the i-th turn
    currently_selected.insert(0, [])
    currently_selected.pop()

    # Visualize in notebooks/mark_overturn.ipynb
    overturned_strict_immediate = determine_overturned(
        selected, deselected, strict=True, immediately=True)
    # True if at least one of the select was later deselected, or a deselect was reselected
    overturned_strict_eventual = determine_overturned(
        selected, deselected, strict=True, immediately=False)
    # similar to above, except we only look at the very next turn (overturned as a result of the immediate feedback)
    overturned_loose_immediate = determine_overturned(
        selected, deselected, strict=False, immediately=True)
    overturned_loose_eventual = determine_overturned(
        selected, deselected, strict=False, immediately=False)
    return {
        "currently_selected": currently_selected,
        "clicks": clicks,
        "selected": selected,
        # when it [[], [], []]. HF datasets is unhappy, enforce _pre_split_feature
        "deselected": deselected,
        "is_good_select": good_select,
        "is_good_deselect": good_deselect,
        "turn_id": list(range(num_turns)),
        "augment_strategy": "",
        "chat_feedback": chat_feedback,
        "overturned_strict_immediate": overturned_strict_immediate,
        "overturned_strict_eventual": overturned_strict_eventual,
        "overturned_loose_immediate": overturned_loose_immediate,
        "overturned_loose_eventual": overturned_loose_eventual,
    }


def split_turns(batch):
    ret_accum = {
        "chats": [],
        "clicks": [],  # needed to build trajectory
        "currently_selected": [],
        "selected": [],
        "deselected": [],
    }
    ret_turn = {
        "currently_selected": [],
        "selected": [],
        "deselected": [],
        "is_good_select": [],
        "is_good_deselect": [],
        "turn_id": [],
        "chat_feedback": [],
        "overturned_strict_immediate": [],
        "overturned_strict_eventual": [],
        "overturned_loose_immediate": [],
        "overturned_loose_eventual": [],
    }
    ret_const = {
        "targets": [],
        "context": [],
        "treatment": [],
        "game_id": [],
        "end": [],
        "split": [],
        "similarity": [],
        "augment_strategy": [],
    }
    num_rows = len(batch["chats"])
    for row in range(num_rows):
        num_turns = len(batch["chats"][row])
        # Each turn accumulates from previous turns in one round
        for k in ret_accum:
            ret_accum[k].extend([batch[k][row][:i+1]
                                for i in range(num_turns)])
        # Each turn has a unique state independent from previous turns
        for k in ret_turn:
            ret_turn[k].extend([batch[k][row][i] for i in range(num_turns)])
        # Each turn has the same values throughout one round
        for k in ret_const:
            ret_const[k].extend([batch[k][row] for _ in range(num_turns)])
    ret_accum["pre_click_selected_accum"] = ret_accum["currently_selected"]
    ret_turn["select_accum"] = ret_accum["selected"]
    ret_turn["deselect_accum"] = ret_accum["deselected"]
    return ret_accum | ret_turn | ret_const


_feature_1d_list_string = Sequence(Value(dtype='string'))
_feature_2d_list_string = Sequence(_feature_1d_list_string)

_pre_split_features = Features(
    # accumulative
    chats=_feature_1d_list_string,
    clicks=_feature_2d_list_string,

    # const
    targets=_feature_1d_list_string,
    split=Value(dtype='string'),
    treatment=Value(dtype='string'),
    game_id=Value(dtype='string'),
    end=Value(dtype='int8'),
    context=_feature_1d_list_string,
    similarity=Value(dtype='string'),
    augment_strategy=Value(dtype='string'),

    # turn-wise
    turn_id=Sequence(Value(dtype='int8')),
    currently_selected=Sequence(_feature_1d_list_string),
    selected=Sequence(_feature_1d_list_string),
    deselected=Sequence(_feature_1d_list_string),
    is_good_select=Sequence(Value(dtype='bool')),
    is_good_deselect=Sequence(Value(dtype='bool')),
    chat_feedback=Sequence(Value(dtype='string')),
    overturned_strict_immediate=Sequence(Value(dtype='bool')),
    overturned_strict_eventual=Sequence(Value(dtype='bool')),
    overturned_loose_immediate=Sequence(Value(dtype='bool')),
    overturned_loose_eventual=Sequence(Value(dtype='bool')),
)

_features = Features(
    # accumulative
    chats=_feature_1d_list_string,
    clicks=_feature_2d_list_string,
    pre_click_selected_accum=_feature_2d_list_string,
    select_accum=_feature_2d_list_string,
    deselect_accum=_feature_2d_list_string,

    # const
    targets=_feature_1d_list_string,
    split=Value(dtype='string'),
    treatment=Value(dtype='string'),
    game_id=Value(dtype='string'),
    end=Value(dtype='int8'),
    context=_feature_1d_list_string,
    similarity=Value(dtype='string'),
    augment_strategy=Value(dtype='string'),

    # turn-wise
    turn_id=Value(dtype='int8'),
    currently_selected=_feature_1d_list_string,
    selected=_feature_1d_list_string,
    deselected=_feature_1d_list_string,
    is_good_select=Value(dtype='bool'),
    is_good_deselect=Value(dtype='bool'),
    chat_feedback=Value(dtype='string'),
    overturned_strict_immediate=Value(dtype='bool'),
    overturned_strict_eventual=Value(dtype='bool'),
    overturned_loose_immediate=Value(dtype='bool'),
    overturned_loose_eventual=Value(dtype='bool'),
)


def main(_):
    ds = datasets.concatenate_datasets([load_raw(games_folder.joinpath(ann))
                                        for ann in FLAGS.anns])

    ds = ds.filter(lambda x: x["game_id"] not in games_ids_to_ignore)

    if FLAGS.treatment != "":
        ds = ds.filter(lambda x: x["treatment"] == FLAGS.treatment)

    ds = ds.map(add_turn_labels, num_proc=8,
                desc="add turn labels", features=_pre_split_features)
    ds = ds.map(split_turns, batched=True, batch_size=1, features=_features,
                desc="split games into turns")
    ds = ds.map(
        lambda x: {'game_turn_id': x['game_id'] + "_" + str(x['turn_id'])},
        desc="add game_turn_id"
    )
    ds = ds.filter(lambda x: len(x["chats"][-1]) > 0,
                   desc="remove turns if speaker idled")
    ds = ds.filter(lambda x: len(x["clicks"][-1]) > 0,
                   desc="remove turns if listener idled")

    # House keeping
    ds = ds.with_format("torch", device="cpu")
    logging.info("dataset has been built")
    logging.info(ds)
    logging.info(ds[0])
    ds.save_to_disk(FLAGS.save_as)
    logging.info(f"saved to {FLAGS.save_as}")
    ds.to_json(f"{FLAGS.save_as}/plain.json")
    logging.info(f"saved to {FLAGS.save_as}/plain.json")


if __name__ == "__main__":
    app.run(main)
