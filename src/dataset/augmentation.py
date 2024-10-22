"""
python src/dataset/augmentation.py --augment_strategy remove_last_turn_selection --dataset data/apr_23_dev --save_as data/apr_23_dev_remove_last_turn_selection
"""

import copy
import random

import datasets
from absl import app, flags, logging


class AugmentStrategy:
    """ sample are from post split turns
    """

    def is_eligible(self, sample):
        raise NotImplementedError

    def augment(self, sample):
        new_sample = copy.deepcopy(sample)
        new_sample["augment_strategy"] = str(self.__class__)
        return new_sample

    def get_click_candidates(self, sample):
        raise NotImplementedError

    def apply_this_clicks(self, sample):
        clicks = sample["clicks"][-1]
        currently_selected = set(sample["currently_selected"])

        currently_selected_post = set(sample["currently_selected"])
        for click in clicks:
            if click in currently_selected:
                currently_selected_post.remove(click)
            else:
                currently_selected_post.add(click)
        return list(currently_selected_post)


class RemoveLastTurnSelection(AugmentStrategy):
    prompts = [
        "Bad select",
        "Deselect the last one",
        "Remove what you just selected",
        "Wrong, undo what you selected",
        "What you just selected was wrong. Deselect that"
    ]

    def get_click_candidates(self, sample):
        return list(sample["select_accum"][-1])

    def is_eligible(self, sample):
        if sample["is_good_select"]:
            return False
        return len(self.get_click_candidates(sample)) > 0

    def augment(self, sample):
        new_sample = super().augment(sample)
        new_sample["turn_id"] += 1
        new_sample["game_turn_id"] = f"{sample['game_id']}_{sample['turn_id'].item()}_aug_remove_last"
        deselect_clicks = self.get_click_candidates(sample)
        new_sample["chats"].append(random.choice(self.prompts))
        new_sample["clicks"].append(deselect_clicks)
        new_sample["currently_selected"] = self.apply_this_clicks(sample)
        new_sample["select_accum"].append([])
        new_sample["deselect_accum"].append(deselect_clicks)
        new_sample["pre_click_selected_accum"].append(
            sample["currently_selected"])

        new_sample["is_good_select"] = False
        new_sample["is_good_deselect"] = True
        new_sample["selected"] = []
        new_sample["deselected"] = deselect_clicks
        return new_sample


class RemoveCurrentSelection(AugmentStrategy):
    prompts = [
        "Clear all",
        "Remove all currently selected",
        "Deselect everything. Let's start over",
        "Remove everything that you have selected",
        "No, remove everything."
    ]

    def get_click_candidates(self, sample):
        return self.apply_this_clicks(sample)

    def is_eligible(self, sample):
        return len(self.get_click_candidates(sample)) > 0

    def augment(self, sample):
        new_sample = super().augment(sample)
        new_sample["turn_id"] += 1
        new_sample["game_turn_id"] = f"{sample['game_id']}_{sample['turn_id'].item()}_aug_remove_current"

        deselect_clicks = self.get_click_candidates(sample)
        new_sample["chats"].append(random.choice(self.prompts))
        new_sample["clicks"].append(deselect_clicks)
        new_sample["currently_selected"] = self.apply_this_clicks(sample)
        new_sample["select_accum"].append([])
        new_sample["deselect_accum"].append(deselect_clicks)
        new_sample["pre_click_selected_accum"].append(
            new_sample["currently_selected"])

        new_sample["is_good_select"] = False
        new_sample["is_good_deselect"] = True
        new_sample["selected"] = []
        new_sample["deselected"] = deselect_clicks
        return new_sample


class SingleClick(AugmentStrategy):

    def get_click_candidates(self, sample):
        actual_click = sample["clicks"][-1]
        if len(actual_click) == 1 and actual_click[0] in sample["context"][0]:
            return sample["context"][1]
        return sample["context"][0]

    def is_eligible(self, sample):
        return True

    def augment(self, sample):
        new_sample = super().augment(sample)
        turn_id = sample["turn_id"]
        if not isinstance(turn_id, int):
            turn_id = turn_id.item()
        new_sample["game_turn_id"] = f"{sample['game_id']}_{turn_id}_aug_single_click"
        click = self.get_click_candidates(sample)

        new_sample["clicks"][-1] = [click]

        if click in sample["currently_selected"]:
            # deselect
            new_sample["deselected"] = [click]
            new_sample["deselect_accum"][-1] = [click]
            new_sample["selected"] = []
            new_sample["select_accum"][-1] = []
        else:
            # select
            new_sample["selected"] = [click]
            new_sample["select_accum"][-1] = [click]
            new_sample["deselected"] = []
            new_sample["deselect_accum"][-1] = []
        # neither should be used
        new_sample["is_good_select"] = None
        new_sample["is_good_deselect"] = None
        # should be overwritten
        new_sample["prob_action_poor"] = float("nan")
        return new_sample


def main(_):
    ds = datasets.load_from_disk(FLAGS.dataset)
    augment_strategy = FLAGS.augment_strategy
    if augment_strategy == "remove_last_turn_selection":
        augmenter = RemoveLastTurnSelection()
    elif augment_strategy == "remove_current_selection":
        augmenter = RemoveCurrentSelection()
    elif augment_strategy == "single_click":
        augmenter = SingleClick()
    else:
        raise NotImplementedError(f"{augment_strategy} not supported")

    ds = ds.filter(augmenter.is_eligible).map(augmenter.augment,
                                              desc=f"applying {augment_strategy=}",
                                              features=ds.features)

    ds = ds.with_format("torch", device="cpu")
    logging.info("dataset has been built")
    logging.info(ds)
    logging.info(ds[0])
    ds.save_to_disk(FLAGS.save_as)
    logging.info(f"saved to {FLAGS.save_as}")
    ds.to_json(f"{FLAGS.save_as}/plain.json")
    logging.info(f"saved to {FLAGS.save_as}/plain.json")
    return


if __name__ == "__main__":
    flags.DEFINE_string("save_as", None, "save as", required=True)
    flags.DEFINE_string("dataset", None, help="load from", required=True)
    flags.DEFINE_string("augment_strategy", None,
                        "augment strategy", required=True)

    FLAGS = flags.FLAGS

    NUM_CONTEXT = 10
    app.run(main)
