import csv
import os
from tqdm import tqdm
from absl import logging


class PolicyLib:
    def __init__(self) -> None:
        self._storage = dict()

    def _get_key(self, game_id, turn_id, policy_id):
        return (game_id, turn_id, policy_id)

    def _read_key(self, key):
        return key

    def get(self, game_id, turn_id, policy_id) -> float:
        if policy_id == "1/10":
            return 0.1
        if policy_id == "1/20":
            return 0.05
        if not isinstance(turn_id, int):
            turn_id = turn_id.item()
        key = self._get_key(game_id, turn_id, policy_id)
        return self._storage[key]

    def set(self, game_id, turn_id, policy_id, value) -> None:
        assert isinstance(value, float)
        assert isinstance(game_id, str)
        assert isinstance(turn_id, int)
        assert isinstance(policy_id, str)
        key = self._get_key(game_id, turn_id, policy_id)
        if key in self._storage:
            logging.error(f"Key already exists: {key}")
            raise ValueError("Key already exists")
        self._storage[key] = value

    def save(self, file_path) -> None:
        with open(file_path, "a+") as f:
            writer = csv.writer(f, delimiter=',')
            for key, value in tqdm(self._storage.items()):
                game_id, turn_id, policy_id = self._read_key(key)
                writer.writerow([game_id, turn_id, policy_id, value])
        self._storage = dict()

    def load(self, filepath_or_dir) -> None:
        assert len(self._storage) == 0
        if os.path.isdir(filepath_or_dir):
            filepaths = [
                os.path.join(filepath_or_dir, file_name)
                for file_name in os.listdir(filepath_or_dir)
            ]
        else:
            filepaths = [filepath_or_dir]
        logging.info(f"Loading policy lib from files {filepaths}")
        for filepath in filepaths:
            filepath = os.path.expanduser(filepath)
            with open(filepath, "r") as f:
                reader = csv.reader(f, delimiter=',')
                for row in tqdm(reader):
                    game_id, turn_id, policy_id, value = row
                    self.set(game_id, int(turn_id), policy_id, float(value))

    def populate(self, example, policy_id):
        """ thin wrapper around set suitable for dataset.map """
        game_id = example["game_id"]
        turn_id = example["turn_id"]
        if not isinstance(turn_id, int):
            turn_id = turn_id.item()
        value = example["prob_action_poor"]
        if not isinstance(value, float):
            value = value.item()
        self.set(game_id, turn_id, policy_id, value)
        return dict()  # will be ignored


if __name__ == "__main__":
    policy_lib = PolicyLib()
    policy_lib.set("game1", 1, "policy1", 0.5)
    policy_lib.save("data/policy_lib/test.csv")
    policy_lib.load("data/policy_lib/test.csv")
    assert policy_lib.get("game1", 1, "policy1"), 0.5
