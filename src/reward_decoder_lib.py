from pathlib import Path


def read_txt_as_list(filepath: str):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return [l.strip() for l in lines]


class RewardDecoderLib:
    def __init__(self, accepted_prompt_ids) -> None:
        self._storage = dict()
        self._accepted_prompt_ids = accepted_prompt_ids
        # 3 column: game_turn_id, prompt_id, value (+1, 0, -1)

    def _extract_prompt_id(self, folder_name: str) -> str:
        for apid in self._accepted_prompt_ids:
            if folder_name.endswith(apid):
                return apid
        return None

    def get(self, game_turn_id, prompt_id) -> float:
        return self._storage[prompt_id][game_turn_id]

    def set(self, prompt_id, game_id, value) -> None:
        if prompt_id not in self._storage:
            self._storage[prompt_id] = dict()
        if game_id not in self._storage[prompt_id]:
            self._storage[prompt_id][game_id] = value
        else:
            assert self._storage[prompt_id][game_id] == value

    def load(self, dir) -> None:
        assert len(self._storage) == 0
        dir = Path(dir)
        for subdir in dir.iterdir():
            prompt_id = self._extract_prompt_id(subdir.name)
            if prompt_id is None:
                continue
            pos = read_txt_as_list(subdir / "pos.txt")
            _ = [self.set(prompt_id, gt_id, 1) for gt_id in pos]
            neg = read_txt_as_list(subdir / "neg.txt")
            _ = [self.set(prompt_id, gt_id, -1) for gt_id in neg]
            neu = read_txt_as_list(subdir / "neu.txt")
            _ = [self.set(prompt_id, gt_id, 0) for gt_id in neu]


if __name__ == "__main__":
    policy_lib = RewardDecoderLib(
        ["02_one_history_binary", "03_one_history_trinary"])
    policy_lib.load("data/reward_decoder_outputs")
    assert list(policy_lib._storage.keys()) == [
        "02_one_history_binary", "03_one_history_trinary"]
