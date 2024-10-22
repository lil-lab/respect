# File: model_manager
# -------------------
# Script for making predictions for listeners and speakers
# in an asynchronous manner

import random
from abc import ABC, abstractmethod

from ray_models import RayIdefics


class BaseManager(ABC):

    @abstractmethod
    def listener_predict(self, image_paths, chats, target_path, previous_selected):
        '''
        The interface assumes the following as input:
            * image_paths: A list of dictionaries where the tangram path is mapped to the key "path"
            * chats: A list of chats (str) so far in the current round
            * target_path: Not to be used outside of debugging. The path of the target image
            * previous_selected: A list of list of .svg file names (str) representing previous guesses of the targets
        The interface should produce an image path (str) as an output
        '''
        pass


class DummyManager(BaseManager):

    async def listener_predict(self, image_paths, chats, target_path, previous_selected):
        random_num = random.randint(1, 2)
        random_paths = random.sample(image_paths, random_num)
        random_paths = [item['path'] for item in random_paths]
        return random_paths


class OracleManager(BaseManager):

    async def listener_predict(self, image_paths, chats, target_path, previous_selected):
        return target_path


class SimpleModelManager(BaseManager):

    def __init__(self, config):
        self.num_models = config["num_duplicates"]
        assert config["treatment_name"].startswith("idefics")
        print(f"Initializing {self.num_models} models for {config['treatment_name']}")
        self.listeners = [RayIdefics.remote(
            config) for _ in range(self.num_models)]

    async def listener_predict(self, image_paths, chats, target_path, previous_selected):
        curr_idx = random.randint(0, self.num_models-1)
        return await self.listeners[curr_idx].predict.remote(image_paths, chats, previous_selected)


def construct_manager(config):
    if config["manager_type"] == "dummy":
        return DummyManager()
    elif config["manager_type"] == "oracle":
        return OracleManager()
    elif config["manager_type"] == "simple":
        return SimpleModelManager(config)
    else:
        raise NotImplementedError
