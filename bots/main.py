# File: main
# ----------
# Main event manager for serving bots to Empirica.

"""
python bots/main.py --request_log feb_19 > bots/logs/feb_19.log
"""

import argparse
import asyncio
import dataclasses
import json
import logging
from pathlib import Path
from time import time
from typing import Dict, List, TypeVar, Union

import ray
import tornado.ioloop
import tornado.web
import yaml
from model_manager import construct_manager
from src.utils import get_logger

# CHANGE ME
ray.init(num_cpus=8, num_gpus=1, log_to_driver=True, logging_level='debug')

# Name type, newtype of str. e.g. "page4-249.svg"
N = TypeVar('N')

@dataclasses.dataclass
class BotRequest:
    image_paths: Dict[str, Union[str, N]]
    last_msg: str
    all_msg: str
    target: List[N]  # ground truth, should only be used by Oracle
    bot_treatment: str
    previous_selected: List[List[N]]  # list of previous "path"
    game_id: str
    turn_id: int

@dataclasses.dataclass
class BotResponse:
    path: List[N]  # path of all guesses (accumulative)
    timePassed: float  # in seconds
    prob: float
    decoded_out: str

class BotTaskAllocator():

    def __init__(self, configs):
        # Initialize the model managers for each config
        self.model_managers = {
            config["treatment_name"]:
                construct_manager(config) for config in configs
        }

    async def respond_to_listener_request(self, bot_treatment, image_paths,
                                          chats, target_path,
                                          previous_selected):
        return await self.model_managers[bot_treatment].listener_predict(
            image_paths, chats, target_path, previous_selected)

class ListenerServer(tornado.web.RequestHandler):

    async def post(self):
        start_time = time()

        args_str = self.request.body.decode("utf-8")
        args = json.loads(args_str)
        bot_request = BotRequest(**args)

        # build request and ask the model
        logger.debug(f"{bot_request=}")
        logger.info(f"Begin prediction for {bot_request.game_id}-{bot_request.turn_id} by {bot_request.bot_treatment}.")

        target_path = await bot_task_allocator.respond_to_listener_request(
            bot_request.bot_treatment, bot_request.image_paths,
            bot_request.all_msg, bot_request.target,
            bot_request.previous_selected)
        prob, decoded_out = None, None
        if isinstance(target_path, dict):
            prob = target_path["prob"]
            decoded_out = target_path["decoded_out"]
            target_path = target_path["path"]
        time_passed = time() - start_time

        logger.info(f"Req bot_treatment: {bot_request.bot_treatment}.")
        logger.info(f"Req game_id: {bot_request.game_id}.")
        logger.info(f"Req turn_id: {bot_request.turn_id}.")
        logger.info(f"Req last_msg: {bot_request.last_msg}.")
        logger.info(f"Res output: {' '.join(decoded_out[-80:].split())}.")
        logger.info(f"Res probability: {prob}.")
        logger.info(f"Res predicted target path: {target_path}.")
        logger.info(f"Res took {time_passed} seconds.")

        bot_response = BotResponse(target_path, time_passed, prob, None)
        bot_response_str = json.dumps(dataclasses.asdict(bot_response))
        self.write(bot_response_str)

        logger.info("End prediction\n")

        bot_response = BotResponse(target_path, time_passed, prob, decoded_out)
        request_and_response = {
            "request": dataclasses.asdict(bot_request),
            "response": dataclasses.asdict(bot_response)
        }
        clean_request_logger.info(json.dumps(request_and_response))


def get_args():
    parser = argparse.ArgumentParser(description="Serving bots to Empirica")
    parser.add_argument('--configuration_file', type=str, default="testing.yaml",
                        help="A configuration file specifying the models to launch for this experiment")
    parser.add_argument('--request_log', type=str, required=True,
                        help="path to save request logs")
    args = parser.parse_args()
    return args

def load_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

def read_experiment_configs(configuration_file):
    bots_path = Path.cwd().joinpath("bots/configs")
    yaml_path = bots_path.joinpath(configuration_file)
    yaml_file = load_yaml(yaml_path)
    configs = [load_yaml(bots_path.joinpath(cfg))
               for cfg in yaml_file["configs"]]
    return configs

def make_app():
    app = tornado.web.Application([
        (r"/predict_target", ListenerServer),
    ],
        debug=True, autoreload=False,
    )
    return app


def get_request_logger(request_log: str):
    path = f"bots/request_logs/request_{request_log}.log"
    fh = logging.FileHandler(path)
    logger = logging.getLogger("clean_request_log")
    # make sure this logger does not print to stdout
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return logger

async def main():
    app = make_app()
    app.listen(8571)
    shutdown_event = asyncio.Event()
    logger.info("Beginning wait")
    await shutdown_event.wait()

if __name__ == "__main__":
    # Get the experiment arguments
    args = get_args()
    logger = get_logger(__name__, level=logging.INFO)
    clean_request_logger = get_request_logger(args.request_log)
    configs = read_experiment_configs(args.configuration_file)
    bot_task_allocator = BotTaskAllocator(configs)
    asyncio.run(main())
