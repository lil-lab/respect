from pprint import pprint

import gin
import gin.torch.external_configurables
from absl import flags
from peft import LoraConfig
from transformers import BitsAndBytesConfig

gin.external_configurable(BitsAndBytesConfig, module="BitsAndBytesConfig")
gin.external_configurable(LoraConfig, module="LoraConfig")

_GIN_FILE = flags.DEFINE_multi_string(
    'gin_file', None, 'List of paths to the config files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_param', None, 'Newline separated list of Gin parameter bindings.')


def gin_config_to_readable_dictionary(gin_config: dict):
    """credit: https://github.com/google/gin-config/issues/154
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data[".".join([name, k])] = v
    return data


def parse_config(verbose=True) -> dict:
    """This should be the only exported function from this file

    Returns:
        dict: of gin configurations
    """
    gin.parse_config_files_and_bindings(
        _GIN_FILE.value, _GIN_BINDINGS.value, print_includes_and_imports=True)
    gin_config_dict = gin_config_to_readable_dictionary(gin.config._CONFIG)
    if verbose:
        pprint(gin_config_dict)
    return gin_config_dict
