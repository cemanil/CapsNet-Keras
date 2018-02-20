import json
from bunch import Bunch


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # Parse the configurations from the config json file provided.
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # Convert the dictionary to a namespace using bunch lib.
    config = Bunch(config_dict)

    return config, config_dict


def process_config(jsonfile):
    config, _ = get_config_from_json(jsonfile)
    return config


def merge_configs(config1, config2):
    combined_config_dict = _merge_two_dicts(config1, config2)

    combined_config = Bunch(combined_config_dict)

    return combined_config


def _merge_two_dicts(x, y):
    z = x.copy()  # Start with x's keys and values.
    z.update(y)  # Modifies z with y's keys and values & returns None.
    return z
