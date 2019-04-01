import json
import random
import numpy as np

def get_params():
    """Loads parameters from json file"""

    config_model = "config_file.json"
    return json.load(open(config_model))


def get_random_availability(params):
    """ Returns random true false array with dimensions patches x timesteps
    """
    timesteps = int(params["num_tidal_cycles"] * params["minutes_in_tidal_cycle"]
                               /params["resolution_min"])
    return np.random.choice([True, False], (timesteps, params["num_patches"]))


def get_random_prey(params):
    """Returns random array with prey between min_prey and max_prey for
    given number of patches
    """
    return [random.randint(params["min_prey"], params["max_prey"])
            for x in range(params["num_patches"])]