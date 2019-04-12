import json
import random
import numpy as np

def get_params():
    """Loads parameters from json file"""

    config_model = "config_file.json"
    return json.load(open(config_model))

def get_timesteps(params):
    """Returns the total number of time steps in the simulation
    """
    return int(params["num_tidal_cycles"] * params["minutes_in_tidal_cycle"]
                    / params["resolution_min"])

def get_random_availability(params):
    """ Returns random true false array with dimensions patches x timesteps
    """
    time_steps = get_timesteps(params)
    return np.random.choice([True, False], (time_steps, params["num_patches"]))


def get_random_prey(params):
    """Returns random array with prey between min_prey and max_prey for
    given number of patches.

    The prey in the final array is the density of prey (# prey/ m2)
    """
    return np.array([random.randint(params["min_prey"], params["max_prey"])
                     for x in range(params["num_patches"])], dtype=float)

def get_random_area(params):
    """
    Returns random array with areas between min_area and max_area for
    given number of patches
    :param params:
    :return:
    """
    return np.array([random.randint(params["min_area"], params["max_area"])
                     for x in range(params["num_patches"])])


def create_patch_list(params):
    """ Returns list with patch type for every patch id.

    :patch_types is list with strings describing patch types
    :patch_type_counts is list with number of patches for each patch type.
    """

    patch_types = params["patch_name_list"]
    patch_type_counts = params["patch_type_counts"]
    patch_name_list = []
    for i in range(len(patch_type_counts)):
        for j in range(patch_type_counts[i]):
            patch_name_list.append(patch_types[i])
    return patch_name_list


def create_random_prey(params, patch_name_list):
    """ Returns prey for all patches.

    Input is list with patch types. Depending on the patch type
    different prey is assigned to the patch. """

    density_mussel = params["density_mussel"]
    density_kokkels_1 = params["density_kokkels_1"]
    density_kokkels_2 = params["density_kokkels_2"]
    density_kokkels_mj = params["density_kokkels_mj"]

    prey = []

    for patch in patch_name_list:
        if patch == "Bed":
            prey.append({"mussel_density": density_mussel})
        if patch == "Mudflat":
            prey.append({"kok1": density_kokkels_1, "kok2": density_kokkels_2,
                         "kokmj": density_kokkels_mj})
    return prey