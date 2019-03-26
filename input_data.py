import numpy as np


def get_availability(patches, timesteps):
    """ Returns random true false array with dimensions patches x timesteps
    """
    return np.random.choice([True, False], (timesteps, patches))


# create some data
num_patches = 10
timesteps = 40

# initial prey for all patches
init_prey = [np.random.randint(0, 100) for x in range(num_patches)]
availability = get_availability(num_patches, timesteps)

print(np.shape(availability))
