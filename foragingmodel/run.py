import data

from model import OystercatcherModel


params = data.get_params()


def initiate_model(params=params):

    """Instantiate model class"""

    # initial prey for all patches
    init_prey = data.get_random_prey(params)

    # availability for all patches and all time steps
    availability = data.get_random_availability(params)

    # area for all patches
    area_of_patches = data.get_random_area(params)

    # instantiate model
    model = OystercatcherModel(params, init_prey, availability, area_of_patches)
    return model


# initiate and run model
model = initiate_model()
model.run_model()

# print([agent for agent in model.agents_on_patches[0] if agent.unique_id != 0])
# print(model.agents_on_patches[0])
