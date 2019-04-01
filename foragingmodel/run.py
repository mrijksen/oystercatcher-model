import data

from model import OystercatcherModel


params = data.get_params()


def initiate_model(params=params):

    """Instantiate model class"""

    # initial prey for all patches
    init_prey = data.get_random_prey(params)

    # availability for all patches and all time steps
    availability = data.get_random_availability(params)

    # instantiate model
    model = OystercatcherModel(params, init_prey, availability)
    return model


# initiate and run model
model = initiate_model()
model.run_model()

