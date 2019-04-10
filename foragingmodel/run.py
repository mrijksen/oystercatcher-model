import data

from oystercatchermodel import OystercatcherModel

import matplotlib.pyplot as plt


params = data.get_params()


def initiate_model(params=params):

    """Instantiate model class"""

    # initial prey for all patches
    # init_prey = data.get_random_prey(params)
    prey = data.create_random_prey(params)

    # availability for all patches and all time steps
    availability = data.get_random_availability(params)

    # area for all patches
    area_of_patches = data.get_random_area(params)

    # instantiate model
    model = OystercatcherModel(params, prey, availability, area_of_patches)
    return model


# initiate and run model
model = initiate_model()
model.run_model()

print(model.schedule.agents[0].weight_throughout_cycle)

plt.figure(1)
plt.plot(model.schedule.agents[0].weight_throughout_cycle, label="weight")
plt.legend()
plt.figure(2)
plt.plot(model.schedule.agents[0].stomach_content_list, label="stomach contents")
# plt.axhline(500, c="r", label="reference weight")
plt.legend()
plt.show()
# print([agent for agent in model.agents_on_patches[0] if agent.unique_id != 0])
# print(model.agents_on_patches[0])
