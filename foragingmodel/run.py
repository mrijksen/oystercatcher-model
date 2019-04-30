import data
from oystercatchermodel import OystercatcherModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def initiate_model(env_data_path, patch_data_path):
    """ Instantiate model class """

    # get parameters todo: dit ook in path zetten?
    params = data.get_params()

    # load patches data todo: hier moeten we dus de echte patch data gaan laden
    patch_name_list = data.create_patch_list(params)
    prey = data.create_random_prey(params, patch_name_list)

    # area for all patches todo: hier dus de echte data
    area_of_patches = data.get_random_area(params)

    # load environmental data
    df_env = pd.read_pickle(env_data_path)

    # load patch data todo
    df_patch = None

    # instantiate model
    model = OystercatcherModel(params, patch_name_list, prey, area_of_patches, df_env, df_patch)
    return model


# def run_model(model, num_steps):
#     """ Runs the model with a given data set for the patches and the environment
#     """
#
#     print("Initial number birds: {}".format(model.init_birds))
#
#     # simulate for given number of num_steps
#     for i in range(num_steps): #todo: geef hier aantal stappen in df mee
#         model.step()
#
#     print("Final number of birds: {}".format(model.schedule.get_agent_count()))


if __name__ == "__main__":

    # location of environmental data
    env_data_dir = 'C:/Users/Marleen/Documents/thesis project/oystercatcher-model/Input data/'
    env_data_filename = '2017_9_1_to_2018_3_1.pkl'
    env_data_path = env_data_dir + env_data_filename

    # location of patch data
    patch_data_path = 1

    # initiate and run model
    model = initiate_model(env_data_path, patch_data_path)
    model.run_model()

    print(np.mean(model.schedule.agents[0].weight_throughout_cycle))

    # plot something
    plt.figure(1)
    plt.plot(model.schedule.agents[0].weight_throughout_cycle, label="weight")
    plt.legend()
    plt.figure(2)
    plt.plot(model.schedule.agents[0].stomach_content_list, label="stomach contents")
    plt.legend()
    plt.show()
