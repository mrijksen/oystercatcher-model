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

    # load environmental data todo: dit kan later weg
    df_env = pd.read_pickle(env_data_path)

    df_high_water = df_env[df_env.extreem == 'HW']
    df_high_water.reset_index(inplace=True)

    # plot reference weight from data, weight from simulation, foragingtime, temperature?
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(df_high_water.weight,  color='purple', label="reference weight")
    ax[0].set_title('Reference weight from data')
    ax[0].plot(model.schedule.agents[0].weight_throughout_cycle, label="actual weight")
    ax[0].legend()
    ax[0].set_ylabel('Gram')

    ax[1].plot(np.array(model.schedule.agents[0].foraging_time_per_cycle[1:])/2, 'go', markersize=2)
    ax[1].set_title('Time spend foraging')
    ax[1].set_ylabel('Hours')
    ax[1].set_ylim(0, 15)


    ax[2].plot(df_high_water.time_steps_in_cycle / 2, 'ro', markersize=2)
    ax[2].set_title('Duration of tidal cycle')
    ax[2].set_ylabel('Hours')
    ax[2].set_ylim(0, 15)
    ax[2].set_xlabel('Tidal cycle number')

    fig.suptitle('Mussel bed')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig('test')

    print(model.schedule.agents[0].stomach_content_list)

    plt.show()
