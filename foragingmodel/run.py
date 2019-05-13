import data
from oystercatchermodel import OystercatcherModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def initiate_model(start_year, run_type='real_data'):
    """ Instantiate model class """

    # get parameters todo: dit ook in path zetten?
    params = data.get_params()
    if run_type == 'artificial':

        # load artificial patch data
        df_patch_data = data.get_artificial_patch_data()

        # load environmental data
        df_env = data.get_part_of_environmental_data()

        # load artificial availability
        df_patch_availability = data.get_artificial_patch_availability()

    else:

        # load patches data todo: hier moeten we dus de echte patch data gaan laden
        patch_name_list = data.create_patch_list(params)
        prey = data.create_random_prey(params, patch_name_list)

        # area for all patches todo: hier dus de echte data
        area_of_patches = data.get_random_area(params)

        # load environmental data
        df_env = data.get_environmental_data(start_year)

        # load patch data todo
        df_patch = None

        # load patch availability
        df_patch_availability = data.get_patch_availability(start_year)

        df_patch_data = 1


    # instantiate model
    # model = OystercatcherModel(params, patch_name_list, prey, area_of_patches, df_env, df_patch, df_patch_availability)
    model = OystercatcherModel(params, df_patch_data, df_patch_availability, df_env)
    return model

if __name__ == "__main__":

    # run parameters
    start_year = 2017
    artificial_run = True

    if artificial_run:

        # initiate and run model
        model = initiate_model(start_year, run_type='artificial')

    else:

        # initiate and run model
        model = initiate_model(start_year)
    model.run_model()

    # # load environmental data todo: dit kan later weg
    # # location of environmental data todo: dit ook in data.py zetten? en alleen startjaar meegevem?
    # env_data_dir = 'C:/Users/Marleen/Documents/thesis project/oystercatcher-model/Input data/'
    # env_data_filename = '{}_9_1_to_{}_3_1.pkl'.format(start_year, start_year + 1)
    # env_data_path = env_data_dir + env_data_filename
    # df_env = pd.read_pickle(env_data_path)

    df_high_water = model.df_env[model.df_env.extreem == 'HW']
    df_high_water.reset_index(inplace=True)

    # plot reference weight from data, weight from simulation, foragingtime, temperature?
    fig, ax = plt.subplots(4, 1)
    ax[0].plot( df_high_water.weight,  color='purple', label="reference weight")
    ax[0].set_title('Weight')
    ax[0].plot( model.schedule.agents[0].weight_throughout_cycle, label="actual weight")
    ax[0].legend()
    ax[0].set_ylabel('Gram')
    ax[0].set_xticklabels([])

    ax[1].plot(np.array(model.schedule.agents[0].foraging_time_per_cycle)/2, 'go', markersize=2)
    ax[1].set_title('Time spend foraging')
    ax[1].set_ylabel('Hours')
    ax[1].set_ylim(0, 15)
    ax[1].set_xticklabels([])


    ax[2].plot(df_high_water.date_time.dt.date, df_high_water.time_steps_in_cycle / 2, 'ro', markersize=2)
    ax[2].set_title('Duration of tidal cycle')
    ax[2].set_ylabel('Hours')
    ax[2].set_ylim(0, 15)
    ax[2].set_xticklabels([])


    ax[3].plot(df_high_water.date_time.dt.date, df_high_water.temperature)
    ax[3].set_title('Temperature')
    ax[3].set_xlabel('Month')
    ax[3].set_ylabel('Degrees Celsius')

    # ax[4].plot(model.schedule.agents[0].start_foraging_list)

    fig.suptitle('Foraging on mussel bed')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig('test')

    print(np.mean(model.schedule.agents[0].foraging_time_per_cycle[1:]))

    plt.show()
