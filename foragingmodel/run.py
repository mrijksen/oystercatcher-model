import data
from oystercatchermodel import OystercatcherModel
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle


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

        # load real patch data
        df_patch_data = data.get_patch_data(start_year)

        # get patchIDs
        patchIDs = df_patch_data.patchID.values

        # load patch availability
        df_patch_availability = data.get_patch_availability(start_year, patchIDs)

        # load environmental data
        df_env = data.get_environmental_data(start_year)

    # instantiate model
    # model = OystercatcherModel(params, patch_name_list, prey, area_of_patches, df_env, df_patch, df_patch_availability)
    model = OystercatcherModel(params, df_patch_data, df_patch_availability, df_env)
    return model

if __name__ == "__main__":

    # run parameters
    start_year = 1996
    artificial_run = False

    if artificial_run:

        # initiate and run model
        model = initiate_model(start_year, run_type='artificial')

    else:

        # initiate and run model
        model = initiate_model(start_year)

    start_time = time.time()
    model.run_model()
    end_time = time.time()
    print(end_time - start_time, "TIME")
    print(len(model.schedule.agents), "number of agents left")

    # save results in file
    output = open('../results/single_simulation_runs/{}_10000agents_newrule_0agg.pkl'.format(start_year), 'wb')
    pickle.dump(model.data, output)
    output.close()

    df_high_water = model.df_env[model.df_env.extreem == 'HW']
    df_high_water.reset_index(inplace=True)

    # plot reference weight from data, weight from simulation, foragingtime, temperature?
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(df_high_water.weight,  color='purple', label="reference weight")
    ax[0].set_title('Mean weight')
    # ax[0].plot( model.schedule.agents[0].weight_throughout_cycle, label="actual weight")
    ax[0].plot(np.array(model.data['mean_weight_w']), markersize=2, label='worm')
    ax[0].plot(np.array(model.data['mean_weight_s']), markersize=2, label='shellfish')
    ax[0].legend()
    ax[0].set_ylabel('Gram')
    ax[0].set_xticklabels([])

    # ax[1].plot(np.array(model.schedule.agents[0].foraging_time_per_cycle)/2, 'go', markersize=2)
    ax[1].set_title('Mean time spend foraging')
    ax[1].plot(np.array(model.data['mean_foraging_w']), markersize=2, label='worm')
    ax[1].plot(np.array(model.data['mean_foraging_s']), markersize=2, label='shellfish')
    ax[1].legend()
    ax[1].set_ylabel('Hours')
    ax[1].set_ylim(0, 15)
    ax[1].set_xticklabels([])

    # ax[2].plot(df_high_water.date_time.dt.date, df_high_water.time_steps_in_cycle / 2, 'ro', markersize=2)
    # ax[2].set_title('Duration of tidal cycle')
    # ax[2].set_ylabel('Hours')
    # ax[2].set_ylim(0, 15)
    # ax[2].set_xticklabels([])

    ax[2].plot(model.waterheight_data)
    ax[2].set_title('Water level')
    ax[2].set_ylabel('cm above NAP')
    ax[2].set_xticklabels([])

    ax[3].plot(df_high_water.date_time.dt.date, df_high_water.temperature)
    ax[3].set_title('Temperature')
    ax[3].set_xlabel('Month')
    ax[3].set_ylabel('Degrees Celsius')

    # ax[4].plot(model.data['total_num_agents'])

    # ax[4].plot(model.schedule.agents[0].start_foraging_list)
    # fig.suptitle('Year {}/{}, number of agents: {}, aggregation factor mudflats 0'.format(start_year, start_year + 1, 7000))


    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig('test')
    plt.show()
