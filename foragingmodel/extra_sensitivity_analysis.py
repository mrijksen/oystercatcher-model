import numpy as np
from SALib.sample import saltelli
import toml
import data
from oystercatchermodel import OystercatcherModel
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
import toml

N = 5
fname = "../results/extra_sensitivity_standardparams.txt"
standard_params = toml.load("standard_params_config_file.toml")


def initiate_model(start_year, i):
    """ Instantiate model class """

    model_params = standard_params.copy()

    # replace value in final parameter file
    for var, val in zip(vars, param_set_vals[i]):
        model_params[var] = val

    # load real patch data
    df_patch_data = data.get_patch_data(start_year)

    # get patchIDs
    patchIDs = df_patch_data.patchID.values

    # load patch availability
    df_patch_availability = data.get_patch_availability(start_year, patchIDs)

    # load environmental data
    df_env = data.get_environmental_data(start_year)

    # instantiate model
    model = OystercatcherModel(model_params, df_patch_data, df_patch_availability, df_env)
    return model


def get_model_data(model):

    # start number of diet specialists
    start_num_w = model.data['total_num_w'][0]
    start_num_s = model.data['total_num_s'][0]

    # final number of diet specialists
    final_num_w = model.data['total_num_w'][-1]
    final_num_s = model.data['total_num_s'][-1]

    # mean foraging time of diet specialists
    final_mean_foraging_w = np.mean(model.data['mean_foraging_w'])
    final_mean_foraging_s = np.mean(model.data['mean_foraging_s'])

    # get mean weight
    final_mean_weight_w = np.mean(model.data['mean_weight_w'])
    final_mean_weight_s = np.mean(model.data['mean_weight_s'])

    # get final weight (for sensitivity analysis)
    end_mean_weight_w = model.data['mean_weight_w'][-1]
    end_mean_weight_s = model.data['mean_weight_s'][-1]

    # get deviation from refweight throughout simulation (mean of mean sum of squares)
    mean_sumsq_weight_w = np.mean(model.data['mean_sum_squares_weight_w'])
    mean_sumsq_weight_s = np.mean(model.data['mean_sum_squares_weight_s'])
    return [start_num_w, start_num_s, final_num_w, final_num_s, final_mean_foraging_w, final_mean_foraging_s,
            final_mean_weight_w, final_mean_weight_s, end_mean_weight_w, end_mean_weight_s, mean_sumsq_weight_w,
            mean_sumsq_weight_s]  # also return dict with variable names

def run_model(i):

    # run parameters
    start_year = 2017

    # initiate and run model
    model = initiate_model(start_year, i)
    model.run_model()

    model_output = get_model_data(model)

    print("Finishing sensitivity run {} out of {}".format(i + 1, N))
    return model_output


def create_extra_parameter_set(standard_params):
    params = standard_params.copy()

    # parameters of interest
    vars = ['minimum_weight',
            'relative_threshold',
            'w_cockle_foraging_mean',
            'w_macoma_foraging_mean',
            'w_worm_foraging_efficiency',
            's_mussel_foraging_mean',
            's_cockle_foraging_mean',
            's_macoma_foraging_mean',
            'agg_factor_mudflats',
            'agg_factor_bed']

    # get standard parameter values
    params = {var: params[var] for var in vars}
    vals = list(params.values())

    # create param set (full set with N reps for every set)
    param_sets = []

    # ranges for parameter values
    min_weight = np.arange(400, 510, 10)
    rel_thres = np.arange(0, 1.6, 0.1)
    w_cockle_for = np.arange(0.5, 1.6, 0.1)
    w_mac_for = np.arange(0.5, 1.6, 0.1)
    w_worm_for = np.arange(0.5, 1.6, 0.1)
    s_mussel_for = np.arange(0.5, 1.6, 0.1)
    s_cockle_for = np.arange(0.5, 1.6, 0.1)
    s_mac_for = np.arange(0.5, 1.6, 0.1)
    agg_factor_mud = np.arange(0, 55, 5)
    agg_factor_bed = np.arange(0, 55, 5)

    # for each parameter create new parameter set
    for value in min_weight:
        new_set = vals.copy()
        new_set[0] = value
        for j in range(N):
            param_sets.append(new_set)

    # and for all the other parameters
    for value in rel_thres:
        new_set = vals.copy()
        new_set[1] = value
        for j in range(N):
            param_sets.append(new_set)

    for value in w_cockle_for:
        new_set = vals.copy()
        new_set[2] = value
        for j in range(N):
            param_sets.append(new_set)

    for value in w_mac_for:
        new_set = vals.copy()
        new_set[3] = value
        for j in range(N):
            param_sets.append(new_set)

    for value in w_worm_for:
        new_set = vals.copy()
        new_set[4] = value
        for j in range(N):
            param_sets.append(new_set)

    for value in s_mussel_for:
        new_set = vals.copy()
        new_set[5] = value
        for j in range(N):
            param_sets.append(new_set)

    for value in s_cockle_for:
        new_set = vals.copy()
        new_set[6] = value
        for j in range(N):
            param_sets.append(new_set)

    for value in s_mac_for:
        new_set = vals.copy()
        new_set[7] = value
        for j in range(N):
            param_sets.append(new_set)

    for value in agg_factor_mud:
        new_set = vals.copy()
        new_set[8] = value
        for j in range(N):
            param_sets.append(new_set)

    for value in agg_factor_bed:
        new_set = vals.copy()
        new_set[9] = value
        for j in range(N):
            param_sets.append(new_set)

    # this is our final set!
    final_param_set = np.array(param_sets)

    # return the set as well as the value keys
    print(vars, final_param_set)
    return vars, final_param_set


# get parameter set
vars, param_set_vals = create_extra_parameter_set(standard_params)

if __name__ == '__main__':

    # run the model in parallel
    starttime = time.time()
    pool = multiprocessing.Pool(processes=50)
    results = pool.map(run_model, range(len(param_set_vals)))
    pool.close()
    print('That took {} seconds'.format(time.time() - starttime))

    # save results for analysis
    np.savetxt(fname, np.array(results))

