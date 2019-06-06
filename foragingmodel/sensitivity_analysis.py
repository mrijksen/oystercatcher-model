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
p = .25
fname = "../results/sensitivity_test.txt"
standard_params = toml.load("config_file.toml")


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


def create_parameter_set(standard_params):

    params = standard_params.copy()

    # delete parameters not included in SA
    del params['resolution_min']
    del params['init_birds']
    del params['w_mussel_foraging_efficiency']
    del params['s_worm_foraging_efficiency']
    del params['mussel_density']
    del params['relative_density']

    # list with parameter names we are going to change
    vars = list(params.keys())
    vals = list(params.values())

    # create param set (full set with N reps for every set)
    param_sets = []

    # for each parameter in the params
    for i in range(len(params)):

        # create new parameter set
        new_set = vals.copy()

        # increase by x%
        new_set[i] *= (1 + p)

        # add N times in parameter set
        for j in range(N):
            param_sets.append(new_set)

    # decrease by x%
    for i in range(len(params)):
        new_set = vals.copy()
        new_set[i] *= (1 - p)
        for k in range(N):
            param_sets.append(new_set)

    # this is our final set!
    final_param_set = np.array(param_sets)

    # return the set as well as the value keys
    return vars, final_param_set


# get parameter set
vars, param_set_vals = create_parameter_set(standard_params)

if __name__ == '__main__':

    # run the model in parallel
    starttime = time.time()
    pool = multiprocessing.Pool(processes=40)
    results = pool.map(run_model, range(len(param_set_vals)))
    pool.close()
    print('That took {} seconds'.format(time.time() - starttime))

    # save results for analysis
    np.savetxt(fname, np.array(results))

