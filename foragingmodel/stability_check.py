import data
from oystercatchermodel import OystercatcherModel
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing

fname = "../results/stability_standardparams.txt"
N = 5   # number of model runs in total
out = 6  # number of model results

def initiate_model(start_year):

    """ Instantiate model class """

    # get parameters
    params = data.get_params()

    # load real patch data
    df_patch_data = data.get_patch_data(start_year)

    # get patchIDs
    patchIDs = df_patch_data.patchID.values

    # load patch availability
    df_patch_availability = data.get_patch_availability(start_year, patchIDs)

    # load environmental data
    df_env = data.get_environmental_data(start_year)

    # instantiate model
    model = OystercatcherModel(params, df_patch_data, df_patch_availability, df_env)
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
    return [start_num_w, start_num_s, final_num_w, final_num_s, final_mean_foraging_w, final_mean_foraging_s]

def run_model(i):

    # run parameters
    start_year = 2017

    # initiate and run model
    model = initiate_model(start_year)
    model.run_model()

    model_output = get_model_data(model)

    print("Finishing sensitivity run {} out of {}".format(i + 1, N))
    return model_output


if __name__ == '__main__':

    # run the model in parallel
    starttime = time.time()
    pool = multiprocessing.Pool(processes=8)
    results = pool.map(run_model, range(N))
    pool.close()
    print('That took {} seconds'.format(time.time() - starttime))

    # save results for analysis
    np.savetxt(fname, np.array(results))