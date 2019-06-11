import data
from oystercatchermodel import OystercatcherModel
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import multiprocessing

start_year = 2017

def initiate_model(start_year):
    """ Instantiate model class """

    # get parameters todo: dit ook in path zetten?
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

def run_model(i):

    # initiate and run model
    model = initiate_model(start_year)
    model.run_model()
    model_output = model.data

    print("Finishing single run {} out of {}".format(i + 1, 5))
    return model_output


if __name__ == "__main__":

    # run the model in parallel
    starttime = time.time()
    pool = multiprocessing.Pool(processes=5)
    results = pool.map(run_model, range(5))
    pool.close()
    print('That took {} seconds'.format(time.time() - starttime))

    # save results in file
    output = open('../results/single_simulation_runs/{}_10000agents_standardparams.pkl'.format(start_year), 'wb')
    pickle.dump(results, output)
    output.close()
