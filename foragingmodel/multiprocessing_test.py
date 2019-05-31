import data
from oystercatchermodel import OystercatcherModel
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing


def initiate_model(start_year, run_type='real_data'):
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

    # run parameters
    start_year = 2017

    # initiate and run model
    model = initiate_model(start_year)

    start_time = time.time()
    model.run_model()
    end_time = time.time()
    print(end_time - start_time, "TIME")
    print(len(model.schedule.agents), "number of agents left")


if __name__ == '__main__':

    starttime = time.time()
    pool = multiprocessing.Pool(processes=8)
    pool.map(run_model, range(0, 10))
    pool.close()
    print('That took {} seconds'.format(time.time() - starttime))