import numpy as np
import pandas as pd
import toml

def get_params():
    """Loads parameters from toml file"""

    # config_model = "config_file.json"
    # return json.load(open(config_model))
    config_file = "config_file.toml"
    return toml.load(config_file)


def create_data_lists_env_data(df_env): # todo: this is unnessesary. also, just do list(df[column])
    """ Creates lists of input data to run the model with
    """
    temperature_data = [x for x in df_env.temperature]
    weight_data = [x for x in df_env.weight]

    waterheight_data = [x for x in df_env.waterheight]
    steps_in_cycle_data = [x for x in df_env.time_steps_in_cycle]
    steps_low_tide_data = [x for x in df_env.time_steps_to_low_tide]
    extreem_data = [x for x in df_env.extreem]

    one_y_fw_cockle_gr = [x for x in df_env['1y_fw_cockle_growth']]
    two_y_fw_cockle_gr = [x for x in df_env['2y_fw_cockle_growth']]
    one_y_wtw_cockle_gr = [x for x in df_env['1y_wtw_cockle_growth']]
    two_y_wtw_cockle_gr = [x for x in df_env['2y_wtw_cockle_growth']]

    proportion_macoma = [x for x in df_env.proportion_macoma]

    return temperature_data, weight_data, waterheight_data, steps_in_cycle_data, steps_low_tide_data, extreem_data,\
           one_y_fw_cockle_gr, two_y_fw_cockle_gr, one_y_wtw_cockle_gr, two_y_wtw_cockle_gr, proportion_macoma


def get_patch_data(start_year): #todo: add grass & roost patch
    """ Load data frame with patch info for specific year.

    Patch characteristics are patchID, type, area, densities of prey
    """
    path = 'C:/Users/Marleen/Documents/thesis project/Data zaken/Data/Patch data/Patch_Info_Vlieland_{}.csv'.format(start_year)
    df_patches = pd.read_csv(path, delimiter=",")

    # select columns to use
    columns = ['patchID', 'type', 'area', 'musselcover',
               'Cockle_1j_FW', 'Cockle_1j_WW', 'Cockle_1j_dens',
               'Cockle_2j_FW', 'Cockle_2j_WW', 'Cockle_2j_dens',
               'Cockle_mj_FW', 'Cockle_mj_WW', 'Cockle_mj_dens',
               'Macoma_WW', 'Macoma_dens']
    df_patches = df_patches[columns]

    # remove patches with only zero prey densities
    columns = columns[3:]
    df_patches = df_patches.fillna(0)
    df_patches = df_patches[(df_patches[columns].T != 0).any()]

    # sort on patch id and reset index
    df_patches = df_patches.sort_values('type')
    df_patches.reset_index(inplace=True, drop=True)
    return df_patches

def get_artificial_patch_data():
    """ Artificial patch data, for testing purposes
    """
    artificial_patches = pd.DataFrame()
    artificial_patches['patchID'] = [1, 2, 3, 4]
    artificial_patches['type'] = ['Bed', "Mudflat", "Grassland", "Mudflat"]
    artificial_patches['area'] = [1000, 10000, 10000, 10000]
    artificial_patches['musselcover'] = [100, 0, 0, 0]
    artificial_patches['Cockle_1j_dens'] = [0, 0, 0, 90]
    artificial_patches['Cockle_2j_dens'] = [0, 0, 0, 20]
    artificial_patches['Cockle_mj_dens'] = [0, 0, 0, 40]
    artificial_patches['Macoma_dens'] = [0, 0, 0, 30]
    artificial_patches['Cockle_1j_FW'] = [0, 5, 0, 5]
    artificial_patches['Cockle_2j_FW'] = [0, 5, 0, 5]
    artificial_patches['Cockle_mj_FW'] = [0, 10, 0, 5]
    artificial_patches['Cockle_1j_WW'] = [0, 1, 0, 1]
    artificial_patches['Cockle_2j_WW'] = [0, 1, 0, 1]
    artificial_patches['Cockle_mj_WW'] = [0, 5, 0, 5]
    artificial_patches['Macoma_WW'] = [0, 0.1, 0, 0.1]
    # artificial_patches['Macoma_dens'] = [0, 10, 0, 0]

    # sort and set index to patchID
    df_patches = artificial_patches.sort_values('type')
    df_patches.index = df_patches.patchID
    return df_patches


def get_environmental_data(start_year):
    """ Loads all environmental data
    """

    # location of environmental data todo: dit ook in data.py zetten? en alleen startjaar meegevem?
    env_data_dir = 'C:/Users/Marleen/Documents/thesis project/oystercatcher-model/Input data/'
    env_data_filename = '{}_9_1_to_{}_3_1.pkl'.format(start_year, start_year + 1)
    env_data_path = env_data_dir + env_data_filename
    df_env = pd.read_pickle(env_data_path)
    return df_env

def get_patch_availability(start_year, patchIDs): #todo: add grass roost patch
    """ Load patch availability data.

    The columns of this data frame are the patchIDs, the last column is the waterheight.

    The values indicate the fraction of the patch available for every waterheight.
    """

    # get data and remove unnessecary column
    path = 'C:/Users/Marleen/Documents/thesis project/Data zaken/Data/Patch data/Patch_Exposure_Vlieland_{}.csv'.\
        format(start_year)
    df_patch_availability_data = pd.read_csv(path, delimiter=",")
    del df_patch_availability_data['Unnamed: 0']

    # set waterheight as index
    df_patch_availability_data.set_index('waterheight')

    # only get relevant columns (patches with nonzero entries)
    df_patch_availability = df_patch_availability_data.iloc[:, patchIDs - 1] #todo: check this

    # todo: add patch availability voor graspatch (always 1)

    return df_patch_availability


def get_artificial_patch_availability():
    """ Creates artificial patch data for testing purposes
    """

    # make patches available if waterheight < 0
    artificial_availability = pd.DataFrame()
    artificial_availability['waterheight'] = np.arange(-300, 300, 1)
    # artificial_availability['1'] = 0
    artificial_availability['1'] = np.where(artificial_availability.waterheight < 0, 1, 0)
    artificial_availability['2'] = 1
    artificial_availability['3'] = np.where(artificial_availability.waterheight < 20, 1, 0)
    artificial_availability['4'] = np.where(artificial_availability.waterheight < 0, 1, 0)

    artificial_availability.set_index('waterheight', inplace=True)
    return artificial_availability


def get_part_of_environmental_data():
    """ Loads part of environmental data (path should be specified here) for testing purposes
    """

    # location of environmental data todo: dit ook in data.py zetten? en alleen startjaar meegevem?
    env_data_dir = 'C:/Users/Marleen/Documents/thesis project/oystercatcher-model/Input data/'
    env_data_filename = '2017_9_1_to_2018_3_1.pkl'
    env_data_path = env_data_dir + env_data_filename
    df_env = pd.read_pickle(env_data_path)
    return df_env
