"""
Foraging model
================================================

"""

from schedule import RandomActivation
from agent import Bird
from model import Model

import random

import data

from collections import defaultdict
import numpy as np


class OystercatcherModel(Model):

    def __init__(self, params, df_patch_data, df_patch_availability, df_env):
        """ Create a new model with given parameters

        :param params: file containing all model parameters
        :param df_patch_data: contains all patch information (prey densities, area)
        :param patch availability: contains fraction of patch available for every waterheight
        :param df_env: all environmental data (temeprature, waterheight, prey growth, reference weight)
        """
        super().__init__()

        # get data files
        self.patch_data = df_patch_data
        self.df_env = df_env
        self.df_patch_availability = df_patch_availability

        # set parameters #todo: zet sommige dingen in param file
        self.init_birds = params["init_birds"]
        self.resolution_min = 30 # time step size # todo: calculate with input data

        # prey characteristics todo: in parameter file
        self.AFDWenergyContent = 22.5 # kJ/gram
        self.RatioAFDWtoWet = 0.16 # afdw per gram wet weight for cockles and mussel
        self.FractionTakenUp = 0.85  # Speakman1987, KerstenVisser1996, KerstenPiersma1987, ZwartsBlomert1996
        self.LeftOverShellfish = 0.1 # ZwartEnsKerstenetal1996
        self.CockFWtoSizeA = 14.610 # Ens, Webtics, L = a FW ^ b(mm = a gram ^b)
        self.CockFWtoSizeB = 0.317766

        # calculate number of time steps in total
        self.num_steps = df_env.shape[0]

        # patches characteristics
        self.patch_ids = df_patch_data.patchID.tolist()
        self.patch_types = df_patch_data.type.tolist()
        self.patch_areas = df_patch_data.area.values
        self.num_patches = df_patch_data.shape[0]

        # cockle data todo: dit moet geupdate worden elke tidal
        self.cockle_fresh_weight = df_patch_data[['Cockle_1j_FW',
                                                  'Cockle_2j_FW',
                                                  'Cockle_mj_FW']].values # [0] = 1j, [1] = 2j, [3] = mj
        self.cockle_wet_weight = df_patch_data[['Cockle_1j_WW',
                                                'Cockle_2j_WW',
                                                'Cockle_mj_WW']].values
        self.cockle_densities = df_patch_data[['Cockle_1j_dens',
                                               'Cockle_2j_dens',
                                               'Cockle_mj_dens']].values

        # macoma data
        self.macoma_wet_weight = df_patch_data['Macoma_WW'].values
        self.macoma_density = df_patch_data['Macoma_dens'].values
        self.handling_time_macoma = self.calculate_handling_time_macoma()  # does not change during simulation

        # mussel data
        self.musselcover = df_patch_data['musselcover'].values # todo: add cover to capture rate/interference
        self.mussel_density = 999999 # infinite mussel density

        # array with number of agents on every patch
        self.num_agents_on_patches = np.zeros(self.num_patches, dtype=int) #todo: misschien overbodig?
        self.agents_on_patches = [[] for _ in range(self.num_patches)] #todo: kan dit misschien sneller? met arrays?

        # use schedule from schedule.py that randomly activates agents
        self.schedule = RandomActivation(self)

        # todo: datacollector here
        self.data = defaultdict(list)

        # mussel patches
        self.mussel_density = 9999999 # infinitely rich mussel patches
        self.mussel_wtw_gain = -0.0025 / (24 / (self.resolution_min / 60)) # wtw gain per time step, GossCustard2001
        self.mussel_afdw = 0.850 # g AFDW GossCustard2001 # todo: dit moet veranderen
        self.mussel_wet_weight = self.mussel_afdw / self.RatioAFDWtoWet # g WtW calculated with conversion factor

        # create lists of environmental data input
        self.temperature_data = df_env['temperature'].tolist()
        self.weight_data = df_env['weight'].tolist()
        self.waterheight_data = df_env['waterheight'].tolist()
        self.steps_in_cycle_data = df_env['time_steps_in_cycle'].tolist()
        self.steps_low_tide_data = df_env['time_steps_to_low_tide'].tolist()
        self.extreem_data = df_env['extreem'].tolist()
        self.one_y_fw_cockle_gr_data = df_env['1y_fw_cockle_growth'].tolist()
        self.two_y_fw_cockle_gr_data = df_env['2y_fw_cockle_growth'].tolist()
        self.one_y_wtw_cockle_gr_data = df_env['1y_wtw_cockle_growth'].tolist()
        self.two_y_wtw_cockle_gr_data = df_env['2y_wtw_cockle_growth'].tolist()
        self.proportion_macoma_data = df_env['proportion_macoma'].tolist()

        # these parameters we get from the environmental data input
        self.time_in_cycle = None
        self.total_number_steps_in_cycle = None
        self.waterheight = None
        self.steps_to_low_tide = None
        self.temperature = None
        self.proportion_macoma = None
        self.reference_weight_birds = None
        self.new_tidal_cycle = None # boolean to check if new cycle starts

        #
        self.cockle_sizes = None
        self.handling_time_cockles = None

        # create birds
        for i in range(self.init_birds):

            # give random initial position #todo: should be according to ideal distribution
            # pos = random.randrange(self.num_patches + 1)
            pos = 2

            # give agent individual properties
            unique_id = self.next_id() # every agent has unique id
            dominance = unique_id # todo: should be taken from distribution/data

            # instantiate class
            bird = Bird(unique_id, pos, self, dominance)

            # add agent to agent overview
            self.agents_on_patches[bird.pos].append(bird)

            # place and add to schedule todo: place agent according to ideal distribution
            self.num_agents_on_patches[pos] += 1
            self.schedule.add(bird)

    def step(self):

        # current time step
        time_step = self.schedule.time

        # check if new tidal cycle starts
        if self.extreem_data[time_step] == 'HW':

            # get new parameters from data file
            self.time_in_cycle = 0
            self.new_tidal_cycle = True
            self.reference_weight_birds = self.weight_data[time_step]
            self.total_number_steps_in_cycle = self.steps_in_cycle_data[time_step]
            self.steps_to_low_tide = self.steps_low_tide_data[time_step]
            self.temperature = self.temperature_data[time_step]
            self.proportion_macoma = self.proportion_macoma_data[time_step]

            # calculate wet weight mussels with self.mussel_wtw_gain # todo: dit moet per patch

            # calculate new fresh weight cockles with extrapolation

            # calculate wet weight cockles (g)

            # calculate new size cockles (mm) with formula that relates fresh weight to length and handling time
            self.cockle_sizes = self.CockFWtoSizeA * (self.cockle_fresh_weight ** self.CockFWtoSizeB)
            self.handling_time_cockles = self.calculate_handling_time_cockles(self.cockle_sizes)

            # # calculate handling time cockles todo: voor alle patches
            # self.handling_time_cockles = []
            # for size in self.cockle_sizes:
            #     self.handling_time_cockles.append(self.calculate_handling_time_cockles(size))



            # todo: misschien als we geen interferentie meenemen hier de intake rate voor mudflats berekenen?
            # todo: sowieso voor elke patch de non-interference IR berekenen?

        # get new waterheight and patch availability
        waterheight = self.waterheight_data[time_step]

        # calculate available area for every patch
        available_area = self.df_patch_availability.loc[waterheight].values * self.patch_areas

        # execute model.step (move agents and let them eat) todo: pas schedule aan
        self.schedule.step()
        self.time_in_cycle += 1

        # maak paar datacollectie functies en roep die aan
        # todo maak defaultdict for data collection

        self.new_tidal_cycle = False

    def run_model(self):
        """ Run the model for the time steps indicated in the data set
        """

        print("Initial number birds: {}".format(self.init_birds))

        # simulate for given number of num_steps
        for i in range(self.num_steps):  # todo: geef hier aantal stappen in df mee
            self.step()

        print("Final number of birds: {}".format(self.schedule.get_agent_count()))

    @staticmethod
    def calculate_handling_time_cockles(cockle_size):
        """ Helper method to calculate the handling time for each cockle size class
        :param cockle_size: size of cockle in mm
        """
        # parameters
        # leoA = 0.000860373# Zwarts et al. (1996b), taken from WEBTICS
        leoB = 0.220524 # Zwarts et al.(1996b)
        leoC = 1.79206
        return leoB * (cockle_size ** leoC)

    def calculate_handling_time_macoma(self): # todo: gewicht voor macoma balthica verschilt per patch
        """ Helper method to calculate handling time for macoma balthica.

            Currently this handling time is only based on the initial weight of macoma balthica.

            The input should be given in g! not mg.
        """

        # parameters
        hiddinkA = 0.000625     # by Hiddink2003
        hiddinkB = 0.000213     # handling parameter (Hiddink2003)
        return (hiddinkB / hiddinkA) * (1000 * self.macoma_wet_weight * self.RatioAFDWtoWet)


    # @staticmethod
    # def get_steps(num_tidal_cycles, minutes_in_tidal_cycle, resolution_min): #todo: dit doen we dubbel (ook in data.py)
    #     """Helper method to calculate number of steps based on number of tidal cycles
    #     and resolution of model.
    #
    #     Note that this can be changed to something more realistic (e.g, varying tidal cycles)
    #     """
    #     return int((num_tidal_cycles * minutes_in_tidal_cycle) / resolution_min)

    # @staticmethod #todo: this should be L on current patch, not in total system! should we put this in agents.py?
    # def calculate_L(total_num_agents, dominance):
    #     """ Returns total number of encounters won (in percentages) based on number
    #     of agents currently in system and an agent's dominance"""
    #     if total_num_agents > 1:
    #         return (total_num_agents - dominance) * 100 / (total_num_agents - 1)
    #     else:
    #         return 100 #todo: should this be 100?

    # # tidal cycle parameters and total number of model steps
    # self.num_tidal_cycles = params["num_tidal_cycles"]

    # self.minutes_in_tidal_cycle = params["minutes_in_tidal_cycle"] # minutes in tidal cycle, 720 = 12 hours