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

        # set parameters #todo: zet in param file
        self.init_birds = params["init_birds"]
        self.resolution_min = 30 # time step size # todo: calculate with input data

        # prey characteristics todo: in parameter file
        self.AFDWenergyContent = 22.5 # kJ/gram
        self.RatioAFDWtoWet = 0.16 # afdw per gram wet weight for cockles and mussel
        self.FractionTakenUp = 0.85  # Speakman1987, KerstenVisser1996, KerstenPiersma1987, ZwartsBlomert1996
        self.LeftOverShellfish = 0.1 # ZwartEnsKerstenetal1996
        self.CockFWtoSizeA = 14.610 # Ens, Webtics, L = a FW ^ b(mm = a gram ^b)
        self.CockFWtoSizeB = 0.317766

        # threshold to leave patch
        self.leaving_threshold = 20.45685 # IR at which 623g bird with mean efficientie needs 12 hours of foraging J/s

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

        # mussel patches
        self.mussel_density = 9999999  # infinitely rich mussel patches
        self.mussel_wtw_gain = -0.0025 / (24 / (self.resolution_min / 60))  # wtw gain per time step, GossCustard2001
        self.mussel_afdw = 0.850  # g AFDW GossCustard2001 # todo: dit moet veranderen
        self.mussel_wet_weight = self.mussel_afdw / self.RatioAFDWtoWet  # g WtW calculated with conversion factor
        # self.mussel_intake_rates =

        # array with number of agents on every patch
        self.num_agents_on_patches = np.zeros(self.num_patches, dtype=int) #todo: misschien overbodig?
        self.agents_on_patches = [[] for _ in range(self.num_patches)] #todo: kan dit misschien sneller? met arrays?

        # use schedule from schedule.py that randomly activates agents
        self.schedule = RandomActivation(self)

        # todo: datacollector here
        self.data = defaultdict(list)

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
        self.cockle_sizes = None
        self.handling_time_cockles = None

        self.mussel_potential_wtw_intake, self.mussel_potentional_energy_intake = [None, None]
        self.mudflats_potential_wtw_intake, self.mudflats_potential_energy_intake, self.capture_rates_mudflats \
            = [None, None, None]

        # create birds
        for i in range(self.init_birds):

            # give random initial position #todo: should be according to ideal distribution
            pos = 2 # todo: maak dit anders. Zorg ervoor dat er duidelijker onderscheid is tussen mossel/mudflats

            # give agent individual properties
            unique_id = self.next_id() # every agent has unique id
            dominance = np.random.randint(1, 101) # todo: should be taken from distribution/data

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

        # get new waterheight and patch availability
        self.waterheight = self.waterheight_data[time_step]

        # calculate available area for every patch
        self.available_areas = self.df_patch_availability.loc[self.waterheight].values * self.patch_areas

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

            # calculate new fresh weight cockles

            # calculate wet weight cockles (g)

            # calculate new mussel weight

            # calculate new size cockles (mm) with formula that relates fresh weight to length and handling time
            self.cockle_sizes = self.CockFWtoSizeA * (self.cockle_fresh_weight ** self.CockFWtoSizeB)
            self.handling_time_cockles = self.calculate_handling_time_cockles(self.cockle_sizes)


            # todo: misschien als we geen interferentie meenemen hier de intake rate voor mudflats berekenen?
            # todo: sowieso voor elke patch de non-interference IR berekenen (in plaats van in agents)

        # calculate intake rate for mussel patches (without interference)
        self.mussel_potential_wtw_intake, self.mussel_potentional_energy_intake = self.mussel_potential_intake()

        # calculate intake rate for mudflats (without interference)
        self.mudflats_potential_wtw_intake, self.mudflats_potential_energy_intake, self.capture_rates_mudflats \
            = self.mudflats_potential_intake()

        # execute model.step (move agents and let them eat) todo: pas schedule aan
        self.schedule.step()
        self.time_in_cycle += 1 # time STEPS in cycle

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

    def mussel_potential_intake(self):  #todo: in agent zelf opnieuw E intake berekenen op basis v beschikbare maaginhoud
        """
        Calculates final potential intake on mussel patches (interference thus excluded) for one time step.

        :return: potential wtw intake (g) and energy intake (kJ)
        """

        # calculate capture rate
        capture_rate = self.functional_response_mussel(self.mussel_density, self.mussel_afdw)

        # wet intake rate
        intake_wtw = capture_rate * self.mussel_wet_weight  # g WtW/s
        intake_wtw *= (1 - self.LeftOverShellfish)

        # get total capture rate/IRs in one time step todo: dit kan iets netter
        conversion_s_to_timestep = self.resolution_min * 60
        total_intake_wtw = intake_wtw * conversion_s_to_timestep  # g/time step

        # calculate potential energy intake
        energy_intake = total_intake_wtw * self.FractionTakenUp * self.RatioAFDWtoWet \
                        * self.AFDWenergyContent
        return total_intake_wtw, energy_intake

    @staticmethod
    def functional_response_mussel(mussel_density, mussel_afdw):
        """
        Functional response as described in WEBTICS. They converted
        the intake of stillman to a capture rate.

        :param prey_weight: ash free dry mass weight of mussels (no size classes) in g
        :return: capture rate in # prey/ s
        """

        # parameters todo: in parameter file
        attack_rate = 0.00057  # mosselA in webtics
        mussel_intake_rate_A = 0.092  # parameters for max intake rate (plateau)
        mussel_intake_rate_B = 0.506

        # calculate plateau of functional response mussel #todo: is de IR niet gewoon dit plateau?
        max_intake_rate = mussel_intake_rate_A * (mussel_afdw * 1000) ** mussel_intake_rate_B

        # calculate handling time and capture rate
        handling_time = (mussel_afdw * 1000) / max_intake_rate  # convert prey to mg
        capture_rate = (attack_rate * mussel_density) / (1 + attack_rate * handling_time * mussel_density)
        return capture_rate

    def mudflats_potential_intake(self):
        """
        Calculates final potential intake on mussel patches (interference thus excluded) for one time step.

        :return: potential wtw intake (g) and energy intake (kJ)
        """

        # calculate capture rate
        capture_rate_kok1, capture_rate_kok2, capture_rate_kokmj, capture_rate_mac \
            = self.functional_response_mudflats(self.handling_time_cockles, self.cockle_densities,
                                                self.handling_time_macoma, self.macoma_density,
                                                self.proportion_macoma)

        # wet weight intake rate (g/s)
        patch_wet_intake = capture_rate_kok1 * self.cockle_wet_weight[:, 0] \
                           + capture_rate_kok2 * self.cockle_wet_weight[:, 1] \
                           + capture_rate_kokmj * self.cockle_wet_weight[:, 2] \
                           + capture_rate_mac * self.macoma_wet_weight
        patch_wet_intake *= (1 - self.LeftOverShellfish)

        # convert to intake rate of one time step
        conversion_s_to_timestep = self.resolution_min * 60  # todo: dubbel
        total_intake_wtw = patch_wet_intake * conversion_s_to_timestep

        # calculate potential energy intake
        energy_intake = total_intake_wtw * self.FractionTakenUp * self.RatioAFDWtoWet \
                        * self.AFDWenergyContent

        # calculate total captured prey in one time step
        total_captured_prey = [capture_rate_kok1 * conversion_s_to_timestep,
                               capture_rate_kok2 * conversion_s_to_timestep,
                               capture_rate_kokmj * conversion_s_to_timestep,
                               capture_rate_mac * conversion_s_to_timestep]
        return total_intake_wtw, energy_intake, total_captured_prey

    @staticmethod
    def functional_response_mudflats(handling_time_cockles, cockle_densities, handling_time_mac, mac_density,
                                     proportion_macoma):
        """
        Functional response as described in webtics.

        :return: capture rate in # prey/s for all different prey types
        """

        # get density and size of all cockle size classes on patch #todo: dit kan op nettere manier uitgepakt
        kok1_handling_time = handling_time_cockles[:, 0]
        kok2_handling_time = handling_time_cockles[:, 1]
        kokmj_handling_time = handling_time_cockles[:, 2]
        kok1_density = cockle_densities[:, 0]
        kok2_density = cockle_densities[:, 1]
        kokmj_density = cockle_densities[:, 2]

        # parameters todo: zet in parameter file
        leoA = 0.000860373  # Zwarts et al. (1996b), taken from WEBTICS
        leoB = 0.220524  # Zwarts et al.(1996b)
        hiddinkA = 0.000625  # Hiddink2003
        attack_rate = leoA * leoB

        # calculate capture rate for every cockle size class (number of cockles/s)
        capture_rate_kok1_num = attack_rate * kok1_density  # numerator of eq 5.9 webtics
        capture_rate_kok1_den = attack_rate * kok1_handling_time * kok1_density  # denominator without 1 +
        capture_rate_kok2_num = attack_rate * kok2_density
        capture_rate_kok2_den = attack_rate * kok2_handling_time * kok2_density
        capture_rate_kokmj_num = attack_rate * kokmj_density
        capture_rate_kokmj_den = attack_rate * kokmj_handling_time * kokmj_density

        # capture rate macoma
        capture_rate_mac_num = hiddinkA * mac_density * proportion_macoma  # only take available macoma into account
        capture_rate_mac_den = capture_rate_mac_num * handling_time_mac

        # final denominator 5.9 webtics
        final_denominator = 1 + capture_rate_kok1_den + capture_rate_kok2_den + capture_rate_kokmj_den \
                            + capture_rate_mac_den

        # calculate number of captured prey for each size class
        capture_rate_kok1 = (capture_rate_kok1_num / final_denominator)
        capture_rate_kok2 = (capture_rate_kok2_num / final_denominator)
        capture_rate_kokmj = (capture_rate_kokmj_num / final_denominator)
        capture_rate_mac = capture_rate_mac_num / final_denominator
        return capture_rate_kok1, capture_rate_kok2, capture_rate_kokmj, capture_rate_mac

