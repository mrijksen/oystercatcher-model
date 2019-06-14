"""
Foraging model

Created by: Marleen Rijksen
"""

from schedule import RandomActivation
from agent import Bird
from model import Model

from collections import defaultdict
import numpy as np
import random

np.seterr(divide='ignore', invalid='ignore')


class OystercatcherModel(Model):

    def __init__(self, params, df_patch_data, df_patch_availability, df_env):
        """ Create a new model with given parameters

        :param params: file containing all model parameters
        :param df_patch_data: contains all patch information (prey densities, area)
        :param patch availability: contains fraction of patch available for every waterheight
        :param df_env: all environmental data (temeprature, waterheight, prey growth, reference weight)
        """
        super().__init__()

        # SA parameters
        self.relative_threshold = params["relative_threshold"]
        self.agg_factor_mudflats = params["agg_factor_mudflats"]
        self.agg_factor_bed = params["agg_factor_bed"]

        # scenario analysis parameters
        self.temperature_change = params["temperature_change"]
        self.waterlevel_change = params["waterlevel_change"]
        self.mussel_density_change = params["mussel_density_change"]
        self.cockle_density_change = params["cockle_density_change"]
        self.macoma_density_change = params["macoma_density_change"]

        # get data files
        self.patch_data = df_patch_data
        self.df_env = df_env
        self.df_patch_availability = df_patch_availability

        self.params = params

        # set parameters
        self.init_birds = params["init_birds"]
        self.resolution_min = params["resolution_min"] # time step size # todo: calculate with input data

        # prey characteristics
        self.AFDWenergyContent = params["AFDWenergyContent"] # kJ/gram
        self.RatioAFDWtoWet = params["RatioAFDWtoWet"] # afdw per gram wet weight for cockles and mussel
        self.CockFWtoSizeA = params["CockFWtoSizeA"] # Ens, Webtics, L = a FW ^ b(mm = a gram ^b)
        self.CockFWtoSizeB = params["CockFWtoSizeB"]

        self.FractionTakenUp = params["FractionTakenUp"]  # Speakman1987, KerstenVisser1996, KerstenPiersma1987, ZwartsBlomert1996
        self.LeftOverShellfish = params["LeftOverShellfish"]  # ZwartEnsKerstenetal1996

        self.leoA = params["leoA"]
        self.leoB = params["leoB"]  # Zwarts et al.(1996b)
        self.leoC = params["leoC"]

        self.hiddinkA = params["hiddinkA"]
        self.hiddinkB = params["hiddinkB"]

        self.attack_rate = params["attack_rate"]
        self.mussel_intake_rate_A = params["mussel_intake_rate_A"]
        self.mussel_intake_rate_B = params["mussel_intake_rate_B"]

        # threshold to leave patch
        self.leaving_threshold = params["leaving_threshold"] * self.relative_threshold

        # calculate number of time steps in total
        self.num_steps = df_env.shape[0]

        # patches characteristics
        self.patch_ids = df_patch_data.patchID.tolist()
        self.patch_types = df_patch_data.type.values
        self.patch_areas = df_patch_data.area.values
        self.num_patches = df_patch_data.shape[0]

        # patches for shellfish/wormspecialists
        self.patch_indices_mudflats = np.where(self.patch_types == "Mudflat")[0]
        self.patch_indices_beds = np.where(self.patch_types == "Bed")[0]
        self.patch_max_bed_index = np.max(self.patch_indices_beds)
        self.patch_index_grassland = np.where(self.patch_types == "Grassland")[0]

        # cockle data
        self.cockle_fresh_weight = df_patch_data[['Cockle_1j_FW',
                                                  'Cockle_2j_FW',
                                                  'Cockle_mj_FW']].values # [0] = 1j, [1] = 2j, [3] = mj
        self.cockle_wet_weight = df_patch_data[['Cockle_1j_WW',
                                                'Cockle_2j_WW',
                                                'Cockle_mj_WW']].values
        self.cockle_densities = df_patch_data[['Cockle_1j_dens',
                                               'Cockle_2j_dens',
                                               'Cockle_mj_dens']].values * self.cockle_density_change

        # macoma data
        self.macoma_wet_weight = df_patch_data['Macoma_WW'].values
        self.macoma_density = df_patch_data['Macoma_dens'].values * self.macoma_density_change
        self.handling_time_macoma = self.calculate_handling_time_macoma()  # does not change during simulation

        # mussel data
        self.mussel_density = params["mussel_density"] * self.mussel_density_change# infinitely rich mussel patches
        self.mussel_wtw_gain = params["mussel_wtw_gain"] / (24 / (self.resolution_min / 60))  # wtw gain per time step
        self.mussel_afdw = params["mussel_afdw"]  # g, at 1 september
        self.mussel_wet_weight = self.mussel_afdw / self.RatioAFDWtoWet  # g WtW calculated with conversion factor
        # self.mussel_intake_rates =

        # array with number of agents on every patch
        self.num_agents_on_patches = np.zeros(self.num_patches, dtype=int) #todo: misschien overbodig?
        self.agents_on_patches = [[] for _ in range(self.num_patches)] #todo: kan dit misschien sneller? met arrays?

        # use schedule from schedule.py that randomly activates agents
        self.schedule = RandomActivation(self)

        # todo: datacollector here, think about all data we want to collect
        self.data = defaultdict(list)
        self.cockle_fresh_weight_list = []
        self.cockle_wet_weight_list = []
        self.mussel_weight_list = []
        self.cockle_sizes_list = []

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
        self.day_night_data = df_env['day_night'].tolist()
        self.date_time_data = df_env['date_time']

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
        self.available_areas = None
        self.day_night = None

        # intake rate variables
        self.mussel_potential_wtw_intake, self.mussel_potential_energy_intake = [None, None]
        self.mudflats_potential_wtw_intake, self.mudflats_potential_energy_intake, self.energy_intake_cockle, \
        self.energy_intake_mac, self.capture_rates_mudflats = [None, None, None, None, None]
        self.grassland_potential_wtw_intake, self.grassland_potential_energy_intake = self.grassland_potential_intake()

        # positions for specialists
        possible_positions_worm = np.concatenate((self.patch_indices_mudflats, self.patch_index_grassland)) # todo grass?
        possible_positions_shellfish = np.concatenate((self.patch_indices_mudflats, self.patch_indices_beds))

        # create birds todo: gebruik hier de verdeling van HK voor de vogels, en maak een lijst met alle vogels
        for i in range(self.init_birds):

            # give agent individual properties
            unique_id = self.next_id() # every agent has unique id

            # choose sex with proportions males/females and worm/shellfish
            sex = np.random.choice(['male', 'female'], p=[0.55, 0.45])
            if sex == 'male':
                specialization = np.random.choice(['worm', 'shellfish'], p=[0.23, 0.77])
            elif sex == 'female':
                specialization = np.random.choice(['worm', 'shellfish'], p=[0.66, 0.34])

            # dominance
            dominance = np.random.randint(1, 101)

            # foraging efficiency and position for specializations for different prey
            if specialization == 'worm':
                mussel_foraging_efficiency = self.params["w_mussel_foraging_efficiency"]
                cockle_foraging_efficiency = np.random.normal(self.params["w_cockle_foraging_mean"], 0.1)
                macoma_foraging_efficiency = np.random.normal(self.params["w_macoma_foraging_mean"], 0.1)
                worm_foraging_efficiency = np.random.normal(self.params["w_worm_foraging_efficiency"], 0.1)
                pos = np.random.choice(possible_positions_worm)
            elif specialization == 'shellfish':
                mussel_foraging_efficiency = np.random.normal(self.params["s_mussel_foraging_mean"], 0.1)
                cockle_foraging_efficiency = np.random.normal(self.params["s_cockle_foraging_mean"], 0.1)
                macoma_foraging_efficiency = np.random.normal(self.params["s_macoma_foraging_mean"], 0.1)
                worm_foraging_efficiency = self.params["s_worm_foraging_efficiency"]
                pos = np.random.choice(possible_positions_shellfish)

            # instantiate bird class
            bird = Bird(unique_id, pos, self, dominance, specialization, mussel_foraging_efficiency,
                        cockle_foraging_efficiency, macoma_foraging_efficiency, worm_foraging_efficiency)

            # place and add to schedule
            self.num_agents_on_patches[pos] += 1
            self.schedule.add(bird)

    def step(self):

        # current time step
        time_step = self.schedule.time

        # get new waterheight and patch availability
        self.waterheight = self.waterheight_data[time_step] + self.waterlevel_change

        # date time
        self.date_time = self.date_time_data[time_step]

        # check day or night
        self.day_night = self.day_night_data[time_step]

        # calculate available area for every patch
        self.available_areas = self.df_patch_availability.loc[self.waterheight].values * self.patch_areas

        # calculate new mussel weight
        self.mussel_wet_weight += self.mussel_wet_weight * self.mussel_wtw_gain
        self.mussel_afdw = self.mussel_wet_weight * self.RatioAFDWtoWet

        # check if new tidal cycle starts
        if self.extreem_data[time_step] == 'HW':

            # get new parameters from data file todo: is this needed?
            self.time_in_cycle = 0
            self.new_tidal_cycle = True
            self.reference_weight_birds = self.weight_data[time_step]
            self.total_number_steps_in_cycle = self.steps_in_cycle_data[time_step]
            self.steps_to_low_tide = self.steps_low_tide_data[time_step]
            self.temperature = self.temperature_data[time_step] + self.temperature_change
            self.proportion_macoma = self.proportion_macoma_data[time_step]

            # calculate new fresh weight cockles for 1y and 2y cockles
            self.cockle_fresh_weight[:, 0] += self.cockle_fresh_weight[:, 0] * self.one_y_fw_cockle_gr_data[time_step]
            self.cockle_fresh_weight[:, 1] += self.cockle_fresh_weight[:, 1] * self.two_y_fw_cockle_gr_data[time_step]

            # calculate new wet weight cockles for all year classes (g)
            self.cockle_wet_weight[:, 0] += self.cockle_wet_weight[:, 0] * self.one_y_wtw_cockle_gr_data[time_step]
            self.cockle_wet_weight[:, [1, 2]] += self.cockle_wet_weight[:, [1, 2]] * \
                                                 self.two_y_wtw_cockle_gr_data[time_step]

            # calculate new size cockles (mm) with formula that relates fresh weight to length and handling time
            self.cockle_sizes = self.CockFWtoSizeA * (self.cockle_fresh_weight ** self.CockFWtoSizeB)
            self.handling_time_cockles = self.calculate_handling_time_cockles(self.cockle_sizes)



        # calculate intake rate for mussel patches (without interference)
        self.mussel_potential_wtw_intake, self.mussel_potential_energy_intake = self.mussel_potential_intake()

        # calculate intake rate for mudflats (without interference)
        self.mudflats_potential_wtw_intake, self.mudflats_potential_energy_intake, self.energy_intake_cockle, \
        self.energy_intake_mac, self.capture_rates_mudflats \
            = self.mudflats_potential_intake()



        # execute model.step (move agents and let them eat)
        self.schedule.step()

        # at the end of the tidal cycle todo: check if position is correct
        if self.time_in_cycle == self.total_number_steps_in_cycle - 1:
            self.collect_data()

        self.time_in_cycle += 1 # time STEPS in cycle



        self.new_tidal_cycle = False
        # print(self.num_agents_on_patches)

    def run_model(self):
        """ Run the model for the time steps indicated in the data set
        """

        print("Initial number birds: {}".format(self.init_birds))

        # simulate for given number of num_steps
        for i in range(self.num_steps):  # todo: geef hier aantal stappen in df mee
            self.step()

        print("Final number of birds: {}".format(self.schedule.get_agent_count()))

    def collect_data(self):

        # list of all agents
        worm_specialists = [agent for agent in self.schedule.agents if agent.specialization == 'worm']
        shellfish_specialists = [agent for agent in self.schedule.agents if agent.specialization == 'shellfish']

        # calculate number of agents
        self.data['total_num_w'].append(len(worm_specialists))
        self.data['total_num_s'].append(len(shellfish_specialists))
        self.data['total_num_agents'].append(len(self.schedule.agents))

        # calculate mean weight of diet specialization groups #todo: if no agents left, mean = 0
        mean_weight_w = np.mean([agent.weight for agent in worm_specialists])
        mean_weight_s = np.mean([agent.weight for agent in shellfish_specialists])
        mean_weight_w_std = np.std([agent.weight for agent in self.schedule.agents if agent.specialization == 'worm'])
        mean_weight_s_std = np.std([agent.weight for agent in self.schedule.agents if agent.specialization == 'shellfish'])
        self.data['mean_weight_w'].append(mean_weight_w)
        self.data['mean_weight_s'].append(mean_weight_s)
        self.data['mean_weight_w_std'].append(mean_weight_w_std)
        self.data['mean_weight_s_std'].append(mean_weight_s_std)

        # calculate mean foraging time (in hours)
        mean_foraging_w = np.mean([agent.time_foraged for agent in worm_specialists]) * self.resolution_min / 60
        mean_foraging_s = np.mean([agent.time_foraged for agent in shellfish_specialists]) * self.resolution_min / 60
        mean_foraging_w_std = np.std(np.array([agent.time_foraged for agent in worm_specialists]) * self.resolution_min / 60)
        mean_foraging_s_std = np.std(np.array([agent.time_foraged for agent in shellfish_specialists]) * self.resolution_min / 60)
        self.data['mean_foraging_w'].append(mean_foraging_w)
        self.data['mean_foraging_s'].append(mean_foraging_s)
        self.data['mean_foraging_w_std'].append(mean_foraging_w_std)
        self.data['mean_foraging_s_std'].append(mean_foraging_s_std)

        # calculate deviation from reference weight (mean sum of squares)
        self.data['mean_sum_squares_weight_w'].append(np.sum([((agent.weight - self.reference_weight_birds) ** 2)
                                                              for agent in worm_specialists]) / len(worm_specialists))
        self.data['mean_sum_squares_weight_s'].append(np.sum([((agent.weight - self.reference_weight_birds) ** 2)
                                                              for agent in shellfish_specialists]) / len(shellfish_specialists))

        # visited positions
        self.data['worm_positions'].append([agent.pos for agent in worm_specialists])
        self.data['shellfish_positions'].append([agent.pos for agent in shellfish_specialists])

        # foraging times (for histogram)
        self.data['foraging_times_w'].append([agent.time_foraged for agent in worm_specialists])
        self.data['foraging_times_s'].append([agent.time_foraged for agent in shellfish_specialists])

        # todo: deze ook in defaultdict opslaan
        self.cockle_fresh_weight_list.append(self.cockle_fresh_weight[:, 0][-1])
        self.cockle_wet_weight_list.append(self.cockle_wet_weight[:, 0][-1])
        self.mussel_weight_list.append(self.mussel_wet_weight)
        self.cockle_sizes_list.append(self.cockle_sizes[:, 0][-1])

    def calculate_handling_time_cockles(self, cockle_size):
        """ Helper method to calculate the handling time for each cockle size class
        :param cockle_size: size of cockle in mm
        """
        return self.leoB * (cockle_size ** self.leoC)

    def calculate_handling_time_macoma(self): # todo: gewicht voor macoma balthica verschilt per patch
        """ Helper method to calculate handling time for macoma balthica.

            Currently this handling time is only based on the initial weight of macoma balthica.

            The input should be given in g! not mg.
        """
        return (self.hiddinkB / self.hiddinkA) * (1000 * self.macoma_wet_weight * self.RatioAFDWtoWet)

    def mussel_potential_intake(self):  #todo: in agent zelf opnieuw E intake berekenen op basis v beschikbare maaginhoud
        """
        Calculates final potential intake on mussel patches (interference thus excluded) for one time step.

        :return: potential wtw intake (g) and energy intake (kJ) per time step
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

    def functional_response_mussel(self, mussel_density, mussel_afdw):
        """
        Functional response as described in WEBTICS. They converted
        the intake of stillman to a capture rate.

        :param prey_weight: ash free dry mass weight of mussels (no size classes) in g
        :return: capture rate in # prey/ s
        """

        # calculate plateau of functional response mussel #todo: is de IR niet gewoon dit plateau?
        max_intake_rate = self.mussel_intake_rate_A * (mussel_afdw * 1000) ** self.mussel_intake_rate_B

        # calculate handling time and capture rate
        handling_time = (mussel_afdw * 1000) / max_intake_rate  # convert prey to mg
        capture_rate = (self.attack_rate * mussel_density) / (1 + self.attack_rate * handling_time * mussel_density)
        return capture_rate

    def mudflats_potential_intake(self):
        """
        Calculates final potential intake excluding interference on mussel patches for one time step.

        :return: potential wtw intake (g/time step) and energy intake (kJ/time step)
        """

        # calculate capture rate
        capture_rate_kok1, capture_rate_kok2, capture_rate_kokmj, capture_rate_mac \
            = self.functional_response_mudflats(self.handling_time_cockles, self.cockle_densities,
                                                self.handling_time_macoma, self.macoma_density,
                                                self.proportion_macoma)

        # wet weight intake rate (g/s)
        patch_wet_intake_cockle_sec = (capture_rate_kok1 * self.cockle_wet_weight[:, 0] \
                           + capture_rate_kok2 * self.cockle_wet_weight[:, 1] \
                           + capture_rate_kokmj * self.cockle_wet_weight[:, 2]) * (1 - self.LeftOverShellfish) #todo check dit
        patch_wet_intake_mac_sec = (capture_rate_mac * self.macoma_wet_weight) * (1 - self.LeftOverShellfish)

        # convert to intake rate of one time step
        conversion_s_to_timestep = self.resolution_min * 60  # todo: dubbel
        total_intake_wtw = (patch_wet_intake_cockle_sec + patch_wet_intake_mac_sec) * conversion_s_to_timestep

        # for cockle and macoma only
        total_intake_wtw_cockle = patch_wet_intake_cockle_sec * conversion_s_to_timestep
        total_intake_wtw_mac = patch_wet_intake_mac_sec * conversion_s_to_timestep

        # calculate potential energy intake excluding interference
        energy_intake = total_intake_wtw * self.FractionTakenUp * self.RatioAFDWtoWet \
                        * self.AFDWenergyContent # todo: moet fractiontakenup hier wel?
        energy_intake_cockle = total_intake_wtw_cockle * self.FractionTakenUp * self.RatioAFDWtoWet \
                        * self.AFDWenergyContent
        energy_intake_mac = total_intake_wtw_mac * self.FractionTakenUp * self.RatioAFDWtoWet \
                        * self.AFDWenergyContent

        # calculate total captured prey in one time step
        total_captured_prey = [capture_rate_kok1 * conversion_s_to_timestep,
                               capture_rate_kok2 * conversion_s_to_timestep,
                               capture_rate_kokmj * conversion_s_to_timestep,
                               capture_rate_mac * conversion_s_to_timestep]
        return total_intake_wtw, energy_intake, energy_intake_cockle, energy_intake_mac, total_captured_prey


    def functional_response_mudflats(self, handling_time_cockles, cockle_densities, handling_time_mac, mac_density,
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

        # calculate capture rate for every cockle size class (number of cockles/s)
        capture_rate_kok1_num = self.leoA * kok1_density  # numerator of eq 5.9 webtics
        capture_rate_kok1_den = self.leoA * kok1_handling_time * kok1_density  # denominator without 1 +
        capture_rate_kok2_num = self.leoA * kok2_density
        capture_rate_kok2_den = self.leoA * kok2_handling_time * kok2_density
        capture_rate_kokmj_num = self.leoA * kokmj_density
        capture_rate_kokmj_den = self.leoA * kokmj_handling_time * kokmj_density

        # capture rate macoma
        capture_rate_mac_num = self.hiddinkA * mac_density * proportion_macoma  # only take available macoma into account
        capture_rate_mac_den = capture_rate_mac_num * handling_time_mac

        # final denominator 5.9 webtics
        final_denominator = 1 + capture_rate_kok1_den + capture_rate_kok2_den + capture_rate_kokmj_den \
                            + capture_rate_mac_den

        # calculate number of captured prey for each size class
        capture_rate_kok1 = capture_rate_kok1_num / final_denominator
        capture_rate_kok2 = capture_rate_kok2_num / final_denominator
        capture_rate_kokmj = capture_rate_kokmj_num / final_denominator
        capture_rate_mac = capture_rate_mac_num / final_denominator
        return capture_rate_kok1, capture_rate_kok2, capture_rate_kokmj, capture_rate_mac

    def grassland_potential_intake(self):
        """ Method that lets agent forage on grassland patch. Based on the energy goal and the stomach content
        the intake of an agent is evaluated.

        The patch depletion is also implemented.

        Returns the wet weight consumed (g) and the energy consumed (kJ) per time step.
        """

        # parameters
        conversion_afdw_wtw = self.params["conversion_afdw_wtw"]

        # intake from Stillman (also used in webtics)
        afdw_intake_grassland = (0.53 * 60 * self.resolution_min) / 1000 # g / time step, Stillman2000

        # wtw intake
        wtw_intake = afdw_intake_grassland / conversion_afdw_wtw # g / time step

        # calculate energy intake
        energy_intake = (afdw_intake_grassland * self.AFDWenergyContent)  # kJ
        return wtw_intake, energy_intake

