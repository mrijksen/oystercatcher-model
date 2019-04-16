"""
Foraging model
================================================

"""

from schedule import RandomActivation
from agent import Bird
from model import Model

import random

from collections import defaultdict
import numpy as np


class OystercatcherModel(Model):

    def __init__(self, params, patch_name_list, prey_densities, availability, patch_areas):
        """ Create a new model with given parameters
        :param init_prey: list with initial prey on patches #todo: divide in diff prey
        :param availability: array with availability on all patches for all t
        :param temperature: global temperature for all t #todo: add this
        :param init_birds: number of agents to start with
        :param mussel: boolean to indicate if forage is on or off
        :param num_tidal_cycles: number of tidal cycles we want to simulate
        """
        super().__init__()

        # set parameters #todo: zet sommige dingen in param file
        self.prey = prey_densities
        self.availability = availability
        self.init_birds = params["init_birds"]
        self.mussel = params["mussel"]
        self.num_patches = params["num_patches"]

        # prey characteristics
        self.init_mussel_dry_weight = params["init_mussel_dry_weight"]
        self.init_mussel_wet_weight = params["init_mussel_wet_weight"]
        self.AFDWenergyContent = 22.5 # kJ/gram todo: in parameter file
        self.RatioAFDWtoWet = 0.16 # afdw per gram wet weight for cockles and mussel
        # self.RatioDrytoWetEarthworm = 0.175 # gram dry weight per wet weight for earthworms Heppleston1971
        # self.EarthwormDryEnergyContent = 3.778 * 4.1867  # kJ per g dry weight

        self.temperature = params["temperature"] #todo: moet in data set komen
        self.reference_weight_birds = params["reference_weight"] #todo: moet in data set komen

        # tidal cycle parameters and total number of model steps
        self.num_tidal_cycles = params["num_tidal_cycles"]
        self.resolution_min = params["resolution_min"] # time step size
        self.minutes_in_tidal_cycle = params["minutes_in_tidal_cycle"] # minutes in tidal cycle, 720 = 12 hours

        # calculate number of time steps in total and in one tidal cycle
        self.num_steps = self.get_steps(self.num_tidal_cycles, self.minutes_in_tidal_cycle, self.resolution_min)
        self.steps_per_tidal_cycle = self.get_steps(1, self.minutes_in_tidal_cycle, self.resolution_min)

        # Patches characteristics
        # array with number of agents on every patch
        self.num_agents_on_patches = np.zeros(self.num_patches, dtype=int) #todo: misschien overbodig?
        self.patch_name_list = patch_name_list # todo: dit moet als argument mee worden gegeven
        self.patch_areas = patch_areas
        self.agents_on_patches = [[] for _ in range(self.num_patches)] #todo: kan dit misschien sneller? met arrays?

        # keep track of time steps within current tidal cycle
        self.time_in_cycle = None

        # use schedule from schedule.py that randomly activates agents
        self.schedule = RandomActivation(self)

        # todo: datacollector here
        self.data = defaultdict(list)

        # cockle data
        self.cockle_sizes = params["cockle_sizes"] # in mm todo: deze waardes moeten veranderen door de tijd
        self.handling_time_cockles = []
        self.cockle_fresh_weight = params["cockle_fresh_weight"]
        self.cockle_wet_weight = params["cockle_wet_weight"]

        # create birds
        for i in range(self.init_birds):

            # give random initial position #todo: should be according to ideal distribution
            pos = random.randrange(self.num_patches)

            unique_id = self.next_id()
            dominance = unique_id # todo: should be taken from distribution/data

            # initial energy
            energy = 10 #todo: should be taken from distr/data

            # instantiate class
            bird = Bird(unique_id, pos, self, dominance)

            # add agent to agent overview
            self.agents_on_patches[bird.pos].append(bird)

            # place and add to schedule todo: place agent on something
            self.num_agents_on_patches[pos] += 1
            self.schedule.add(bird)

    def step(self):

        # update time step within cycle
        self.time_in_cycle = self.schedule.time % self.steps_per_tidal_cycle

        # calculate wet weight mussels with formula

        # calculate new fresh weight cockles with extrapolation

        # calculate new size cockles (mm) with formula that relates fresh weight to length

        # calculate wet weight cockles (g)

        # calculate handling time cockles
        self.handling_time_cockles = []
        for size in self.cockle_sizes:
            self.handling_time_cockles.append(self.calculate_handling_time_cockles(size))



        # todo: hier mossel kokkel gewicht updaten
        # todo: misschien als we geen interferentie meenemen hier de intake rate voor mudflats berekenen?

        # if self.schedule.time % self.steps_per_tidal_cycle == 0:
        #     print("New tidal cycle!", "schedule time: ", self.schedule.time)

        # print(self.time_in_cycle)
        # print("\nNew model step")
        # for i in range(self.num_patches):
            # print("#####Patch:{} ######".format(i))
            # print("prey:", self.prey[i])
            # print("num_agents:", self.num_agents_on_patches[i])

        # execute model.step (move agents and let them eat) todo: pas schedule aan
        self.schedule.step()

        # collect data on patches and agents
        # maak paar datacollectie functies en roep die aan
        # maak defaultdict

    def run_model(self):
        print("Initial number birds: {}".format(self.init_birds))

        # simulate for given number of num_steps
        for i in range(self.num_steps):
            # print("\nstep:", i, "hours passed: ", (i * 10/60))
            self.step()
        print("Final number of birds: {}".format(self.schedule.get_agent_count()))

    @staticmethod
    def get_steps(num_tidal_cycles, minutes_in_tidal_cycle, resolution_min): #todo: dit doen we dubbel (ook in data.py)
        """Helper method to calculate number of steps based on number of tidal cycles
        and resolution of model.

        Note that this can be changed to something more realistic (e.g, varying tidal cycles)
        """
        return int((num_tidal_cycles * minutes_in_tidal_cycle) / resolution_min)

    # @staticmethod #todo: this should be L on current patch, not in total system! should we put this in agents.py?
    # def calculate_L(total_num_agents, dominance):
    #     """ Returns total number of encounters won (in percentages) based on number
    #     of agents currently in system and an agent's dominance"""
    #     if total_num_agents > 1:
    #         return (total_num_agents - dominance) * 100 / (total_num_agents - 1)
    #     else:
    #         return 100 #todo: should this be 100?

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



