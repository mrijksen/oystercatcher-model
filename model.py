"""
Foraging model
================================================

"""

from schedule import RandomActivation
from agent import Bird

import time
import random

import numpy as np

from numba import njit

class Model:
    """ Base class for models.

    **Taken from mesa. **
     """

    def __new__(cls, *args, **kwargs):
        """Create a new model object and instantiate its RNG automatically."""

        model = object.__new__(cls)  # This only works in Python 3.3 and above
        model._seed = time.time()
        if "seed" in kwargs and kwargs["seed"] is not None:
            model._seed = kwargs["seed"]
        model.random = random.Random(model._seed)
        return model

    def __init__(self):
        """ Create a new model. Overload this method with the actual code to
        start the model.
        Attributes:
            schedule: schedule object
            running: a bool indicating if the model should continue running
        """

        self.running = True
        self.schedule = None
        self.current_id = 0

    def run_model(self):
        """ Run the model until the end condition is reached. Overload as
        needed.
        """
        while self.running:
            self.step()

    def step(self):
        """ A single step. Fill in here. """
        pass

    def next_id(self):
        """ Return the next unique ID for agents, increment current_id"""
        self.current_id += 1
        return self.current_id

    def reset_randomizer(self, seed=None):
        """Reset the model random number generator.
        Args:
            seed: A new seed for the RNG; if None, reset using the current seed
        """


class OystercatcherModel(Model):

    def __init__(self, init_prey, availability, temperature, init_birds, num_tidal_cycles, mussel, resolution_min=10,
                 minutes_in_tidal_cycle=720):
        """ Create a new model with given parameters
        :param init_prey: list with initial prey on patches #todo: divide in diff prey
        :param availability: array with availability on all patches for all t
        :param temperature: global temperature for all t
        :param init_birds: number of agents to start with
        :param mussel: boolean to indicate if forage is on or off
        """
        super().__init__()

        # set parameters #todo: zet sommige dingen in param file
        self.prey = init_prey
        self.availability = availability
        self.init_birds = init_birds
        self.mussel = mussel

        # self.temperature = temperature todo: later toevoegen

        # tidal cycle parameters and total number of model steps
        self.num_tidal_cycles = num_tidal_cycles
        self.resolution_min = resolution_min # time step size
        self.minutes_in_tidal_cycle = minutes_in_tidal_cycle # minutes in tidal cycle, 720 = 12 hours
        self.num_steps = self.get_steps(num_tidal_cycles, self.minutes_in_tidal_cycle, self.resolution_min)

        # use schedule from schedule.py
        self.schedule = RandomActivation(self)

        self.steps_per_tidal_cycle = self.get_steps(1, self.minutes_in_tidal_cycle, self.resolution_min)

        # todo: datacollector here

        # create birds
        for i in range(self.init_birds):

            # give random position #todo: should be according to ideal distribution
            pos = random.randrange(len(self.prey))

            unique_id = self.next_id()
            dominance = unique_id # todo: should be taken from distribution/data

            # initial energy
            energy = 10 #todo: should be taken from distr/data

            # instantiate class
            bird = Bird(unique_id, pos, self, dominance, energy)

            # place and add to schedule todo: place agent on something
            self.schedule.add(bird)

    def step(self):
        # print("\nNew model step")

        # if we reach a new tidal cycle #todo: this can be done more elegant
        if self.schedule.time % (self.minutes_in_tidal_cycle / self.resolution_min) == 0:
            start = time.time()
            self.route_calculation()
            print("Route time", time.time() - start)

        # - we have to update prey decline for all time steps

        # execute model.step (move agents and let them eat) todo: pas schedule aan
        self.schedule.step()

        # collect data on patches and agents

    def run_model(self):

        print("Initial number birds: {}".format(self.init_birds))

        # simulate
        for i in range(self.num_steps):
            # print(i)
            self.step()

        print("Final number of birds: {}".format(self.schedule.get_agent_count()))

    def route_calculation(self):
        """ Here we should include the calculation of agent routes """

        num_agents_on_patches = np.zeros([self.num_steps, len(self.prey)], dtype=int)
        total_num_agents = len(self.schedule.agents)

        # iterate over all agents
        for agent in self.schedule.agents: #todo: is this in random order?

            # calculate number of encounters won
            agent.L = self.calculate_L(total_num_agents, agent.dominance)

            # # calculate route with for loops
            # self.calculate_route_agent_for_loops(agent, num_agents_on_patches)

            m0 = 0.6
            m1 = -0.008

            # calculate route with matrix multiplication for time steps

            self.calculate_route_agent_matrices(agent, num_agents_on_patches, m0, m1)

            # self.calculate_route_agent_for_loops(agent, num_agents_on_patches, m0, m1)



    @staticmethod
    def get_steps(num_tidal_cycles, minutes_in_tidal_cycle, resolution_min):
        """Helper method to calculate number of steps based on number of tidal cycles
        and resolution of model.

        Note that this can be changed to something more realistic (e.g, varying tidal cycles)
        """
        return int((num_tidal_cycles * minutes_in_tidal_cycle) / resolution_min)

    def calculate_route_agent_for_loops(self, agent, num_agents_all_patches, m0, m1):
        """Method to calculate route based on number of agents on patches,
        current prey on patches and availability.

        In this case, we use for loops to iterate over time.

        Updates route for one agent.

        In case a patch is not available, IR becomes zero

        :param agent: agent we are currently calculating route for.
        :param num_agents_all_patches: number of agent on all patches for all time steps
        """
        test_list = np.zeros(self.steps_per_tidal_cycle)
        # iterate over time steps in one tidal cycle
        for step in range(self.steps_per_tidal_cycle):

            # apply formulas for intake rate
            available_prey = self.prey * self.availability[step]  # unavailable prey patches become zero
            intake_rate_all_patches = available_prey * ((num_agents_all_patches[step] + 1) ** (-m0 - m1 * agent.L))

            # get index of patch with maximal IR
            index_patch_max_intake = np.argmax(intake_rate_all_patches)

            # change num_agents_all_patches
            num_agents_all_patches[step, index_patch_max_intake] += 1

            # add new position to agent's route
            test_list[step] = index_patch_max_intake
        # return test_list
        # print(test_list, "agent route for loops")

    def calculate_route_agent_matrices(self, agent, num_agents_all_patches, m0, m1):
        """ Calculate route for agent (for all timesteps via matrix multiplication to speed code up a bit.
        """
        # ONLY take availability array for one tidal !!!!! todo:: VERY IMPORTANT

        # get index patch max_intake
        available_prey = np.multiply(self.availability, [self.prey])

        # apply formula to all elements of array
        intake_rate_all_patches = np.multiply(available_prey, ((num_agents_all_patches + 1) ** (-m0 - m1 * agent.L)))

        # get indices for each time step with patch with maximal ir
        agent.route = intake_rate_all_patches.argmax(axis=1)

        # change num agents all patches
        num_agents_all_patches[np.arange(num_agents_all_patches.shape[0]), agent.route] += 1

        # print(num_agents_all_patches[0])
        #
        # print(agent.route)


    @staticmethod
    def calculate_L(total_num_agents, dominance):
        """ Returns total number of encounters won (in percentages) based on number
        of agents currently in system and an agent's dominance"""
        if total_num_agents > 1:
            return (total_num_agents - dominance) * 100 / (total_num_agents - 1)
        else:
            return 100 #todo: should this be 100?













