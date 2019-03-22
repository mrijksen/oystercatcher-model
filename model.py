"""
Foraging model
================================================

"""

from schedule import RandomActivation
from agent import Bird

import time
import random


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

    def __init__(self, init_prey, availability, temperature, init_birds, mussel):
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
        # self.temperature = temperature todo: later toevoegen
        self.init_birds = init_birds
        self.mussel = mussel

        # use schedule from schedule.py
        self.schedule = RandomActivation(self)

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
        print("new model step")

        # if we reach a new tidal cycle:
        #       - we should calculate a new route for all agents
        #       - we have to update prey decline

        # execute model.step (move agents and let them eat)
        self.schedule.step()

        # collect data on patches and agents

    def run_model(self, step_count=200):

        print("Initial number birds: {}".format(self.init_birds))

        # simulate
        for i in range(step_count):
            print(i)
            self.step()

        print("Final number of birds: {}".format(self.schedule.get_agent_count()))

    def route_calculation(self):
        """ Here we should include the calculation of agent routes """

        # todo: misschien meerdere functies maken voor verschillende routetypes?
        # todo: hoe werkt inheritance hier?










