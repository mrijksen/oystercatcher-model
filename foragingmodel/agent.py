# class Agent:
#     """ Base class for a model agent. Taken from mesa. """
#     def __init__(self, unique_id, model):
#         """ Create a new agent. """
#         self.unique_id = unique_id
#         self.model = model
#
#     def step(self):
#         """ A single step of the agent. """
#         pass
#
#     @property
#     def random(self):
#         return self.model.random
import numpy as np

class Bird:
    """
    Instantiations represent oystercatchers foraging according to some
    optimisation rule
    """

    def __init__(self, unique_id, pos, model, dominance, energy=None):
        self.unique_id = unique_id
        self.model = model
        self.dominance = dominance
        self.energy = energy
        self.pos = pos

        # array with route for coming time steps
        self.route = np.zeros(model.get_steps(model.num_tidal_cycles, model.minutes_in_tidal_cycle,
                                              model.resolution_min))

        # number of encounters won
        self.L = None

    def step(self): # todo:
        """A model step. Move, then eat. """

        # # get current location
        # print("position: ", self.pos)

        # get new location

        #( calculate actual ir todo: willen we deze baseren op route of opnieuw? )

        # reduce energy

        # eat prey

        # (update patch todo: should this be here? or do we substract sum of all ir's?)

        # apply death





