
# These classes were derived from mesa github

from collections import OrderedDict


class BaseScheduler:
    """ Simplest scheduler, activates agents one at a time, in the order
    they were added.

    Assumes that each agent has a *step* method which takes no arguments

    (This is explicitly meant to replicate the scheduler in MASON).
    """

    def __init__(self, model):
        """ Create a new, empty BaseScheduler"""
        self.model = model
        self.steps = 0
        self.time = 0
        self._agents = OrderedDict

    def add(self, agent):
        """ Add an Agent object to the schedule.
        Args:
            agent: An Agent to be added to the schedule. NOTE: The agent must
            have a step() method.
        """
        self._agents[agent.unique_id] = agent

    def remove(self, agent):
        """ Remove all instances of a given agent from the schedule.
        Args:
            agent: An agent object.
        """
        del self._agents[agent.unique_id]

    def step(self):
        """ Execute the step of all the agents, one at a time. """
        for agent in self.agent_buffer(shuffled=False):
            agent.step()
        self.steps += 1
        self.time += 1

    def get_agent_count(self):
        """ Returns the current number of agents in the queue. """
        return len(self._agents.keys())

    @property
    def agents(self):
        return list(self._agents.values())

    def agent_buffer(self, shuffled=False):
        """ Simple generator that yields the agents while letting the user
        remove and/or add agents during stepping.
        """
        agent_keys = list(self._agents.keys())
        if shuffled:
            self.model.random.shuffle(agent_keys)

        for key in agent_keys:
            if key in self._agents:
                yield self._agents[key]


class RandomActivation(BaseScheduler):
    """ A scheduler which activates each agent once per step, in random order,
    with the order reshuffled every step.
    This is equivalent to the NetLogo 'ask agents...' and is generally the
    default behavior for an ABM.
    Assumes that all agents have a step(model) method.
    """

    def step(self):
        """ Executes the step of all agents, one at a time, in
        random order.
        """ #todo: is this necessary?
        for agent in self.agent_buffer(shuffled=True):
            agent.step()
        self.steps += 1
        self.time += 1