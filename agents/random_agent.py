# A set of agents for environment debugging purposes only.
from aasma.agent import Agent
from aasma.snake_environment.snake_environment import Action, PRE_IDS
from gym.utils import seeding

import agents.utils as utils

import math

class RandomAgent(Agent):
    """
    Blindly, randomly
    """
    def __init__(self, seed=None):
        super(RandomAgent, self).__init__("Monkey Random Agent")
        self._rng, _ = seeding.np_random(seed)

    def action(self, observation) -> int:
        return math.floor(self._rng.uniform(-1, 2))

class LessDumbRandomAgent(Agent):
    def __init__(self, seed=None):
        super(LessDumbRandomAgent, self).__init__("Less Dumb Random Agent")
        self._rng, _ = seeding.np_random(seed)
    
    def action(self, observation) -> int:
        # Where are we?
        
        # Will anything kill us? (compute the space excluding actions that will guaranteedly kill us)
        posValid = []
        for action in Action:
            aux = utils.sum(observation["agents"][observation["self"]][0], observation["directions"][utils.fixDirection(action.value, observation, observation["self"])])
            if utils.wall(aux, observation["grid_shape"]):
                continue
            obs = observation["grid"][aux[0]][aux[1]]
            if obs != PRE_IDS['empty'] and obs != PRE_IDS['food']:
                continue
            else:
                posValid += [action.value]

        if len(posValid) == 0:
            # We're likely going to die anyway
            return Action.FORWARD.value
        elif len(posValid) == 1:
            return posValid[0]
        else:
            return posValid[math.floor(self._rng.uniform(0, len(posValid)))]
