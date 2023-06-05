# A set of agents that move kind of randomly
from aasma.agent import Agent
from aasma.snake_environment.snake_environment import Action
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
        direction_ptr = observation["direction_ptr"][observation["self"]]
        # print(direction_ptr)
        actionValid = utils.get_valid_actions(direction_ptr, observation, observation["agents"][observation["self"]][0])
        if len(actionValid) == 0:
            # We're likely going to die anyway
            return Action.FORWARD.value
        elif len(actionValid) == 1:
            return actionValid[0]
        else:
            return actionValid[math.floor(self._rng.uniform(0, len(actionValid)))]
