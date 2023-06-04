# A set of agents for environment debugging purposes only.
from aasma.agent import Agent
from aasma.snake_environment.snake_environment import Action
from gym.utils import seeding
import math

class RandomAgent(Agent):
    def __init__(self, seed=None):
        super(RandomAgent, self).__init__("Monkey Random Agent")
        self._rng, _ = seeding.np_random(seed)
    
    def action(self) -> int:
        return math.floor(self._rng.uniform(-1, 2))
