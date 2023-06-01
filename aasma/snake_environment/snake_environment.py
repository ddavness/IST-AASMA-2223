import copy
import logging
import random
import copy

import math
import numpy as np
import functools

logger = logging.getLogger(__name__)

from PIL import ImageColor
import gym
from gym import spaces
from gym.utils import seeding

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace

from enum import Enum
class Action(Enum):
    LEFT = -1
    FORWARD = 0
    RIGHT = 1

class SnakeEnvironment(gym.Env):

    """
    A modified version of ma_gym.envs.predator_prey.predator_prey.PredatorPrey to fit a snake game

    grid_shape: (x, y) - The shape of the grid, how big is it? Must be at least 10x10
    n_agents: int - The number of players/snakes
    max_steps: int - The maximum number of moves in the game. Defaults to x * y
    food_equilibrium: (int) -> float - The target number of food points in the environment. Defaults to (3/2)*area^(2/3)
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __food_equilibrium_default(self, area):
        return 3/2 * area^(2/3)

    def __init__(self, grid_shape=(10, 10), n_agents=2, max_steps=None, food_equilibrium=None):
        if grid_shape[0] < 10 or grid_shape[1] < 10:
            raise ValueError("Grid must be at least 10x10")

        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._max_steps = max_steps if max_steps is not None else grid_shape[0] * grid_shape[1]
        self._food_equilibrium_f = food_equilibrium if food_equilibrium is not None else self.__food_equilibrium_default
        self._step_count = None

        self.action_space = MultiAgentActionSpace([spaces.Discrete(3) for _ in range(self.n_agents)])
        self.agent_body = {_: None for _ in range(self.n_agents)}

        self._grid = self.__create_grid()
        self.viewer = None
        
        self.alive = None
        self.body = {_: [] for _ in range(self.n_agents)}  # self.body= a list of all the bodys from agents than are a list on its own[(x1,y1),(x2,y2)]
        self.head_directions = [(1, 0), (0, -1), (-1, 0), (0, 1)] # Turning right will
        self.direction = {_: random.randint(0, 3) for _ in range(self.n_agents)} # (1, 0), (0, -1), (-1, 0), (0, 1)
        self.food = []

        self._obs_high = np.tile([1.0] * np.prod(grid_shape), self.n_agents)
        self._obs_low = np.tile([0.0] * np.prod(grid_shape), self.n_agents)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self.seed()

    def simplified_features(self):
        current_grid = np.array(self._grid)

        self.body = {_: [] for _ in range(self.n_agents)}
        for agent_id in range(self.n_agents):
            if self.alive[agent_id]:
                tag = f"H{agent_id + 1}"
                row, col = np.where(current_grid == tag)
                row = row[0]
                col = col[0]
                self.body[agent_id].append((col, row))

        features = np.array(self.body).reshape(-1)

        return features

    def reset(self):
        self.body = {}
        self.prey_pos = {}

        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self.alive = [True for _ in range(self.n_agents)]
        
        self.get_agent_obs()
        return [self.simplified_features() for _ in range(self.n_agents)]
    
    def __regenerate_food(self):
        area_total = self._grid_shape[0] * self._grid_shape[1]
        grid_players = list(functools.reduce(lambda a, b: a + b, self.agent_body))
        area_available = area_total - len(grid_players)
        food_target = self._food_equilibrium_f(area_available)
        if len(self.food) > food_target:
            return
        sample = self.np_random.uniform(0, food_target)
        if sample > food_target - len(self.food):
            return
        else:
            # Spawn a food block in any random free space
            while True:
                x = math.floor(self.np_random.uniform(3, self._grid_shape[0] - 3 + 1))
                y = math.floor(self.np_random.uniform(3, self._grid_shape[1] - 3 + 1))
                if self._grid[x][y] != PRE_IDS['empty']:
                    continue
                else:
                    self.food += [(x, y)]
                    self._grid[x][y] = PRE_IDS['food']
                    return

    def create_snakes(self):
        for a in self.agent_body:
            x = math.floor(self.np_random.uniform(3, self._grid_shape[0] + 1))
            y = math.floor(self.np_random.uniform(3, self._grid_shape[1] + 1))
            a.append((x, y))

    def step(self, agents_action):
        self._step_count += 1

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)

        if (self._step_count >= self._max_steps) or (sum(self.alive)<2):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        self.get_agent_obs()

        self.__regenerate_food()

        return [self.simplified_features() for _ in range(self.n_agents)], self._agent_dones

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __init_full_obs(self):
        self._grid = self.__create_grid()

        for agent_i in range(self.n_agents):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                       self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos):
                    self.body[agent_i] = pos
                    break
            self.__update_agent_view(agent_i)

        self.__draw_base_img()

    def get_agent_obs(self):
        # All agents have full observation
        return np.array(self._grid).flatten().tolist()

    def __body_exists(self, pos):
        row, col = pos
        return PRE_IDS['body'] in self._grid[row, col] or PRE_IDS['head'] in self._grid[row, col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._grid[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __update_agent_pos(self, agent_i, move):
        curr_pos = copy.copy(self.body[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.body[agent_i] = next_pos
            self._grid[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __update_agent_view(self, agent_i):
        self._grid[self.body[agent_i][0]][self.body[agent_i][1]] = PRE_IDS['head'] + str(agent_i + 1)

    def _dispose_agent(self, agent_i):
        self.alive[agent_i] = False

    def _check_colisions(self):
        toKill = []
        def schedule_kill(i):
            if not i in toKill:
                toKill += [i]
        for agent_i in range(self.n_agents):
            if self.alive[agent_i]:
                if (self.body[agent_i][0] in self.body[agent_i][1:]):
                    # Snake collided with it's own body
                    schedule_kill(agent_i)
                if not self.is_valid(self.body[agent_i][0]):
                    # Snake went out of bounds
                    schedule_kill(agent_i)
                for agent_x in range(self.n_agents):
                    if self.alive[agent_x]:
                        if agent_i == agent_x:
                            continue
                        if (self.body[agent_i][0] in self.body[agent_x]):
                            # Snake i collided with x
                            schedule_kill(agent_i)
                        if (self.body[agent_x][0] in self.body[agent_i]):
                            # Snake x collided with i (both are possible)
                            schedule_kill(agent_x)
        for agent in toKill:
            self.alive[agent] = False
        for agent in toKill:
            self._dispose_agent(agent)

    def render(self, mode='human'):
        img = copy.copy(self._base_img)
        for agent_i in range(self.n_agents):
            for cell in self.body[agent_i]:
                fill_cell(img, cell, cell_size=CELL_SIZE, fill=self.getColor(agent_i, body=True), margin=0.1)

        for agent_i in range(self.n_agents):
            draw_circle(img, self.body[agent_i][0], cell_size=CELL_SIZE, fill=self.getColor(agent_i))
            write_cell_text(img, text=str(agent_i + 1), pos=self.body[agent_i][0], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def getColor(self, agent_i, body=False):
        color = ImageColor.getcolor(AGENT_COLOR[agent_i], mode='HSV')
        if(body):
            color = color + (200,)
        return color 

AGENT_COLOR = ["red","blue","green","yellow","purple","orange","pink","brown"] #8 agent colors
FOOD_COLOR = "black"

CELL_SIZE = 35

WALL_COLOR = 'black'

PRE_IDS = {
    'head': 'H',
    'food': 'F',
    'body': 'B',
    'empty': '0'
}
