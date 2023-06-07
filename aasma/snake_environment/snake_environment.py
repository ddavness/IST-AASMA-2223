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
        return 0.5 * area**(0.5)

    def __init__(self, grid_shape=(10, 10), n_agents=2, max_steps=None, food_equilibrium=None):
        if grid_shape[0] < 10 or grid_shape[1] < 10:
            raise ValueError("Grid must be at least 10x10")

        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._max_steps = max_steps if max_steps is not None else grid_shape[0] * grid_shape[1]
        self._food_equilibrium_f = food_equilibrium if food_equilibrium is not None else self.__food_equilibrium_default
        self._step_count = None

        self.action_space = MultiAgentActionSpace([spaces.Discrete(3) for _ in range(self.n_agents)])
        
        self._grid = self.__create_grid()
        self.viewer = None
        
        self.alive = [True for _ in range(self.n_agents)]
        self.body = {_: [] for _ in range(self.n_agents)}  # self.body= a list of all the bodys from agents than are a list on its own[(x1,y1),(x2,y2)]
        self.head_directions = [(1, 0), (0, 1), (-1, 0), (0, -1)] # Turning right will
        self.direction_ptr = {_: 0 for _ in range(self.n_agents)} # (1, 0), (0, -1), (-1, 0), (0, 1)
        self.food = []

        self._obs_high = np.tile([1.0] * np.prod(grid_shape), self.n_agents)
        self._obs_low = np.tile([0.0] * np.prod(grid_shape), self.n_agents)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self.seed()
        self.__draw_base_img()

    def simplified_features(self):
        #current_grid = np.array(self._grid).flatten()
        #print(current_grid)

        # self.body = {_: [] for _ in range(self.n_agents)}
        # for agent_id in range(self.n_agents):
        #     if self.alive[agent_id]:
        #         tag = f"H{agent_id + 1}"
        #         
        #         self.body[agent_id][0](col, row))

        features = np.array(self.body).reshape(-1)

        return features

    def reset(self):
        self.body = {_: [] for _ in range(self.n_agents)}
        self.direction_ptr = {_: 0 for _ in range(self.n_agents)}
        self.food = []
        self._toKill = []
        self._grid = self.__create_grid()

        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self.alive = [True for _ in range(self.n_agents)]
        
        return [self.simplified_features() for _ in range(self.n_agents)]
    
    def __regenerate_food(self):
        area_total = self._grid_shape[0] * self._grid_shape[1]
        player_area = 0
        for snake in self.body:
            player_area += len(self.body[snake])

        area_available = area_total - player_area
        food_target = self._food_equilibrium_f(area_available)
        if len(self.food) > food_target:
            return
        sample = self.np_random.uniform(0, food_target)
        if sample > food_target - len(self.food):
            return
        else:
            # Spawn a food block in any random free space
            while True:
                x = math.floor(self.np_random.uniform(0, self._grid_shape[0]))
                y = math.floor(self.np_random.uniform(0, self._grid_shape[1]))
                if self._grid[x][y] == PRE_IDS['empty']:
                    self.food += [(x, y)]
                    self._grid[x][y] = PRE_IDS['food']
                    return

    def create_snakes(self):
        for i in self.body:
            clear = False
            while not clear:
                x = math.floor(self.np_random.uniform(3, self._grid_shape[0] - 3))
                y = math.floor(self.np_random.uniform(3, self._grid_shape[1] - 3))
                dirptr = math.floor(self.np_random.uniform(0, 4))
                dir = self.head_directions[dirptr]
                self.direction_ptr[i] = dirptr
                if self._grid[x][y] == PRE_IDS['empty'] \
                and self._grid[x - dir[0]][y - dir[1]] == PRE_IDS['empty'] \
                and self._grid[x - 2 * dir[0]][y - 2 * dir[1]] == PRE_IDS['empty']:
                    self.body[i].append((x, y))
                    self.body[i].append((x - dir[0], y - dir[1]))
                    self.body[i].append((x - 2 * dir[0], y - 2 * dir[1]))
                    self.__update_agent_view(i)
                    clear = True

    def step(self, agents_action):
        self._step_count += 1

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)

        if (self._step_count >= self._max_steps) or (sum(self.alive)<2):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        self._check_collisions()

        for i in self.body:
            self.__update_agent_view(i)

        self.__regenerate_food()

        return [self.simplified_features() for _ in range(self.n_agents)], self._agent_dones

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __init_full_obs(self):
        self._grid = self.__create_grid()
        self.create_snakes()
        self.__draw_base_img()

    def get_agent_obs(self, agent_i):
        # All agents have full observation
        return {
            "self": agent_i,
            "grid": np.array(self._grid).tolist(),
            "grid_shape": self._grid_shape,
            "agents": {i: self.body[i] if self.alive[i] else [] for i in self.body},
            "directions": self.head_directions.copy(),
            "direction_ptr": {i: self.direction_ptr[i] if self.alive[i] else None for i in self.direction_ptr},
            "food": np.array(self.food).tolist()
        }

    def __body_exists(self, pos):
        row, col = pos
        return PRE_IDS['body'] in self._grid[row, col] or PRE_IDS['head'] in self._grid[row, col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._grid[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __update_agent_pos(self, agent_i, move):
        if not self.alive[agent_i]:
            return

        self.direction_ptr[agent_i] = (self.direction_ptr[agent_i] + move) % 4
        dir = self.head_directions[self.direction_ptr[agent_i]]
        # Next head position
        next_pos = (self.body[agent_i][0][0] + dir[0], self.body[agent_i][0][1] + dir[1])

        if next_pos is not None:
            if next_pos[0] < 0 or next_pos[0] >= self._grid_shape[0] or next_pos[1] < 0 or next_pos[1] >= self._grid_shape[1]:
                self.alive[agent_i] = False
                self._dispose_agent(agent_i)
                return
            elif self._grid[next_pos[0]][next_pos[1]] == PRE_IDS['food']:
                # We grow by one unit
                self.body[agent_i] = [next_pos] + self.body[agent_i]
                foodset = set(self.food)
                foodset.remove(next_pos)
                self.food = list(foodset)
                self._grid[next_pos[0]][next_pos[1]] = PRE_IDS['empty']
            else:
                tail = self.body[agent_i][-1]
                self._grid[tail[0]][tail[1]] = PRE_IDS['empty']
                try:
                    foodset = set(self.food)
                    foodset.remove(tail)
                    self.food = list(foodset)
                except:
                    pass
                self.body[agent_i] = [next_pos] + self.body[agent_i][:-1]
            self.__update_agent_view(agent_i)

    def __update_agent_view(self, agent_i):
        if self.alive[agent_i]:
            self._grid[self.body[agent_i][0][0]][self.body[agent_i][0][1]] = PRE_IDS['head'] + str(agent_i + 1)
            for b in self.body[agent_i][1:]:
                # print(b)
                self._grid[b[0]][b[1]] = PRE_IDS['body'] + str(agent_i + 1)

    def _dispose_agent(self, agent_i):
        print(f"Disposing agent {agent_i}")
        for bodypart in range(1, len(self.body[agent_i])):
            if self._grid[self.body[agent_i][bodypart][0]][self.body[agent_i][bodypart][1]] == PRE_IDS['head'] + str(agent_i + 1) \
            or self._grid[self.body[agent_i][bodypart][0]][self.body[agent_i][bodypart][1]] == PRE_IDS['body'] + str(agent_i + 1) \
            or self._grid[self.body[agent_i][bodypart][0]][self.body[agent_i][bodypart][1]] == PRE_IDS['empty']:
                self._grid[self.body[agent_i][bodypart][0]][self.body[agent_i][bodypart][1]] = PRE_IDS['empty']
            if bodypart % 2 == 1 and self._grid[self.body[agent_i][bodypart][0]][self.body[agent_i][bodypart][1]] == PRE_IDS['empty']:
                # print(self.body[agent_i][bodypart])
                self.food += [self.body[agent_i][bodypart]]
                self.food = list(set(self.food))
                self._grid[self.body[agent_i][bodypart][0]][self.body[agent_i][bodypart][1]] = PRE_IDS['food']

                
        self.alive[agent_i] = False

    def _schedule_kill(self, i):
        self._toKill += [i]

    def _check_collisions(self):
        for agent_i in range(self.n_agents):
            if self.alive[agent_i]:
                if (self.body[agent_i][0] in self.body[agent_i][1:]):
                    # Snake collided with it's own body
                    self._schedule_kill(agent_i)
                for agent_x in range(self.n_agents):
                    if self.alive[agent_x]:
                        if agent_i == agent_x:
                            continue
                        if (self.body[agent_i][0] in self.body[agent_x]):
                            # Snake i collided with x
                            self._schedule_kill(agent_i)
                        if (self.body[agent_x][0] in self.body[agent_i]):
                            # Snake x collided with i (both are possible)
                            self._schedule_kill(agent_x)
        for agent in self._toKill:
            self.alive[agent] = False
        for agent in self._toKill:
            self._dispose_agent(agent)
        self._toKill = []

    def render(self, mode='human'):
        img = copy.copy(self._base_img)
        for agent_i in range(self.n_agents):
            if not self.alive[agent_i]:
                continue

            fill_cell(img, self.body[agent_i][0], cell_size = CELL_SIZE, fill = self.getColor(agent_i, body=False), margin=0.075)
            t = 1
            for cell in self.body[agent_i][1:]:
                fill_cell(img, cell, cell_size = CELL_SIZE, fill = self.getColor(agent_i, body=True), margin = 0.1 + (0.075 / (len(self.body[agent_i]) - 1)) * t)
                t += 1

            # Draw head
            draw_circle(img, self.body[agent_i][0], cell_size=CELL_SIZE, fill=self.getColor(agent_i))
            write_cell_text(img, text=str(agent_i + 1), pos=self.body[agent_i][0], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)
        
        for point in self.food:
            draw_circle(img, point, cell_size=CELL_SIZE, fill=FOOD_COLOR)

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

AGENT_COLOR = [
    "#ff0000",
    "#ffbf00",
    "#80ff00",
    "#00ff40",
    "#00ffff",
    "#0040ff",
    "#8000ff",
    "#ff00bf"
]
FOOD_COLOR = "black"

CELL_SIZE = 35

WALL_COLOR = 'black'

PRE_IDS = {
    'head': 'H',
    'food': 'F',
    'body': 'B',
    'empty': '0'
}
