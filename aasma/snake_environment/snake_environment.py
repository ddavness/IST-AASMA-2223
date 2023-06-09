import copy
import logging
import copy

import math
import numpy as np
logger = logging.getLogger(__name__)

import colorsys
import gym
from gym.utils import seeding
from data.utils import mkhash

from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

from enum import Enum
class Action(Enum):
    LEFT = -1
    FORWARD = 0
    RIGHT = 1

class DeathReason(str, Enum):
    MAP_EDGE = "MAP_EDGE"
    COLLIDE_SELF = "COLLIDE_SELF"
    COLLIDE_OTHER_BODY = "COLLIDE_OTHER_BODY"
    COLLIDE_OTHER_HEAD = "COLLIDE_OTHER_HEAD"

class SnakeEnvironment(gym.Env):

    """
    A modified version of ma_gym.envs.predator_prey.predator_prey.PredatorPrey to fit a snake game

    grid_shape: (x, y) - The shape of the grid, how big is it? Must be at least 10x10
    n_agents: int - The number of players/snakes
    max_steps: int - The maximum number of moves in the game. Defaults to x * y
    food_equilibrium: (int) -> float - The target number of food points in the environment. Defaults to (1/2)*area^(1/2)
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(10, 10), n_agents=2, max_steps=None, food_equilibrium=None):
        if grid_shape[0] < 10 or grid_shape[1] < 10:
            raise ValueError("Grid must be at least 10x10")

        self._grid_shape = grid_shape
        self._numagents = n_agents
        self._max_steps = max_steps if max_steps is not None else grid_shape[0] * grid_shape[1]
        self._food_equilibrium_f = food_equilibrium if food_equilibrium is not None else self._food_equilibrium_default
        self._step_count = None
        self._finished = False
        
        self._agent_hues = []
        self._grid = self._create_grid()
        self._viewer = None
        
        self._alive = [True for _ in range(self._numagents)]
        self._dead_because = [None for _ in range(self._numagents)]
        self._body = {_: [] for _ in range(self._numagents)}  # self._body= a list of all the bodys from agents than are a list on its own[(x1,y1),(x2,y2)]
        self._head_directions = [(1, 0), (0, 1), (-1, 0), (0, -1)] # Turning right will
        self._direction_ptr = {_: 0 for _ in range(self._numagents)} # (1, 0), (0, -1), (-1, 0), (0, 1)
        self._food = []

        self.seed()
        self._draw_base_img()

    def reset(self, seed=None):
        self.seed(seed)

        self._body = {_: [] for _ in range(self._numagents)}
        self._direction_ptr = {_: 0 for _ in range(self._numagents)}
        self._food = []
        self._toKill = []
        self._grid = self._create_grid()

        self._init_full_obs()
        self._step_count = 0
        self._finished = False
        self._alive = [True for _ in range(self._numagents)]
        self._dead_because = [None for _ in range(self._numagents)]

        return self.stats()

    def stats(self):
        stats = {
            "finished": self._finished,
            "agents": {},
            "food": len(self._food),
            "food_equilibrium": self._food_equilibrium(),
            "step": self._step_count
        }

        for i in range(self._numagents):
            stats["agents"][i] = {
                "alive": self._alive[i],
                "dead_because": self._dead_because[i],
                "length": len(self._body[i])
            }

        return stats

    def step(self, agents_action):
        if self._finished:
            return self.stats
        self._step_count += 1

        for agent_i, action in enumerate(agents_action):
            if self._alive[agent_i]:
                self._update_agent_pos(agent_i, action)

        if (self._step_count >= self._max_steps) or (sum(self._alive) < 2):
            for i in range(self._numagents):
                self._finished = True

        self._check_collisions()

        for i in self._body:
            self._update_agent_view(i)

        self._regenerate_food()

        return self.stats()
    
    def get_agent_obs(self, agent_i):
        # All agents have full observation
        return {
            "self": agent_i,
            "grid": np.array(self._grid).tolist(),
            "grid_shape": self._grid_shape,
            "agents": {i: self._body[i] if self._alive[i] else [] for i in self._body},
            "directions": self._head_directions.copy(),
            "direction_ptr": {i: self._direction_ptr[i] if self._alive[i] else None for i in self._direction_ptr},
            "food": np.array(self._food).tolist()
        }
    
    def render(self, mode='human'):
        img = copy.copy(self._base_img)
        for agent_i in range(self._numagents):
            if not self._alive[agent_i]:
                continue

            fill_cell(img, self._body[agent_i][0], cell_size = CELL_SIZE, fill = self._agent_color(agent_i), margin=0.075)
            t = 1
            for cell in self._body[agent_i][1:]:
                fill_cell(img, cell, cell_size = CELL_SIZE, fill = self._agent_color(agent_i), margin = 0.15 + ((0.1 / (1 + 2**(3-(len(self._body[agent_i]))))) / (len(self._body[agent_i]) - 1)) * t)
                t += 1

            # Draw head
            draw_circle(img, self._body[agent_i][0], cell_size=CELL_SIZE, fill=self._agent_color(agent_i))
            write_cell_text(img, text=str(agent_i + 1), pos=self._body[agent_i][0], cell_size=CELL_SIZE,
                            fill='white', margin=0.5)
        
        for point in self._food:
            draw_circle(img, point, cell_size=CELL_SIZE, fill=FOOD_COLOR)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer(maxwidth=1000)
            self._viewer.imshow(img)
            return self._viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]
    
    def setup_colors(self, agent_names):
        self._agent_hues = [mkhash(n) % 7200 for n in agent_names]

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # Private env functions
    
    def _food_equilibrium_default(self, area):
        return 0.6 * area**0.5
    
    def _food_equilibrium(self):
        area_total = self._grid_shape[0] * self._grid_shape[1]
        player_area = 0
        for snake in self._body:
            player_area += len(self._body[snake])

        area_available = area_total - player_area
        return self._food_equilibrium_f(area_available)
    
    def _regenerate_food(self):
        food_target = self._food_equilibrium()
        if len(self._food) > food_target:
            return
        sample = self.np_random.uniform(0, food_target)
        if sample > food_target - len(self._food):
            return
        else:
            # Spawn a food block in any random free space
            while True:
                x = math.floor(self.np_random.uniform(0, self._grid_shape[0]))
                y = math.floor(self.np_random.uniform(0, self._grid_shape[1]))
                if self._grid[x][y] == PRE_IDS['empty']:
                    self._food += [(x, y)]
                    self._grid[x][y] = PRE_IDS['food']
                    return

    def _create_snakes(self):
        for i in self._body:
            clear = False
            while not clear:
                x = math.floor(self.np_random.uniform(3, self._grid_shape[0] - 3))
                y = math.floor(self.np_random.uniform(3, self._grid_shape[1] - 3))
                dirptr = math.floor(self.np_random.uniform(0, 4))
                dir = self._head_directions[dirptr]
                self._direction_ptr[i] = dirptr
                if self._grid[x][y] == PRE_IDS['empty'] \
                and self._grid[x - dir[0]][y - dir[1]] == PRE_IDS['empty'] \
                and self._grid[x - 2 * dir[0]][y - 2 * dir[1]] == PRE_IDS['empty']:
                    self._body[i].append((x, y))
                    self._body[i].append((x - dir[0], y - dir[1]))
                    self._body[i].append((x - 2 * dir[0], y - 2 * dir[1]))
                    self._update_agent_view(i)
                    clear = True

    def _draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def _create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def _init_full_obs(self):
        self._grid = self._create_grid()
        self._create_snakes()
        self._draw_base_img()

    def _body_exists(self, pos):
        row, col = pos
        return PRE_IDS['body'] in self._grid[row, col] or PRE_IDS['head'] in self._grid[row, col]

    def _cell_is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _cell_is_vacant(self, pos):
        return self.is_valid(pos) and (self._grid[pos[0]][pos[1]] == PRE_IDS['empty'])

    def _update_agent_pos(self, agent_i, move):
        if not self._alive[agent_i]:
            return

        self._direction_ptr[agent_i] = (self._direction_ptr[agent_i] + move) % 4
        dir = self._head_directions[self._direction_ptr[agent_i]]
        # Next head position
        next_pos = (self._body[agent_i][0][0] + dir[0], self._body[agent_i][0][1] + dir[1])

        if next_pos is not None:
            if next_pos[0] < 0 or next_pos[0] >= self._grid_shape[0] or next_pos[1] < 0 or next_pos[1] >= self._grid_shape[1]:
                self._alive[agent_i] = False
                self._dead_because[agent_i] = DeathReason.MAP_EDGE
                self._dispose_agent(agent_i)
                return
            elif self._grid[next_pos[0]][next_pos[1]] == PRE_IDS['food']:
                # We grow by one unit
                self._body[agent_i] = [next_pos] + self._body[agent_i]
                foodset = set(self._food)
                foodset.remove(next_pos)
                self._food = list(foodset)
                self._grid[next_pos[0]][next_pos[1]] = PRE_IDS['empty']
            else:
                tail = self._body[agent_i][-1]
                self._grid[tail[0]][tail[1]] = PRE_IDS['empty']
                try:
                    foodset = set(self._food)
                    foodset.remove(tail)
                    self._food = list(foodset)
                except:
                    pass
                self._body[agent_i] = [next_pos] + self._body[agent_i][:-1]
            self._update_agent_view(agent_i)

    def _update_agent_view(self, agent_i):
        if self._alive[agent_i]:
            self._grid[self._body[agent_i][0][0]][self._body[agent_i][0][1]] = PRE_IDS['head'] + str(agent_i + 1)
            for b in self._body[agent_i][1:]:
                # print(b)
                self._grid[b[0]][b[1]] = PRE_IDS['body'] + str(agent_i + 1)

    def _dispose_agent(self, agent_i):
        print(f"Disposing agent {agent_i + 1}")
        for bodypart in range(1, len(self._body[agent_i])):
            if self._grid[self._body[agent_i][bodypart][0]][self._body[agent_i][bodypart][1]] == PRE_IDS['head'] + str(agent_i + 1) \
            or self._grid[self._body[agent_i][bodypart][0]][self._body[agent_i][bodypart][1]] == PRE_IDS['body'] + str(agent_i + 1) \
            or self._grid[self._body[agent_i][bodypart][0]][self._body[agent_i][bodypart][1]] == PRE_IDS['empty']:
                self._grid[self._body[agent_i][bodypart][0]][self._body[agent_i][bodypart][1]] = PRE_IDS['empty']
            if bodypart % 2 == 1 and self._grid[self._body[agent_i][bodypart][0]][self._body[agent_i][bodypart][1]] == PRE_IDS['empty']:
                # print(self._body[agent_i][bodypart])
                self._food += [self._body[agent_i][bodypart]]
                self._food = list(set(self._food))
                self._grid[self._body[agent_i][bodypart][0]][self._body[agent_i][bodypart][1]] = PRE_IDS['food']

                
        self._alive[agent_i] = False

    def _schedule_kill(self, i):
        self._toKill += [i]

    def _check_collisions(self):
        for agent_i in range(self._numagents):
            if self._alive[agent_i]:
                if (self._body[agent_i][0] in self._body[agent_i][1:]):
                    # Snake collided with it's own body
                    self._dead_because[agent_i] = DeathReason.COLLIDE_SELF
                    self._schedule_kill(agent_i)
                for agent_x in range(self._numagents):
                    if self._alive[agent_x]:
                        if agent_i == agent_x:
                            continue
                        if (self._body[agent_i][0] in self._body[agent_x]):
                            # Snake i collided with x
                            if self._body[agent_i][0] == self._body[agent_x][0]:
                                self._dead_because[agent_i] = DeathReason.COLLIDE_OTHER_HEAD
                            else:
                                self._dead_because[agent_i] = DeathReason.COLLIDE_OTHER_BODY
                            self._schedule_kill(agent_i)
        for agent in self._toKill:
            self._alive[agent] = False
        for agent in self._toKill:
            self._dispose_agent(agent)
        self._toKill = []

    def _agent_color(self, agent_i):
        hue = self._agent_hues[agent_i]
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(hue/7200, 0.5 + 0.06 * agent_i, 1 - 0.04 * agent_i))

FOOD_COLOR = "black"

CELL_SIZE = 35

WALL_COLOR = 'black'

PRE_IDS = {
    'head': 'H',
    'food': 'F',
    'body': 'B',
    'empty': '0'
}
