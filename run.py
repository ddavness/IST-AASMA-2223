import time
import argparse
import numpy as np
from gym import Env
from typing import Sequence

from aasma import Agent
from aasma.utils import compare_results
from aasma.snake_environment import SnakeEnvironment

from agents.debug_agent import ForwardAgent
from agents.random_agent import RandomAgent, LessDumbRandomAgent
from agents.algorithm_agent import AStarNearest

from data.export import data_export

from pprint import pprint

def run_multi_agent(environment: SnakeEnvironment, agents: Sequence[Agent], n_episodes: int, render: bool) -> np.ndarray:
    episodes = {}
    for episode in range(n_episodes):
        data = []
        results = environment.reset()
        data.append(results)

        while not results["finished"]:
            if render:
                environment.render()
            results = environment.step([agents[i].action(environment.get_agent_obs(i)) if results["agents"][i]["alive"] else 0 for i in range(len(agents))])
            data.append(results)
            #print(results)
            if render:
                pass#time.sleep(.05)

        pprint(results)

        if render:
            environment.close()
        episodes[episode + 1] = data

    return episodes


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument('-e', '--export', nargs='?', const='./export.json')
    parser.add_argument("-r", "--render", action="store_true")
    opt = parser.parse_args()

    # 1 - Setup the environment
    environment = SnakeEnvironment(grid_shape=(3*10, 4*10), n_agents=8, max_steps=None)
    environment.seed()

    # 2 - Setup the teams
    teams = {
        "Debug": [
            AStarNearest(),
            AStarNearest(),
            AStarNearest(),
            AStarNearest(),
            LessDumbRandomAgent(),
            LessDumbRandomAgent(),
            LessDumbRandomAgent(),
            LessDumbRandomAgent()
        ],
    }

    # 3 - Evaluate teams
    results = {}
    for team, agents in teams.items():
        result = run_multi_agent(environment, agents, opt.episodes, opt.render)
        results[team] = result

    # 4 - Compare results
    if opt.export:
        data_export(results, opt.export)
