import time
import argparse
import numpy as np
from gym import Env
from typing import Sequence

from aasma import Agent
from aasma.utils import compare_results
from aasma.snake_environment import SnakeEnvironment

from agents.debug_agent import ForwardAgent
from agents.random_agent import RandomAgent

def run_multi_agent(environment: SnakeEnvironment, agents: Sequence[Agent], n_episodes: int) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        steps = 0
        terminals = [False for _ in range(len(agents))]
        observations = environment.reset()

        while not all(terminals):
            steps += 1
            environment.render()
            results = environment.step([agent.action() for agent in agents])
            time.sleep(1)

        results[episode] = steps

        environment.close()

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100)
    opt = parser.parse_args()

    # 1 - Setup the environment
    environment = SnakeEnvironment(grid_shape=(15, 15), n_agents=8, max_steps=100)
    environment.seed()

    # 2 - Setup the teams
    teams = {
        "Debug": [
            RandomAgent(),
            RandomAgent(),
            RandomAgent(),
            ForwardAgent(),
            ForwardAgent(),
            ForwardAgent(),
            RandomAgent(),
            RandomAgent()
        ],
    }

    # 3 - Evaluate teams
    results = {}
    for team, agents in teams.items():
        result = run_multi_agent(environment, agents, opt.episodes)
        results[team] = result

    # 4 - Compare results
    compare_results(
        results,
        title="Teams Comparison on 'Predator Prey' Environment",
        colors=["orange", "green", "blue"]
    )

