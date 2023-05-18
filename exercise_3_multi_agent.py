import argparse
import numpy as np
from gym import Env
from typing import Sequence

from aasma import Agent
from aasma.utils import compare_results
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from exercise_1_single_random_agent import RandomAgent
from exercise_2_single_random_vs_greedy import GreedyAgent


def run_multi_agent(environment: Env, agents: Sequence[Agent], n_episodes: int) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        steps = 0
        terminals = [False for _ in range(len(agents))]
        observations = environment.reset()

        while not all(terminals):
            steps += 1
            # TODO - Main Loop (4-6 lines of code)
            raise NotImplementedError()

        results[episode] = steps

        environment.close()

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100)
    opt = parser.parse_args()

    # 1 - Setup the environment
    environment = SimplifiedPredatorPrey(grid_shape=(7, 7), n_agents=4, n_preys=1, max_steps=100)

    # 2 - Setup the teams
    teams = {

        "Random Team": [
            RandomAgent(environment.action_space[0].n),
            RandomAgent(environment.action_space[1].n),
            RandomAgent(environment.action_space[2].n),
            RandomAgent(environment.action_space[3].n),
        ],

        "Greedy Team": [
            GreedyAgent(agent_id=0, n_agents=4),
            GreedyAgent(agent_id=1, n_agents=4),
            GreedyAgent(agent_id=2, n_agents=4),
            GreedyAgent(agent_id=3, n_agents=4)
        ],

        "1 Greedy + 3 Random": [
            GreedyAgent(agent_id=0, n_agents=4),
            RandomAgent(environment.action_space[1].n),
            RandomAgent(environment.action_space[2].n),
            RandomAgent(environment.action_space[3].n)
        ]
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

