import time
import multiprocessing
import argparse
import numpy as np
from gym import Env
from typing import Sequence

from aasma import Agent
from aasma.utils import compare_results
from aasma.snake_environment import SnakeEnvironment

from data.export import data_export

from config import scenarios

from pprint import pprint

def run_multi_agent(environment: SnakeEnvironment, agents: Sequence[Agent], n_episodes: int, render: bool, episode_index: int, episode_every: int) -> np.ndarray:
    episodes = {}
    environment.setup_colors(agent.name for agent in agents)
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
                time.sleep(.08)

        print(f"* Episode {(episode * episode_every) + episode_index} done")

        if render:
            environment.close()
        episodes[(episode * episode_every) + episode_index] = data

    return episodes


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--scenario", type=str, required=True, help="The scenario to run")
    parser.add_argument("-n", "--episodes", type=int, default=1, help="How many episodes to run. Defaults to 1")
    parser.add_argument("-e", "--export", nargs='?', const="", help="Export the results to a file. Optionally indicate where you want to save the results (otherwise the default in the configuration will be used)")
    parser.add_argument("-r", "--render", action="store_true", help="Render the episodes on the screen")
    parser.add_argument("-p", "--processes", type=int, default=1, help="How many processes to spawn to run this?")
    opt = parser.parse_args()

    s = scenarios.get(opt.scenario)
    if s is None:
        raise ValueError("I don't see this scenario configured!")
    
    environment = SnakeEnvironment(grid_shape=s["grid"], n_agents=len(s["agents"]), max_steps=None)
    environment.seed()

    results = {}
    results[opt.scenario] = {}

    if opt.processes == 1:
        result = run_multi_agent(environment, s["agents"], opt.episodes, opt.render, 1, 1)
        results[opt.scenario]["data"] = result
    else:
        if (opt.episodes / opt.processes) % 1 != 0:
            raise ValueError(f"Cannot divide {opt.episodes} among {opt.processes} processes! ({opt.episodes/opt.processes})")
        # Spawn a worker pool
        with multiprocessing.Pool(processes=opt.processes) as procs:
            argmap = [(environment, s["agents"], opt.episodes//opt.processes, opt.render, n+1, opt.processes) for n in range(opt.processes)]
            results[opt.scenario]["data"] = {}
            for result in procs.starmap(run_multi_agent, argmap):
                results[opt.scenario]["data"].update(result)

    results[opt.scenario]["max_steps"] = environment._max_steps
    results[opt.scenario]["members"] = {}
    for t in range(len(s["agents"])):
        results[opt.scenario]["members"][t] = s["agents"][t].name

    # Export results
    if opt.export is not None:
        where = opt.export
        if where == "":
            where = s["output"]
        data_export(results, where)

