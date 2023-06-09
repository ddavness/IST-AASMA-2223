import matplotlib as mpl
import matplotlib.pyplot as plt
import colorsys
import numpy as np
import sys

from data.utils import mkhash
from data.export import data_import

from config import scenarios

# make data
def snake_lengths(data, step):
    lookup = {}
    dump = {}
    alive = {}
    for m in data["members"]:
        dump[data["members"][m]] = []
        lookup[m] = data["members"][m]

    num_running_episodes = 0
    for episode in data["data"]:
        if len(data["data"][episode]) <= step:
            alive[episode] = None
            continue
        alive[episode] = {}
        currentstep = data["data"][episode][step]
        num_running_episodes += 1
        for a in currentstep["agents"]:
            membership = lookup[a]
            agent = currentstep["agents"][a]
            if not agent["alive"]:
                alive[episode][a] = False
                continue
            alive[episode][a] = True
            dump[membership].append(agent["length"])

    if num_running_episodes > 0:
        return {
            "data": dump,
            "alive": alive
        }
    else:
        return None

def parse_data(data, f):
    # Sort by agent type
    dump = {
        "data": {},
        "alive": {}
    }
    alivedata = {}
    for m in data["members"]:
        dump["data"][data["members"][m]] = {}
        dump["alive"][data["members"][m]] = {}

    x = 0
    while True:
        yields = f(data, x)
        if yields is None:
            # Nothing here
            dump["max_seen_steps"] = x
            break
        x += 1
        for k in yields["data"]:
            dump["data"][k][x] = yields["data"][k]
        for episode in yields["alive"]:
            if yields["alive"][episode] is None:
                # Episode already over, copy from previous
                yields["alive"][episode] = alivedata[x-1][episode]
        alivedata[x] = yields["alive"]
    
    lookup = {}
    for m in data["members"]:
        dump[data["members"][m]] = []
        lookup[m] = data["members"][m]

    # Parse all of these booleans into actual numbers
    for step in alivedata:
        counter = {}
        for l in lookup:
            counter[lookup[l]] = 0
        for episode in alivedata[step]:
            for agent in alivedata[step][episode]:
                counter[lookup[agent]] += 1 if alivedata[step][episode][agent] else 0
        for l in counter:
            dump["alive"][l][step] = counter[l]
    
    return dump


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: plot.py FILE")
        sys.exit(0)

    rawdata = data_import(sys.argv[1])
    scenarioname = list(rawdata.keys())[0]
    print(f"Processing {scenarioname}")
    data = rawdata[scenarioname]
    parsed_data = parse_data(data, snake_lengths)

    # Plot
    with mpl.rc_context(({'axes.titlesize': 22, 'axes.labelsize': 'xx-large', 'legend.fontsize': 'xx-large', 'xtick.labelsize': 'large', 'ytick.labelsize': 'large'})):
        plt.style.use('_mpl-gallery')
        fig, ax, = plt.subplots()
        ay = ax.twinx()

        xmax = []

        for agent_type in parsed_data["data"]:
            agent_data = parsed_data["data"][agent_type]
            q0 = []
            q1 = []
            q2 = []
            q3 = []
            q4 = []
            num = []

            length = 0
            for x in agent_data:
                arr = agent_data[x]
                if len(arr) == 0:
                    break
                length += 1
                # From this array, compute it's max, min, Q1 (25%), Q2 (median) and Q3 (75%)
                n = len(arr)
                num.append(parsed_data["alive"][agent_type][x] / len(data["data"]))
                quartiles = np.quantile(arr, [0.01,0.25,0.5,0.75,1]).tolist()
                q0.append(quartiles[0])
                q1.append(quartiles[1])
                q2.append(quartiles[2])
                q3.append(quartiles[3])
                q4.append(quartiles[4])

            x = np.linspace(start=0, stop=length, num=length, endpoint=False)

            clr = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(mkhash(agent_type), 1, 1))
            clrhex = "#{:02x}{:02x}{:02x}".format(*clr)

            ax.plot(x, q2, linewidth=2, color=clrhex, label="(Median Length) " + agent_type)
            ax.fill_between(x, q1, q3, alpha=0.4, linewidth=0, color=clrhex)
            ax.plot(x, q0, linewidth=.8, color=clrhex)
            ax.plot(x, q4, linewidth=.8, color=clrhex)

            clr2 = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(mkhash(agent_type), .8, 0.8))
            clr2hex = "#{:02x}{:02x}{:02x}".format(*clr2)
            ay.plot(x, num, linewidth=3, color=clr2hex, linestyle="--", label="(Avg. Snakes Alive) " + agent_type)
            ay.fill_between(x, num, 0, alpha=0.04, linewidth=0, color=clr2hex)

            xmax.append(np.max(q4))

        topval = ((np.max(xmax) // 20) + 2) * 20

        ax.set(xlim=(0, data["max_steps"]), xticks=np.arange(200, data["max_steps"] + 1, 200),
                ylim=(0, topval), yticks=np.arange(10, topval + 1, 10))

        # Get the most dominant group for the right y axis
        aycounter = {}
        for m in data["members"]:
            aycounter[data["members"][m]] = 1 if aycounter.get(data["members"][m]) is None else aycounter[data["members"][m]] + 1
        #print(list(aycounter.values()))
        ayh = max(aycounter.values())

        plotsteps = (ayh + 0.5)/(topval/10)
        ay.set(xlim=(0, data["max_steps"]), xticks=np.arange(200, data["max_steps"] + 1, 200),
                ylim=(0, ayh + 0.5), yticks=np.arange(plotsteps, ayh + 0.6, plotsteps))
        print(ayh/plotsteps, ayh + 0.6)

        ax.set_xlabel("Steps")
        ax.set_ylabel("Snake Length")
        ay.set_ylabel("Snakes Alive")

        ax.legend(loc=2)
        ay.legend(loc=1)

        print(plt.rcParams.keys())
        if scenarios[scenarioname]:
            scenario = scenarios[scenarioname]
            agentcounts = {}
            for a in scenario["agents"]:
                agentcounts[a.__class__.__name__] = 1 if agentcounts.get(a.__class__.__name__) is None else agentcounts[a.__class__.__name__] + 1
            somestr = " + ".join([f"{agentcounts[n]}x{n}" for n in agentcounts])
            plt.title(label = f"{scenarioname}, {scenario['grid'][0]}x{scenario['grid'][1]}, {somestr}, ep = {len(data['data'])}")
        else:
            plt.title(label = f"{scenarioname}, ep = {len(data['data'])}")
        plt.show()
