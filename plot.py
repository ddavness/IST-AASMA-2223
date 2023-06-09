import matplotlib.pyplot as plt
import colorsys
import numpy as np
import sys

from data.utils import mkhash
from data.export import data_import

plt.style.use('_mpl-gallery')

# make data
def snake_lengths(data, step):
    lookup = {}
    dump = {}
    for m in data["members"]:
        dump[data["members"][m]] = []
        lookup[m] = data["members"][m]

    num_running_episodes = 0
    for k in data["data"]:
        if len(data["data"][k]) <= step:
            continue
        currentstep = data["data"][k][step]
        num_running_episodes += 1
        for a in currentstep["agents"]:
            membership = lookup[a]
            agent = currentstep["agents"][a]
            if not agent["alive"]:
                continue
            dump[membership].append(agent["length"])

    if num_running_episodes > 0:
        return {
            "data": dump,
            "episodes_running": num_running_episodes
        }
    else:
        return None

def parse_data(data, f):
    # Sort by agent type
    dump = {
        "episodes_running": {},
        "data": {}
    }
    for m in data["members"]:
        dump["data"][data["members"][m]] = {}

    x = 0
    while True:
        yields = f(data, x)
        x += 1
        if yields is None:
            # Nothing here
            return dump
        for k in yields["data"]:
            dump["data"][k][x] = yields["data"][k]
        dump["episodes_running"][x] = yields["episodes_running"]

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
    fig, ax, = plt.subplots()
    ay = ax.twinx()

    xmax = []

    for agent_type in parsed_data["data"]:
        agent_data = parsed_data["data"][agent_type]
        min = []
        q1 = []
        q2 = []
        q3 = []
        max = []
        num = []

        length = 0
        for x in agent_data:
            arr = agent_data[x]
            if len(arr) == 0:
                break
            length += 1
            # From this array, compute it's max, min, Q1 (25%), Q2 (median) and Q3 (75%)
            n = len(arr)
            num.append(n / parsed_data["episodes_running"][x])
            quartiles = np.quantile(arr, [0.25,0.5,0.75]).tolist()
            min.append(np.min(arr))
            max.append(np.max(arr))
            q1.append(quartiles[0])
            q2.append(quartiles[1])
            q3.append(quartiles[2])

        x = np.linspace(start=0, stop=length, num=length, endpoint=False)

        clr = tuple(round(i * 255) for i in colorsys.hsv_to_rgb((mkhash(agent_type) % 3600) / 3600, 1, 1))
        clrhex = "#{:02x}{:02x}{:02x}".format(*clr)

        ax.plot(x, q2, linewidth=2, color=clrhex, label=agent_type)
        ax.fill_between(x, q1, q3, alpha=0.4, linewidth=0, color=clrhex)
        ax.plot(x, min, linewidth=.8, color=clrhex)
        ax.plot(x, max, linewidth=.8, color=clrhex)

        ay.plot(x, num, linewidth=3, color=clrhex)

        xmax.append(np.max(max))

    topval = ((np.max(xmax) // 20) + 2) * 20

    ax.set(xlim=(0, data["max_steps"]), xticks=np.arange(100, data["max_steps"] + 1, 100),
            ylim=(0, topval), yticks=np.arange(10, topval + 1, 10))

    ay.set(xlim=(0, data["max_steps"]), xticks=np.arange(100, data["max_steps"] + 1, 100),
            ylim=(0, len(data["members"])))

    ax.set_xlabel("Steps")
    ax.set_ylabel("Snake Length")
    ay.set_ylabel("Agents Alive (per running episode)")

    plt.show()