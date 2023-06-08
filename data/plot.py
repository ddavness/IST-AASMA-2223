import matplotlib.pyplot as plt
import numpy as np

from export import data_import

plt.style.use('_mpl-gallery')

# make data
def snake_lengths(data, step):
    lookup = {}
    dump = {}
    for m in data["members"]:
        dump[data["members"][m]] = []
        lookup[m] = data["members"][m]

    for k in data:
        if k == "members":
            continue
        if len(data[k]) <= step:
            continue
        currentstep = data[k][step]
        for a in currentstep["agents"]:
            membership = lookup[a]
            agent = currentstep["agents"][a]
            if not agent["alive"]:
                continue
            dump[membership].append(agent["length"])

    total_length = 0
    for m in dump:
        total_length += len(dump[m])
        dump[m].sort()

    if total_length > 0:
        return dump
    else:
        return None

def parse_data(data, f):
    # Sort by agent type
    dump = {}
    for m in data["members"]:
        dump[data["members"][m]] = {}

    x = 0
    while True:
        yields = f(data, x)
        x += 1
        if yields is None:
            # Nothing here
            return dump
        for k in yields:
            dump[k][x] = yields[k]

data = data_import("./sample2.json")
parsed_data = parse_data(data["Debug"], snake_lengths)

# Plot
fig, ax = plt.subplots()
colors_lookup = ["#0060ff", "#ff6000"]
colors_lookup_idx = 0

xmax = []

for agent_type in parsed_data:
    agent_data = parsed_data[agent_type]
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
        num.append(n)
        quartiles = np.quantile(arr, [0.25,0.5,0.75]).tolist()
        min.append(arr[0])
        max.append(arr[-1])
        q1.append(quartiles[0])
        q2.append(quartiles[1])
        q3.append(quartiles[2])

    x = np.linspace(start=0, stop=length, num=length, endpoint=False)

    ax.plot(x, q2, linewidth=2, color=colors_lookup[colors_lookup_idx])
    ax.fill_between(x, q1, q3, alpha=0.4, linewidth=0, color=colors_lookup[colors_lookup_idx])
    ax.plot(x, min, linewidth=.8, color=colors_lookup[colors_lookup_idx])
    ax.plot(x, max, linewidth=.8, color=colors_lookup[colors_lookup_idx])

    xmax.append(np.max(max))

    colors_lookup_idx += 1

topval = ((np.max(xmax) // 20) + 2) * 20

ax.set(xlim=(0, 4800), xticks=np.arange(100, 4801, 100),
        ylim=(0, topval), yticks=np.arange(10, topval + 1, 10))

ax.set_xlabel("Steps")
ax.set_ylabel("Snake Length")

plt.show()