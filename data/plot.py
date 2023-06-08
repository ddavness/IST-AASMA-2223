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
            print(k, len(yields[k]))
            dump[k][x] = yields[k]

data = data_import("./sample2.json")
print(parse_data(data["Debug"], snake_lengths))

np.random.seed(1)
x = np.linspace(0, 8, 16)
y1 = 3 + 4*x/8 + np.random.uniform(0.0, 0.5, len(x))
y2 = 1 + 2*x/8 + np.random.uniform(0.0, 0.5, len(x))

# plot
fig, ax = plt.subplots()

ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
ax.plot(x, (y1 + y2)/2, linewidth=2)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()