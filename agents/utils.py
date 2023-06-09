import math
from aasma.snake_environment.snake_environment import Action, PRE_IDS

def sum(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])

def distance(p1, p2):
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])

def fixDirection(action: int, direction_ptr: int, observation: dict):
    if direction_ptr + action == len(observation["directions"]):
        return 0
    elif direction_ptr + action < 0:
        return len(observation["directions"]) - 1
    else:
        return direction_ptr + action

def wall(next_pos, grid_shape):
    return next_pos[0] < 0 or next_pos[0] >= grid_shape[0] or next_pos[1] < 0 or next_pos[1] >= grid_shape[1]


#receives percentage (0 to 1) so 0.5 = 50%
def food(percentage, observation: dict, pos=None):
    currentPos = None
    if(pos != None):
        currentPos = pos
    else:
        currentPos = getCurrentPos(observation)
    closestFood = []
    distances = []

    for i in observation["food"]:
        distances += [distance(currentPos, i)]
    
    sorted_indexes = sorted(range(len(distances)), key=lambda i: distances[i])

    n_food = math.ceil(len(distances) * percentage)

    for i in range(n_food):
        closestFood += [(observation["food"][sorted_indexes[i]][0], observation["food"][sorted_indexes[i]][1])]

    return closestFood

def getCurrentPos(observation):
    return observation["agents"][observation["self"]][0]

def get_valid_actions(direction_ptr: int, observation: dict, pos: tuple):
    # print("*", pos)
    actionValid = []
    for action in Action:
        aux = sum(pos, observation["directions"][fixDirection(action.value, direction_ptr, observation)])
        if wall(aux, observation["grid_shape"]):
            continue
        obs = observation["grid"][aux[0]][aux[1]]
        if obs != PRE_IDS['empty'] and obs != PRE_IDS['food']:
            continue
        else:
            # print("**", aux)
            actionValid += [action.value]
    return actionValid

def get_neighbours(direction_ptr: int, observation: dict, pos: tuple):
    neighbours = []
    actionValid = get_valid_actions(direction_ptr, observation, pos)

    for action in actionValid:
        neighbours += [sum(pos, observation["directions"][fixDirection(action, direction_ptr, observation)])]

    return neighbours

def get_action(direction_ptr: int, observation: dict, currentpos: tuple, nextpos: tuple):
    x = nextpos[0] - currentpos[0]
    y = nextpos[1] - currentpos[1]
    size = len(observation["directions"])
    
    
    for i in range(size):
        if observation["directions"][i] == (x,y):

            ci = i
            if abs(i - direction_ptr) > 1:
                if i > direction_ptr:
                    ci -= size
                else:
                    ci += size
            if abs(ci - direction_ptr) > 1:
                return None # cannot turn back!
            return ci - direction_ptr

def get_direction_ptr(observation, currentpos, nextpos):
    x = nextpos[0] - currentpos[0]
    y = nextpos[1] - currentpos[1]
    for i in range(len(observation["directions"])):
        if observation["directions"][i] == (x,y):
            return i

#get nearby agents heads exluding self
def get_agents_nearby(observation, pos, head):
    neighbours = [(pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]), (pos[0], pos[1] - 1), (pos[0], pos[1] + 1)]
    agents_pos = []
    if head in neighbours:
        neighbours.remove(head)
    for neighbour in neighbours:
        if wall(neighbour, observation["grid_shape"]):
            continue
        for agentId in observation["agents"]:
            if len(observation["agents"][agentId]) == 0:
                # Agent is dead
                continue
            if observation["agents"][agentId][0] == neighbour:
                agents_pos.append(neighbour)
    return agents_pos