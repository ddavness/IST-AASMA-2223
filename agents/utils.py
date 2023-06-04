def sum(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])

def fixDirection(action, observation, agent_i):
    if observation["direction_ptr"][agent_i] + action == len(observation["directions"]):
        return 0
    elif observation["direction_ptr"][agent_i] + action < 0:
        return len(observation["directions"]) - 1
    else:
        return observation["direction_ptr"][agent_i] + action

def wall(next_pos, grid_shape):
    return next_pos[0] < 0 or next_pos[0] >= grid_shape[0] or next_pos[1] < 0 or next_pos[1] >= grid_shape[1]
    
