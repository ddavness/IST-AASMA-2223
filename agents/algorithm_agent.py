# Agents that follow an algorithm/strategy to maximize their length
from abc import abstractmethod

from aasma.agent import Agent
from aasma.snake_environment.snake_environment import Action, PRE_IDS

from agents.random_agent import LessDumbRandomAgent

import agents.utils as utils
import math
import heapq
from enum import Enum

class BullyStates(Enum):
    PASSIVE = 0    #behaves like a AstarAgent
    ACTIVE = 1     #trys to remove the target from the episode

class AStarAgent(Agent):
    """
    Heuristic: The nearest food point (path-wise)
    """
    def __init__(self, seed=None):
        super(AStarAgent, self)
        self.fallback = LessDumbRandomAgent(seed)
        self.step = 0
        self.goal = None
        self.reevaluateEvery = None

    @abstractmethod
    def setup(self, observation):
        raise NotImplementedError()

    @abstractmethod
    def setGoal(self, observation):
        raise NotImplementedError()
        
    def astar(self, observation, _goal, pos = None, direction_ptr = None):
        goal = _goal
        if pos != None:
            start = pos
        else:
            start = utils.getCurrentPos(observation)

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {}
        cost_so_far = {}
        direction_ptrs ={}
        came_from[start] = None
        cost_so_far[start] = 0
        if(direction_ptr != None):
            direction_ptrs[start] = direction_ptr
        else:
            direction_ptrs[start] = observation["direction_ptr"][observation["self"]]

        while frontier:
            _, current = heapq.heappop(frontier)

            if goal[0] is None:
                self.step = self.reevaluateEvery
                goal = [(0, 0)]

            if current in goal:
                break
            
            # print(goal, current, direction_ptrs[current])          
            for next_node in utils.get_neighbours(direction_ptrs[current], observation, current):
                # print(next_node)
                new_cost = cost_so_far[current] + 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + utils.distance(next_node, goal[0])
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
                    direction_ptrs[next_node] = utils.get_direction_ptr(observation, current, next_node)
                    # print("*", current, next_node, utils.get_direction_ptr(observation, current, next_node))

        path = []
        current = goal[0]
        # print(came_from)
        while current != start:
            path.append(current)
            # print(path)
            if came_from.get(current) is not None:
                current = came_from[current]
            else:
                return None # No possible path!
        path.reverse()
        return path

    @abstractmethod
    def action(self, observation) -> int:
        raise NotImplementedError()

class AStarNearest(AStarAgent):
    def __init__(self, seed=None):
        super(AStarAgent, self).__init__("A* Searching Agent - Nearest Food")
        super(AStarNearest, self).__init__(seed)

    def setup(self, observation):
        # "Configure" the agent according to the environment
        grid_shape = observation["grid_shape"]
        self.reevaluateEvery = 0.5 * (grid_shape[0] * grid_shape[1]) ** 0.5
        pass

    def setGoal(self, observation):
        self.goal = None
        candidate_goals = utils.food(0.5, observation) #, observation["self"])
        # Sort by distance
        closest_distance = math.inf
        for goal in candidate_goals:
            path = self.astar(observation, [goal])
            if path is not None:
                distance = len(path)
                if distance < closest_distance:
                    closest_distance = distance
                    self.goal = goal
    
    def action(self, observation) -> int:
        head = observation["agents"][observation["self"]][0]
        if(self.reevaluateEvery == None):
            self.setup(observation)
        # print(self.goal)
        if(self.reevaluateEvery == self.step or self.goal == None or len(self.goal) == 0 or self.goal == head):
            self.setGoal(observation)
            self.step = 0
        self.step += 1
        path = self.astar(observation, [self.goal])
        next_pos = None
        if path is None or len(path) == 0:
            # Reevaluate goal
            self.setGoal(observation)
            self.step = 1
            path = self.astar(observation, [self.goal])
            # And even then, if there's still no way around, fallback
            if path is None or len(path) == 0:
                return self.fallback.action(observation)
        next_pos = path[0]
        return utils.get_action(observation["direction_ptr"][observation["self"]], observation, head, next_pos)

class AStarCautious(AStarAgent):
    """
    Heuristic: weigh between nearest food point (path-wise) and longer brute distance from the other agents
    (rationale: so that it doesn't face competition from other snakes)
    """
    def __init__(self, seed=None):
        super(AStarAgent, self).__init__("A* Searching Agent - Cautious")
        super(AStarCautious, self).__init__(seed)

    def setup(self, observation):
        # "Configure" the agent according to the environment
        grid_shape = observation["grid_shape"]
        self.reevaluateEvery = 0.5 * (grid_shape[0] * grid_shape[1]) ** 0.5
        pass

    def setGoal(self, observation, exceptions = None):
        self.goal = None
        candidate_goals = utils.food(0.5, observation) #, observation["self"])
        if exceptions != None:
            for exception in exceptions:
                if exception in candidate_goals:
                    candidate_goals.remove(exception)
        # Sort by distance
        closest_distance = math.inf
        for goal in candidate_goals:
            path = self.astar(observation, [goal])
            if path is not None:
                distance = len(path)
                if distance < closest_distance:
                    closest_distance = distance
                    self.goal = goal
    
    def predictGoal(self, observation, agent_pos):
        agent_goal = None
        candidate_goals = utils.food(0.5, observation, pos=agent_pos) #, observation["self"])
        # Sort by distance
        closest_distance = math.inf
        for goal in candidate_goals:
            path = self.astar(observation, [goal])
            if path is not None:
                distance = len(path)
                if distance < closest_distance:
                    closest_distance = distance
                    agent_goal = goal
        return agent_goal

    def action(self, observation) -> int:
        head = observation["agents"][observation["self"]][0]
        if(self.reevaluateEvery == None):
            self.setup(observation)
        # print(self.goal)
        if(self.reevaluateEvery == self.step or self.goal == None or len(self.goal) == 0 or self.goal == head):
            self.setGoal(observation)
            self.step = 0
        self.step += 1
        path = self.astar(observation, [self.goal])
        next_pos = None
        if path is None or len(path) == 0:
            # Reevaluate goal
            self.setGoal(observation)
            self.step = 1
            path = self.astar(observation, [self.goal])
            # And even then, if there's still no way around, fallback
            if path is None or len(path) == 0:
                return self.fallback.action(observation)
        next_pos = path[0]

        agents_pos = utils.get_agents_nearby(observation, next_pos, head)
        agents_goals = []
        if(agents_pos != []):
            for agent_pos in agents_pos:
                agents_goals.append(self.predictGoal(observation, agent_pos))
        if self.goal in agents_goals:
            self.setGoal(observation, agents_goals)
            self.step = 1
            path = self.astar(observation, [self.goal])
            # And even then, if there's still no way around, fallback
            if path is None or len(path) == 0:
                return self.fallback.action(observation)
            next_pos = path[0]
        return utils.get_action(observation["direction_ptr"][observation["self"]], observation, head, next_pos)

class AStarNearestTailCheck(AStarAgent):
    def __init__(self, seed=None):
        super(AStarAgent, self).__init__("A* Searching Agent - Nearest Food w/ Tail Check")
        super(AStarNearestTailCheck, self).__init__(seed)
        self.exceptions = [] #goals that will make the snake trap herself
        self.backup = (0,0)
        self.path = []

    def setup(self, observation):
        # "Configure" the agent according to the environment
        grid_shape = observation["grid_shape"]
        self.reevaluateEvery = 0.5 * (grid_shape[0] * grid_shape[1]) ** 0.5
        pass

    def setGoal(self, observation, exceptions = []):
        self.goal = None
        candidate_goals = utils.food(0.5, observation) #, observation["self"])
        # Sort by distance
        closest_distance = math.inf
        if exceptions != []:
            for i in exceptions:
                if i in candidate_goals:
                    candidate_goals.remove(i)
        for goal in candidate_goals:
            path = self.astar(observation, [goal])
            if path is not None:
                distance = len(path)
                if distance < closest_distance:
                    closest_distance = distance
                    self.goal = goal
    
    def action(self, observation) -> int:
        head = observation["agents"][observation["self"]][0]
        if(self.reevaluateEvery == None):
            self.setup(observation)
        # print(self.goal)
        if(self.reevaluateEvery == self.step or self.goal == None or len(self.goal) == 0 or self.goal == head):
            self.setGoal(observation)
            self.step = 0
        self.step += 1
        path = self.astar(observation, [self.goal])

        if(self.path == path):
            return self.fallback.action(observation)
        self.path = path

        next_pos = None
        if path is None or len(path) == 0:
            # Reevaluate goal
            self.setGoal(observation)
            self.step = 1
            path = self.astar(observation, [self.goal])
            # And even then, if there's still no way around, fallback
            if path is None or len(path) == 0:
                return self.fallback.action(observation)
        #print(observation["agents"][observation["self"]][-1])
        tail = observation["agents"][observation["self"]][-1]
        # Simulate that the tail disappears from there. Just so that a* doesn't complain
        observation["grid"][tail[0]][tail[1]] = PRE_IDS['empty']
        tailPath = self.astar(observation, [observation["agents"][observation["self"]][-1]], pos = path[0], direction_ptr = utils.get_direction_ptr(observation, head, path[0]))
        if tailPath == None:
            print("vou me incoralar")
            if self.goal == (0, 0):
                # Options have been exhausted
                return self.fallback.action(observation)
            observation["grid"][tail[0]][tail[1]] = PRE_IDS['body']+str(observation["self"] + 1)
            self.exceptions.append(path[-1])
            self.setGoal(observation, self.exceptions)
            #print(self.goal)
            if(self.goal == None):
                self.goal = self.backup
            self.step = 0
            return self.action(observation)
        elif len(tailPath) >= 2:
            self.backup = tailPath[-2]
        else:
            self.backup = (0,0) 
        next_pos = path[0]
        self.exceptions = []
        return utils.get_action(observation["direction_ptr"][observation["self"]], observation, head, next_pos)

class AStarCautiousTailCheck(AStarAgent):
    """
    Heuristic: weigh between nearest food point (path-wise) and longer brute distance from the other agents
    (rationale: so that it doesn't face competition from other snakes)
    """
    def __init__(self, seed=None):
        super(AStarAgent, self).__init__("A* Searching Agent - Cautious w/ Tail Check")
        super(AStarCautiousTailCheck, self).__init__(seed)
        self.exceptions = [] #goals that will make the snake trap herself
        self.backup = (0,0)
        self.path = []

    def setup(self, observation):
        # "Configure" the agent according to the environment
        grid_shape = observation["grid_shape"]
        self.reevaluateEvery = 0.5 * (grid_shape[0] * grid_shape[1]) ** 0.5
        pass

    def setGoal(self, observation, exceptions = None):
        self.goal = None
        candidate_goals = utils.food(0.5, observation) #, observation["self"])
        if exceptions != None:
            for exception in exceptions:
                if exception in candidate_goals:
                    candidate_goals.remove(exception)
        # Sort by distance
        closest_distance = math.inf
        for goal in candidate_goals:
            path = self.astar(observation, [goal])
            if path is not None:
                distance = len(path)
                if distance < closest_distance:
                    closest_distance = distance
                    self.goal = goal
    
    def predictGoal(self, observation, agent_pos):
        agent_goal = None
        candidate_goals = utils.food(0.5, observation, pos=agent_pos) #, observation["self"])
        # Sort by distance
        closest_distance = math.inf
        for goal in candidate_goals:
            path = self.astar(observation, [goal])
            if path is not None:
                distance = len(path)
                if distance < closest_distance:
                    closest_distance = distance
                    agent_goal = goal
        return agent_goal

    def action(self, observation) -> int:
        head = observation["agents"][observation["self"]][0]
        if(self.reevaluateEvery == None):
            self.setup(observation)
        # print(self.goal)
        if(self.reevaluateEvery == self.step or self.goal == None or len(self.goal) == 0 or self.goal == head):
            self.setGoal(observation)
            self.step = 0
        self.step += 1
        path = self.astar(observation, [self.goal])
        next_pos = None
        if path is None or len(path) == 0:
            # Reevaluate goal
            self.setGoal(observation)
            self.step = 1
            path = self.astar(observation, [self.goal])
            # And even then, if there's still no way around, fallback
            if path is None or len(path) == 0:
                return self.fallback.action(observation)
        #print(path)
        next_pos = path[0]
        
        if(self.path == path):
            return self.fallback.action(observation)
        self.path = path

        agents_pos = utils.get_agents_nearby(observation, next_pos, head)
        agents_goals = []
        if(agents_pos != []):
            for agent_pos in agents_pos:
                agents_goals.append(self.predictGoal(observation, agent_pos))
        if self.goal in agents_goals:
            self.setGoal(observation, agents_goals)
            self.step = 1
            path = self.astar(observation, [self.goal])
            # And even then, if there's still no way around, fallback
            if path is None or len(path) == 0:
                return self.fallback.action(observation)
        tail = observation["agents"][observation["self"]][-1]
        # Simulate that the tail disappears from there. Just so that a* doesn't complain
        observation["grid"][tail[0]][tail[1]] = PRE_IDS['empty']
        tailPath = self.astar(observation, [observation["agents"][observation["self"]][-1]], pos = path[0], direction_ptr = utils.get_direction_ptr(observation, head, path[0]))
        if tailPath == None:
            print("vou me incoralar")
            if self.goal == (0,0):
                # Options have been exhausted
                return self.fallback.action(observation)
            observation["grid"][tail[0]][tail[1]] = PRE_IDS['body']+str(observation["self"] + 1)
            self.exceptions.append(path[-1])
            self.setGoal(observation, self.exceptions)
            if(self.goal == None):
                self.goal = self.backup
            self.step = 0
            return self.action(observation)
        elif len(tailPath) >= 2:
            self.backup = tailPath[-2]
        else:
            self.backup = (0,0) 
        next_pos = path[0]
        self.exceptions = []
        return utils.get_action(observation["direction_ptr"][observation["self"]], observation, head, next_pos)

class AStarBully(AStarAgent):
    """
    INCOMPLETE IMPLEMENTATION, DO NOT USE
    Heuristic: weigh between nearest food point (path-wise) and from a certain size is gonna target an agent to bully (if the target is removed than it assigns a new target)
    (rationale: so that it can get food drops from other snakes and eliminate competition)
    """
    def __init__(self, seed=None):
        super(AStarAgent, self).__init__("A* Searching Agent - Bully")
        super(AStarBully, self).__init__(seed)
        self.state = BullyStates.PASSIVE  
        self.target = None   # agentid = int
        self.passiveToSearching = None # int (size to be used to compare to current size)

    def setup(self, observation):
        # "Configure" the agent according to the environment
        grid_shape = observation["grid_shape"]
        self.reevaluateEvery = 0.5 * (grid_shape[0] * grid_shape[1]) ** 0.5
        height = len(observation["grid_shape"])
        weight = len(observation["grid_shape"][0])
        if height > weight:
            self.passiveToSearching = height
        else:
            self.passiveToSearching = weight
        pass

    def setGoal(self, observation):
        self.goal = None
        candidate_goals = utils.food(0.5, observation) #, observation["self"])
        # Sort by distance
        closest_distance = math.inf
        for goal in candidate_goals:
            path = self.astar(observation, [goal])
            if path is not None:
                distance = len(path)
                if distance < closest_distance:
                    closest_distance = distance
                    self.goal = goal
    
    def predictGoal(self, observation, agent_pos):
        agent_goal = None
        candidate_goals = utils.food(0.5, observation, pos=agent_pos) #, observation["self"])
        # Sort by distance
        closest_distance = math.inf
        for goal in candidate_goals:
            path = self.astar(observation, [goal])
            if path is not None:
                distance = len(path)
                if distance < closest_distance:
                    closest_distance = distance
                    agent_goal = goal
        return agent_goal

    def setTarget(self,observation):
        thisAgent = observation["agents"][observation["self"]]
        res = math.inf
        for agent in observation["agents"]:
            if agent == [] or agent == thisAgent:
                continue 
            else:
                auxDistance = utils.distance( thisAgent[0], agent[0])
                if auxDistance < res:
                    res = auxDistance
                    self.target = agent

    def action(self, observation) -> int:
            currentSize = utils.get_agent_size(observation, observation["self"])
            head = observation["agents"][observation["self"]][0]

            if currentSize >= self.passiveToSearching:
                self.state = BullyStates.ACTIVE 

            if self.state == BullyStates.ACTIVE: 
                if (self.target == None or observation["agents"][self.target] == []):
                    self.target = self.setTarget()
                #presseguicao
                pass
            
            else:
                if(self.reevaluateEvery == None):
                    self.setup(observation)
                # print(self.goal)
                if(self.reevaluateEvery == self.step or self.goal == None or len(self.goal) == 0 or self.goal == head):
                    self.setGoal(observation)
                    self.step = 0
                self.step += 1
                path = self.astar(observation, [self.goal])
                next_pos = None
                if path is None or len(path) == 0:
                    # Reevaluate goal
                    self.setGoal(observation)
                    self.step = 1
                    path = self.astar(observation, [self.goal])
                    # And even then, if there's still no way around, fallback
                    if path is None or len(path) == 0:
                        return self.fallback.action(observation)
                next_pos = path[0]
            return utils.get_action(observation["direction_ptr"][observation["self"]], observation, head, next_pos)
    