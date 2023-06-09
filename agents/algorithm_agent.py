# Agents that follow an algorithm/strategy to maximize their length
from abc import abstractmethod

from aasma.agent import Agent
from aasma.snake_environment.snake_environment import Action, PRE_IDS

from agents.random_agent import LessDumbRandomAgent

import agents.utils as utils
import math
import heapq

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