from agents.debug_agent import ForwardAgent
from agents.random_agent import RandomAgent, LessDumbRandomAgent
from agents.algorithm_agent import AStarNearest, AStarCautious

scenarios = {
    "randoms": {
        "output": "randoms.json",
        "agents": [
            LessDumbRandomAgent(),
            LessDumbRandomAgent()
        ],
        "grid": (10, 10),
    },
    "basic1": {
        "output": "basic1.json",
        "agents": [
            LessDumbRandomAgent(),
            AStarNearest()
        ],
        "grid": (10, 10),
    },
    "basic2": {
        "output": "basic2.json",
        "agents": [
            LessDumbRandomAgent(),
            LessDumbRandomAgent(),
            AStarNearest(),
            AStarNearest()
        ],
        "grid": (25, 25)
    },
    "basic4": {
        "output": "basic4.json",
        "agents": [
            LessDumbRandomAgent(),
            LessDumbRandomAgent(),
            LessDumbRandomAgent(),
            AStarNearest(),
            AStarNearest(),
            AStarNearest()
        ],
        "grid": (40, 40)
    },
    "astar2": {
        "output": "astar2.json",
        "agents": [
            AStarNearest(),
            AStarNearest(),
            AStarCautious(),
            AStarCautious()
        ],
        "grid": (25, 25)
    },
    "astar4": {
        "output": "astar4.json",
        "agents": [
            AStarNearest(),
            AStarNearest(),
            AStarNearest(),
            AStarNearest(),
            AStarCautious(),
            AStarCautious(),
            AStarCautious(),
            AStarCautious()
        ],
        "grid": (40, 40)
    }
}
