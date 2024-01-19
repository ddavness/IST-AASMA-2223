from agents.debug_agent import ForwardAgent
from agents.random_agent import RandomAgent, LessDumbRandomAgent
from agents.algorithm_agent import AStarNearest, AStarCautious, AStarNearestTailCheck, AStarCautiousTailCheck

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
    "astar3": {
        "output": "astar3.json",
        "agents": [
            AStarNearest(),
            AStarNearest(),
            AStarNearest(),
            AStarCautious(),
            AStarCautious(),
            AStarCautious()
        ],
        "grid": (40, 40)
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
    },
    "astar6": {
        "output": "astar6.json",
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
        "grid": (60, 60)
    },
    "tcheck_nearest": {
        "output": "tcheck_nearest.json",
        "agents": [
            AStarNearest(),
            AStarNearest(),
            AStarNearest(),
            AStarNearest(),
            AStarNearestTailCheck(),
            AStarNearestTailCheck(),
            AStarNearestTailCheck(),
            AStarNearestTailCheck()
        ],
        "grid": (50, 50)
    },
    "tcheck_cautious": {
        "output": "tcheck_cautious.json",
        "agents": [
            AStarCautious(),
            AStarCautious(),
            AStarCautious(),
            AStarCautious(),
            AStarCautiousTailCheck(),
            AStarCautiousTailCheck(),
            AStarCautiousTailCheck(),
            AStarCautiousTailCheck()
        ],
        "grid": (50, 50)
    },
    "tcheck_comparison": {
        "output": "tcheck_comparison.json",
        "agents": [
            LessDumbRandomAgent(),
            LessDumbRandomAgent(),
            AStarNearest(),
            AStarNearest(),
            AStarCautious(),
            AStarCautious(),
            AStarNearestTailCheck(),
            AStarNearestTailCheck(),
            AStarNearestTailCheck(),
            AStarNearestTailCheck(),
            AStarNearestTailCheck(),
            AStarCautiousTailCheck(),
            AStarCautiousTailCheck(),
            AStarCautiousTailCheck(),
            AStarCautiousTailCheck(),
            AStarCautiousTailCheck()
        ],
        "grid": (36, 64)
    },
    "overrandom": {
        "output": "overrandom.json",
        "agents": [
            LessDumbRandomAgent(),
            LessDumbRandomAgent(),
            LessDumbRandomAgent(),
            LessDumbRandomAgent(),
            LessDumbRandomAgent()
        ],
        "grid": (9*8, 16*8)
    }
}
