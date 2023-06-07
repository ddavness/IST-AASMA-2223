# A set of agents for environment debugging purposes only.
from aasma.agent import Agent
from aasma.snake_environment.snake_environment import Action

class ForwardAgent(Agent):
    def __init__(self):
        super(ForwardAgent, self).__init__("[Debug] Forward-Only Agent")
    
    def action(self, _) -> int:
        return Action.FORWARD.value
