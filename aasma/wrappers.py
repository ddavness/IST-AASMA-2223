from gym import Wrapper


class SingleAgentWrapper(Wrapper):

    """
    A Wrapper for Multi-Agent environments which only contain a single agent.
    """

    def __init__(self, env, agent_id):
        super(SingleAgentWrapper, self).__init__(env)
        assert env.n_agents == 1, "Single Agent Wrapper for Multi-Agent Environments requires only one agent."
        self.agent_id = agent_id
        self.action_space = env.action_space[agent_id]

    def reset(self):
        return super(SingleAgentWrapper, self).reset()[self.agent_id]

    def step(self, action):
        next_observations, rewards, terminals, info = super(SingleAgentWrapper, self).step([action])
        return next_observations[self.agent_id], rewards[self.agent_id], terminals[self.agent_id], info

    def get_action_meanings(self):
        return self.env.get_action_meanings(self.agent_id)
