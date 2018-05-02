from pysc2.agents import base_agent
from pysc2.lib import actions

import time

class SimpleAgent(base_agent.BaseAgent):
    def step(self, obs):
        # defines behaviors at each step
        super(SimpleAgent, self).step(obs)
        print obs.observation["available_actions"]

        time.sleep(0.1)

        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
