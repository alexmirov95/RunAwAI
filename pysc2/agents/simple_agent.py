from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_NEUTRAL = 3
_PLAYER_FRIENDLY = 1
_NOT_QUEUED = [0]

class SimpleAgent(base_agent.BaseAgent):
    def step(self, obs):
        # defines behaviors at each step
        super(SimpleAgent, self).step(obs)
         # print obs.observation["available_actions"]
         # TODO
         # X/Y coord agent
         # time?
         # enemy coods
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        player = [int(player_x.mean()), int(player_y.mean())]
        print(player)
        time.sleep(0.1)

        # TODO move to a certain point on the screen
        # target = [5 , 5]
        # return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])




        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
