'''\
    relative location to current camera
'''

from pysc2.agents import base_agent
from pysc2.lib import actions

import time

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

def get_location(pixel_y, pixel_x, mode='Single'):
    if mode == 'Multiple':
        raise ValueError('unimplemented for mode=Single')
    y = pixel_y.mean()
    x = pixel_x.mean()
    return (x, y)

class DemoAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        time.sleep(0.2) # slow down simulation
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        pixel_y, pixel_x = (unit_type == _SELECTED_TYPE).nonzeros()

        print('location: {}'.format(get_location(pixel_y, pixel_x)))

        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
