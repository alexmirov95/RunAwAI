
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import time

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index


def get_location(pixel_y, pixel_x, mode='Single'):
    if mode == 'Multiple':
        raise ValueError('unimplemented for mode=Single')
    y = pixel_y.mean()
    x = pixel_x.mean()
    return (x, y)


class MoveTest(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""

  ct = 0
  # square dancing
  locations = [[int(10), int(10)], [int(50), int(15)], [int(50), int(50)], [int(10), int(50)]]
  # linear movement along x axis
  # locations = [[int(5), int(30)], [int(10), int(30)], [int(15), int(30)], [int(20), int(30)]]

  target = [-1] * 2 # uninitialized target value

  def step(self, obs):
    super(MoveTest, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      
      # get player locations
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      locationTuple = [int(player_x.mean()), int(player_y.mean())]

      # how close the units will get to the destination before selecting a new target destination
      distToTarget = 5
      if ( ((abs(locationTuple[0] - self.target[0]) >= distToTarget) or (abs(locationTuple[1] - self.target[1]) >= distToTarget)) and (self.target[0] >= 0)):
        # has not reached (or at least gotten close to) destination
        # no op
        time.sleep(0.2)
        print("No op")
        print("   locationTuple=", locationTuple, "     target=", self.target)
        return actions.FunctionCall(_NO_OP, [])
      else:
        # close enough to destination for a new waypoint
        # set new target destination waypoint
        time.sleep(0.2)
        self.target = self.locations[self.ct % 4] # iterate thru the same 4 array values
        self.ct += 1
        print ("moving to ", self.target[0], ",", self.target[1])
        return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, self.target])

    else:
      print("selecting units")
      # Select all units
      # return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

      # Selection Single Unit
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])



class MoveToBeacon(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""

  def step(self, obs):
    super(MoveToBeacon, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      if not neutral_y.any():
        return actions.FunctionCall(_NO_OP, [])
      target = [int(neutral_x.mean()), int(neutral_y.mean())]
      return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
    else:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])


class CollectMineralShards(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""

  def step(self, obs):
    super(CollectMineralShards, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      if not neutral_y.any() or not player_y.any():
        return actions.FunctionCall(_NO_OP, [])
      player = [int(player_x.mean()), int(player_y.mean())]
      closest, min_dist = None, None
      for p in zip(neutral_x, neutral_y):
        dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
        if not min_dist or dist < min_dist:
          closest, min_dist = p, dist
      return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])
    else:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])


class DefeatRoaches(base_agent.BaseAgent):
  """An agent specifically for solving the DefeatRoaches map."""

  def step(self, obs):
    super(DefeatRoaches, self).step(obs)
    if _ATTACK_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      roach_y, roach_x = (player_relative == _PLAYER_HOSTILE).nonzero()
      if not roach_y.any():
        return actions.FunctionCall(_NO_OP, [])
      index = numpy.argmax(roach_y)
      target = [roach_x[index], roach_y[index]]
      return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
    elif _SELECT_ARMY in obs.observation["available_actions"]:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
    else:
      return actions.FunctionCall(_NO_OP, [])