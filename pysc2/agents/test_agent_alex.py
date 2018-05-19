
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
import random

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


def get_location(obs):
    # get player locations
    player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
    player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
    return [int(player_x.mean()), int(player_y.mean())]

class MoveTest(base_agent.BaseAgent):
  ct = 0 # target step count
  target = [0] * 2 # uninitialized target value
  locations = [[int(10), int(10)], [int(60), int(15)], [int(60), int(50)], [int(10), int(50)]]

  def moveToLocation (self, obs):
    target = self.target
    # get current location
    currentLocation = get_location(obs)
    # how close the units will get to the destination before selecting a new target destination
    distToTarget = 5
    if (((abs(currentLocation[0] - target[0]) >= distToTarget) or 
        (abs(currentLocation[1] - target[1]) >= distToTarget)) and 
        (self.ct > 0)):
      # has not reached (or at least gotten close to) destination
      # no op
      time.sleep(0.2)
      return {
        "function": actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target]),
        "status": "MOVING_TO_TARGET"
      }
    else:
      # close enough to destination for a new waypoint
      # set new target destination waypoint
      print("Arrived at target ", self.target)
      time.sleep(0.2)
      return {
        "function": actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target]),
        "status": "ARRIVED_AT_TARGET"
      }

  def setTargetDestination (self, coordinates):
    """
      Sets the target destination to an x and y tuple in coordinates
    """
    x = coordinates[0]
    y = coordinates[1]
    if ((x > 64) or (0 > x) or 
      (y > 64) or (0 > y)):
      print("Invalid target coordinates.")
    self.target = [x, y]

  def step(self, obs):
    super(MoveTest, self).step(obs)

    if _MOVE_SCREEN in obs.observation["available_actions"]:
      returnObj = self.moveToLocation(obs)

      if returnObj["status"] is "ARRIVED_AT_TARGET":
        # arrived at target, update next target by incrementing location index (ct)
        self.ct += 1
        if self.ct >= 0:

          # AI API WILL GO HERE TO SET DESTINATION!!!!!
          self.setTargetDestination(self.locations[self.ct % len(self.locations)])

          print("SETTING new target to: ", self.target)
        return returnObj["function"]

      elif returnObj["status"] is "MOVING_TO_TARGET":
        # enroute to target, do not update target
        return returnObj["function"]

    else:
      # reset count
      self.ct = 0
      print("Starting new simulation.")
      # Select all units
      print("Selecting units")
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

      # Selection Single Unit
      # return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
