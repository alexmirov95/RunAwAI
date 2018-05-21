
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


class MoveTest(base_agent.BaseAgent):
  ct = 0 # target step count
  target = [0] * 2 # uninitialized target value
  locations = [[int(10), int(10)], [int(60), int(15)], [int(60), int(50)], [int(10), int(50)]]
  maxMapWidth = 64
  maxMapHeight = 64
  obs = None

  def getCurrentLocation(self):
      """
        get player locations
        returns [xcoord, ycoord] of current location
      """
      player_relative = self.obs.observation["screen"][_PLAYER_RELATIVE]
      player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      return [int(player_x.mean()), int(player_y.mean())]

  def moveToLocation (self):
    """
      Moves selected unit(s) to self.target location. Returns an object with the
      movement status of the units, i.e. if they're still enroute to target, or
      if they have arrived, and the return function that actually invokes the 
      movement. 
      The returnObject["function"] must be returned in the main step()
      function to invoke movement.
    """
    target = self.target
    sleepTime = 0.03
    # get current location
    currentLocation = self.getCurrentLocation()
    # how close the units will get to the destination before selecting a new target destination
    distToTarget = 5
    if (((abs(currentLocation[0] - target[0]) >= distToTarget) or 
        (abs(currentLocation[1] - target[1]) >= distToTarget)) and 
        (self.ct > 0)):
      # has not reached (or at least gotten close to) destination
      time.sleep(sleepTime)
      return {
        "function": actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target]),
        "status": "MOVING_TO_TARGET"
      }
    else:
      # close enough to destination for a new waypoint
      # set new target destination waypoint
      time.sleep(sleepTime)
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
    if ((x > self.maxMapWidth) or (0 > x) or 
      (y > self.maxMapHeight) or (0 > y)):
      print("Invalid target coordinates.")
    self.target = [x, y]


  def movementStep (self, direction, distance):
    """
      Will have selected unit(s) move a specified distnace in the
      directions "NORTH", "SOUTH", "EAST", "WEST", or "STAY"
    """
    newTarget = self.getCurrentLocation()

    if direction is "NORTH":
      newTarget[1] += distance
    elif direction is "EAST":
      newTarget[0] += distance
    elif direction is "SOUTH":
      newTarget[1] -= distance
    elif direction is "WEST":
      newTarget[0] -= distance
    elif direction is "NORTHEAST":
      newTarget[0] += distance
      newTarget[1] += distance
    elif direction is "SOUTHEAST":
      newTarget[0] += distance
      newTarget[1] -= distance
    elif direction is "SOUTHWEST":
      newTarget[0] -= distance
      newTarget[1] -= distance
    elif direction is "NORTHWEST":
      newTarget[0] -= distance
      newTarget[1] += distance

    elif direction is "STAY":
      # no movement from current location
      newTarget = newTarget
    else:
      print("Invalid Direction")

    # cap map bounds of new target within map dimensions
    borderLimit = 4 # target will not be set within borderLimit distance of the edge of map
    if newTarget[0] >= self.maxMapHeight - borderLimit:
      newTarget[0] = self.maxMapHeight - borderLimit
    if newTarget[1] >= self.maxMapWidth - borderLimit:
      newTarget[1] = self.maxMapWidth - borderLimit
    if newTarget[0] <= borderLimit:
      newTarget[0] = borderLimit
    if newTarget[1] <= borderLimit:
      newTarget[1] = borderLimit

    self.setTargetDestination(newTarget)


  def step(self, obs):
    super(MoveTest, self).step(obs)
    self.obs = obs

    # checks to see if units can move, i.e. if they're selected
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      # call move function
      returnObj = self.moveToLocation()

      if returnObj["status"] is "ARRIVED_AT_TARGET":
        # arrived at target, update next target by incrementing location index (ct)

        
        # AI API WILL GO HERE TO SET DESTINATION!!!!!
        # self.setTargetDestination(self.locations[self.ct % len(self.locations)])
        movementDirection = random.choice(["NORTH","SOUTH", "EAST", "WEST", "NORTHEAST","SOUTHEAST","SOUTHWEST","NORTHWEST","STAY"])
        stepSize = random.choice(range(1, 25))
        self.movementStep(movementDirection, stepSize)
        print("MOVING", movementDirection, stepSize)

        self.ct += 1
        return returnObj["function"]

      elif returnObj["status"] is "MOVING_TO_TARGET":
        # enroute to target, do not update target
        return returnObj["function"]

    # select units
    else:
      # reset count
      self.ct = 0
      print("Starting new simulation.")
      # Select all units
      print("Selecting all units")
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
