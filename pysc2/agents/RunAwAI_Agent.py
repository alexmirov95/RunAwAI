# Originally written by Kyle Chickering, Xin Jin, Alex Mirov and
# Simon Wu for ECS 170 @ UC Davis, Spring 2018.

"""
  RunAwAI Agent
  Will run away from enemy units.

  TEST USAGE:
    python -m pysc2.bin.agent --map DefeatRoaches --agent pysc2.agents.RunAwAI_Agent.RunAwAI
"""

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


class RunAwAI(base_agent.BaseAgent):
  maxMapWidth = 64
  maxMapHeight = 64
  
  obs = None
  target = [0] * 2 # uninitialized target value
  ct = -1 # target step count

  def getCurrentLocation(self):
    """
      returns [xcoord, ycoord] of current player location
    """
    player_relative = self.obs.observation["screen"][_PLAYER_RELATIVE]
    player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
    return [int(player_x.mean()), int(player_y.mean())]

  def getCurrentEnemyLocation(self):
    """
      returns [xcoord, ycoord] of current enemy location
    """
    player_relative = self.obs.observation["screen"][_PLAYER_RELATIVE]
    player_y, player_x = (player_relative == _PLAYER_HOSTILE).nonzero()
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
    # get current location
    currentLocation = self.getCurrentLocation()
    # how close the units will get to the destination before selecting a new target destination
    distToTarget = 5
    if (((abs(currentLocation[0] - target[0]) >= distToTarget) or 
        (abs(currentLocation[1] - target[1]) >= distToTarget)) and 
        (self.ct > 0)):
      # has not reached (or at least gotten close to) destination
      return {
        "function": actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target]),
        "status": "MOVING_TO_TARGET"
      }
    else:
      # close enough to destination for a new waypoint
      # set new target destination waypoint
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
    super(RunAwAI, self).step(obs)
    self.obs = obs

    time.sleep(0.03) # time to slep per step

    # checks to see if units can move, i.e. if they're selected
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      # call move function
      returnObj = self.moveToLocation()

      if returnObj["status"] is "ARRIVED_AT_TARGET":
        # arrived at target, update next target by incrementing location index (ct)

        # AI API INTEGRATION WILL GO HERE
        """
          Move in a predefined square
        """
        # squareLocations = [[int(10), int(10)], [int(60), int(15)], [int(60), int(50)], [int(10), int(50)]]
        # self.setTargetDestination(squareLocations[self.ct % len(squareLocations)])
        
        """
          Move a random distance in a random direction
        """
        movementDirectionActionSpace = ["NORTH","SOUTH", "EAST", "WEST", "NORTHEAST","SOUTHEAST","SOUTHWEST","NORTHWEST","STAY"]
        movementDirection = random.choice(movementDirectionActionSpace)
        stepSize = random.choice(range(1, 25))
        self.movementStep(movementDirection, stepSize)
        # print("MOVING", movementDirection, stepSize)

        """
          Charge the enemy units
        """
        # enemyLocation = self.getCurrentEnemyLocation()
        # self.setTargetDestination(enemyLocation) # charge the enemy!!

        self.ct += 1
        return returnObj["function"]

      elif returnObj["status"] is "MOVING_TO_TARGET":
        # enroute to target, do not update target
        return returnObj["function"]

    # select units
    else:
      # reset count
      if self.ct > 0:
        print("Survived", self.ct, "steps.")
      self.ct = 0
      print("Starting new simulation.")
      # Select all units
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
