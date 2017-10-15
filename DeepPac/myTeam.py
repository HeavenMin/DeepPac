#!/usr/bin/env python2
# -*- coding: utf-8 -*-


"""
 TEAM : DeepPac
 PURPOSE : for pacman
 VERSION : 1
 DATE : 10.2017
"""

__author__ = 'DeepPac'

# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions, Actions
from util import nearestPoint
import game
from copy import deepcopy

#################
# Team creation #
#################

TEST_INFO_PRINT = False


def createTeam(firstIndex, secondIndex, isRed,
               first='sillyAgent', second='DeepPacDefence'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########
def checkPathExist(walls, start, destination):
    fringe = util.Queue()
    fringe.push(start)
    close = set()
    directs = [(1, 0), (-1, 0), (0, -1), (0, 1)]
    while not fringe.isEmpty():
        node = fringe.pop()
        if node == destination:
            return True, []
        if node not in close:
            close.add(node)
            for direct in directs:
                next_position = tuple((node[0] + direct[0], node[1] + direct[1]))
                if not walls[next_position[0]][next_position[1]]:
                    fringe.push(next_position)
    return False, close


# for debug, draw a debug square
def draw(agent, positions, color='r'):
    if color == 'r':
        agent.debugDraw(positions, (.9, 0, 0), True)
    if color == 'b':
        agent.debugDraw(positions, (0, .3, .9), True)
    if color == 'o':
        agent.debugDraw(positions, (.98, .41, .07), True)
    if color == 'g':
        agent.debugDraw(positions, (.1, .75, .7), True)
    if color == 'y':
        agent.debugDraw(positions, (1.0, 0.6, 0.0), True)


# a list of food you need to eat
def getFood(gameState, agent):
    return agent.getFood(gameState).asList()


# a list of food you need to defend
def getFoodYouAreDefending(gameState, agent):
    return agent.getFoodYouAreDefending(gameState).asList()


# int, num of food left
def getFoodLeft(gameState, agent):
    return len(agent.getFood(gameState).asList())


# int, number of carrying food for a agent
def getFoodCarry(gameState, agent_index):
    return gameState.getAgentState(agent_index).numCarrying


# True of False, if a agent is a pacman
def isPacman(gameState, agent_index):
    return gameState.getAgentState(agent_index).isPacman


# int, the scared time for a ghost left
def getScaredTimeLeft(gameState, agent_index):
    return gameState.getAgentState(agent_index).scaredTimer


# (int, int) position(x, y)
def getAgentPosition(gameState, agent_index):
    return gameState.getAgentPosition(agent_index)


# whether two agent are across corners, need to consider more
def acrossCorners(pos1, pos2):
    return abs(pos1[0] - pos2[0]) == 1 and abs(pos1[1] - pos2[1])


# return a list
def getEnemyPositions(gameState, agent):
    return [getAgentPosition(gameState, index) for index in agent.getOpponents(gameState)]


###############################################################################
##################################  Our Agent #################################
###############################################################################

class basicAgent(CaptureAgent):
    # init
    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        #### Our initialization code ####
        self.startPosition = gameState.getAgentPosition(self.index)
        self.walls = gameState.getWalls()
        self.wallsPosition = self.walls.asList(True)
        self.noWallsPosition = self.walls.asList(False)
        self.width = self.walls.width
        self.height = self.walls.height
        self.mapArea = self.width * self.height
        self.allActions = {Directions.EAST: (1, 0), Directions.SOUTH: (0, -1),
                           Directions.WEST: (-1, 0), Directions.NORTH: (0, 1)}
        self.enemyIndexs = self.getOpponents(gameState)
        self.ourAgentIndexs = self.getTeam(gameState)
        self.enemyStartPosition = getAgentPosition(gameState, self.enemyIndexs[0])
        self.bottleNeck = None
        self.food_abandon = set()
        self.in_neck_area = False

        self.initDefendFood = getFoodYouAreDefending(gameState, self)

        self.centreBoundary = self.width / 2
        self.isRedTeam = gameState.isOnRedTeam(self.index)
        if self.isRedTeam:
            self.centreBoundary -= 1
        self.homeEntries = [(x, y) for (x, y) in self.noWallsPosition if x == self.centreBoundary]

        ######## Test Field #########
        if False:
            print('#####Test Field#####')

            print(self.enemyStartPosition)

            print('######Test End######')
            exit()
            #############################

    # important. for choosing the action
    def chooseAction(self, gameState):
        # get legal action list
        actions = gameState.getLegalActions(self.index)
        # get num of food left
        foodLeft = getFoodLeft(gameState, self)
        foodCarry = getFoodCarry(gameState, self.index)

        # to evaluation time
        if TEST_INFO_PRINT:
            start_time = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        if TEST_INFO_PRINT:
            print 'eval time for agent %d: %.4f' % (self.index, time.time() - start_time)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        if TEST_INFO_PRINT:
            print('agent', self.index, maxValue)

        # back home
        if foodCarry >= 4:
            return self.backhome(gameState, actions)

        return random.choice(bestActions)

    def backhome(self, gameState, actions):
        bestDist = 9999
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(self.startPosition, pos2)
            if dist < bestDist:
                bestAction = action
                bestDist = dist
        return bestAction

    # a aStar method to find the best path from start position to goal position
    def aStarSearch(self, gameState, startPos, goalPos, enemyPos=[]):
        nowPos = startPos
        currentPath = []
        currentCost = 0

        # can also try util.manhattanDistance
        nodeList = util.PriorityQueueWithFunction(lambda arg: arg[0] + self.mapArea
        if arg[1] in enemyPos else 0 +
                                   min(self.getMazeDistance(arg[1], position) for position in goalPos))
        visitedList = set([nowPos])

        while nowPos not in goalPos:
            nextPositions = [((nowPos[0] + vec[0], nowPos[1] + vec[1]),
                              action) for action, vec in self.allActions.items()]
            legalPositions = [(p, a) for p, a in nextPositions if p not in self.wallsPosition]
            for p, a in legalPositions:
                if p not in visitedList:
                    visitedList.add(p)
                    nodeList.push((currentCost + 1, p, currentPath + [a]))

            if len(nodeList.heap) == 0:
                return None
            else:
                currentCost, nowPos, currentPath = nodeList.pop()
        return currentPath, nowPos

    # Finds the next successor which is a grid position (location tuple).
    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # Computes a linear combination of features and feature weights
    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    # Returns a counter of features for the state
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}

    def getnearestOurFood(self, gameState):
        our_food = getFoodYouAreDefending(gameState, self)
        our_food_distance = [self.getMazeDistance(gameState.getAgentPosition(
            self.index), food) for food in our_food]

        nearest_food = our_food[0]
        nearest_distance = our_food_distance[0]

        for i, food_distance in enumerate(our_food_distance):
            if food_distance < nearest_distance:
                nearest_food = our_food[i]
                nearest_distance = our_food_distance[i]
        return nearest_food

    def isGhost(self, gameState, index):
        """
        Returns true ONLY if we can see the agent and it's definitely a ghost
        """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False
        return not (gameState.isOnRedTeam(index) ^ (position[0] < gameState.getWalls().width / 2))

    def isScared(self, gameState, index):
        """
        Says whether or not the given agent is scared
        """
        isScared = bool(gameState.data.agentStates[index].scaredTimer)
        return isScared


class sillyAgent(basicAgent):
    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights, features

    def getFeatures(self, gameState, action):
        features = util.Counter()
        if getAgentPosition(gameState, self.index) == self.startPosition:
            print "you died"

        if action == 'Stop':
            features['stop'] = 1
        successor = self.getSuccessor(gameState, action)

        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        enemyGhostLocations = [gameState.getAgentPosition(i) for i in self.enemyIndexs if
                               self.isGhost(gameState, i) and not self.isScared(gameState, i)]
        if len(foodList) > 0:
            features['successorScore'] = -len(foodList)
            # print checkPathExist(self.walls,getAgentPosition(gameState,self.index),foodList[0])
            minFoodDistance = min(
                [len(self.aStarSearch(gameState, myPos, [food], enemyGhostLocations)[0]) for food in foodList if
                 food not in self.food_abandon] or [100])
            print minFoodDistance
        if minFoodDistance == 100:
            self.food_abandon = set()
        features['distanceToFood'] = minFoodDistance

        myPos = successor.getAgentState(self.index).getPosition()
        enemyGhostLocations = [gameState.getAgentPosition(i) for i in self.enemyIndexs if
                               self.isGhost(gameState, i) and not self.isScared(gameState, i)]
        min_capsules_distance = min(
            [self.getMazeDistance(myPos, food) for food in self.getCapsules(gameState)] or [100])
        features['distanceToCapsules'] = min_capsules_distance * 0.5 * getFoodCarry(gameState, self.index)
        neareast_enemy = None
        if (len(enemyGhostLocations) > 0):
            # print myPos
            a = [self.getMazeDistance(myPos, p) for p in enemyGhostLocations]
            # print enemyGhostLocations
            # print a
            neareast_enemy = min(a)
            if myPos == self.startPosition or neareast_enemy <= 1:
                features['deadArea'] = 10
        self.getCapsules(gameState)
        if isPacman(gameState, self.index) and getFoodCarry(gameState, self.index) > 1:
            cur_position = getAgentPosition(gameState, self.index)
            next_position = getAgentPosition(successor, self.index)
            cur_return_distance = len(
                self.aStarSearch(gameState, cur_position, [self.startPosition], enemyGhostLocations)[0])
            next_return_distance = len(
                self.aStarSearch(gameState, next_position, [self.startPosition], enemyGhostLocations)[0])
            sub_return_distance = cur_return_distance - next_return_distance
            features['return_home'] = 10 * getFoodCarry(gameState, self.index) * sub_return_distance
        if isPacman(gameState, self.index) and not self.in_neck_area:
            new_wall = getAgentPosition(gameState, self.index)
            pre_position = getAgentPosition(successor, self.index)
            new_walls = gameState.getWalls()
            new_walls[new_wall[0]][new_wall[1]] = True
            has_path, close_set = checkPathExist(new_walls, pre_position, self.startPosition)
            new_walls[new_wall[0]][new_wall[1]] = False
            if not has_path:
                features['bottleneck'] = new_wall
                features['abondonArea'] = close_set
                if neareast_enemy is not None and (neareast_enemy - 1) / 2 < minFoodDistance:
                    features['avoidArea'] = 1
                    minFoodDistance = min(
                        [len(self.aStarSearch(gameState, myPos, [food], enemyGhostLocations)[0]) for food in foodList if
                         food not in self.food_abandon] or [100])
        if self.in_neck_area:
            exit_distance = self.getMazeDistance(myPos, self.bottleNeck)
            if neareast_enemy is not None and (neareast_enemy - 1) / 2 < exit_distance:
                features['avoidArea'] = 1
                self.food_abandon = self.food_abandon | self.close_set
                minFoodDistance = min(
                    [len(self.aStarSearch(gameState, myPos, [food], enemyGhostLocations)[0]) for food in foodList if
                     food not in self.food_abandon] or [100])
        # print action
        features['distanceToFood'] = minFoodDistance
        # print self.bottleNeck
        # print features * self.getWeights(gameState,action)
        # if 'deadArea' in features:
        # print action
        # print features
        # print features * self.getWeights(gameState, action)
        return features

    def getWeights(self, gameState, action):
        return {'distanceToCapsules': -0.5, 'successorScore': 100, 'distanceToFood': -1, 'attenuation': 0.8,
                'deadArea': -99999, 'avoidArea': -9999, 'return_home': 1}

    def chooseAction(self, gameState):
        # agentLocations = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState)]
        # actions, po = self.astarSearch(gameState.getAgentPosition(
        #     self.index), gameState, self.getFood(gameState).asList(), agentLocations, True)
        # draw(self, po)

        actions = gameState.getLegalActions(self.index)
        features = [self.getFeatures(gameState, a) for a in actions]
        weight = self.getWeights(gameState, actions[0])
        values = [feature * weight for feature in features]
        # values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [(a, f) for a, v, f in zip(actions, values, features) if v == maxValue]
        foodCarry = getFoodCarry(gameState, self.index)
        # print(self.getSuccessorScore(gameState, bestActions[0]))

        # back home

        action = bestActions[0]
        if not self.in_neck_area and 'bottleneck' in action[1]:
            self.bottleNeck = action[1]['bottleneck']
            self.in_neck_area = True
            self.close_set = action[1]['abondonArea']
        if self.in_neck_area and getAgentPosition(self.getSuccessor(gameState, action[0]),
                                                  self.index) == self.bottleNeck:
            self.bottleNeck = None
            self.in_neck_area = False
            self.close_set = set()
        print "The action we choose is %s", action[0]
        return action[0]

    def getSuccessorScore(self, gameState, action, depth=2):
        if depth == 0:
            return self.evaluate(gameState, action)
        else:
            successor = self.getSuccessor(gameState, action)
            actions = successor.getLegalActions(self.index)
            return self.evaluate(gameState, action) + max(
                [self.getSuccessorScore(successor, a, depth - 1) for a in actions])




class DeepPacDefence(basicAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman:
        features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP:
        features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev:
        features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

  def chooseAction(self, gameState):
      actions = gameState.getLegalActions(self.index)
      position = gameState.getAgentPosition(self.index)
      num_defendFoodLeft = len(getFoodYouAreDefending(gameState, self))

      # if scared, wait in boundary
      if self.isScared(gameState, self.index):
         waitInBound, _ = self.aStarSearch(gameState, position, self.homeEntries)
         if waitInBound != []:
             return waitInBound[0]

      # to evaluation time
      if TEST_INFO_PRINT:
          start_time = time.time()
      #denfence for the food enemy most want
      if isPacman(gameState, self.enemyIndexs[0]) and isPacman(gameState, self.enemyIndexs[1]):
          e1Pos = getAgentPosition(gameState, self.enemyIndexs[0]) if getAgentPosition(gameState, self.enemyIndexs[0]) != None else self.enemyStartPosition
          e2Pos = getAgentPosition(gameState, self.enemyIndexs[1]) if getAgentPosition(gameState, self.enemyIndexs[1]) != None else self.enemyStartPosition
          if self.getMazeDistance(position, e1Pos) <= self.getMazeDistance(position, e2Pos):
              _, foodPosition = self.aStarSearch(gameState, e1Pos, getFoodYouAreDefending(gameState, self))
              ActionToDefence, _ = self.aStarSearch(gameState, position, [foodPosition])
          else:
               _, foodPosition = self.aStarSearch(gameState, e2Pos, getFoodYouAreDefending(gameState, self))
               ActionToDefence, _ = self.aStarSearch(gameState, position, [foodPosition])
      elif isPacman(gameState, self.enemyIndexs[0]):
          e1Pos = getAgentPosition(gameState, self.enemyIndexs[0]) if getAgentPosition(gameState, self.enemyIndexs[0]) != None else self.enemyStartPosition
          _, foodPosition = self.aStarSearch(gameState, e1Pos, getFoodYouAreDefending(gameState, self))
          ActionToDefence, _ = self.aStarSearch(gameState, position, [foodPosition])
      elif isPacman(gameState, self.enemyIndexs[1]):
          e2Pos = getAgentPosition(gameState, self.enemyIndexs[1]) if getAgentPosition(gameState, self.enemyIndexs[1]) != None else self.enemyStartPosition
          _, foodPosition = self.aStarSearch(gameState, e2Pos, getFoodYouAreDefending(gameState, self))
          ActionToDefence, _ = self.aStarSearch(gameState, position, [foodPosition])
      else:
          _, foodPosition = self.aStarSearch(gameState, self.enemyStartPosition, getFoodYouAreDefending(gameState, self))
          ActionToDefence, _ = self.aStarSearch(gameState, position, [foodPosition])

      if TEST_INFO_PRINT:
          print 'eval defence time for agent %d: %.4f' % (self.index, time.time() - start_time)

      if position != foodPosition:
          action = ActionToDefence[0]
          return action

      #get legal action list
      actions = gameState.getLegalActions(self.index)
      #get num of food left
      foodLeft = getFoodLeft(gameState, self)
      foodCarry = getFoodCarry(gameState, self.index)

      # to evaluation time
      values = [self.evaluate(gameState, a) for a in actions]

      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      if TEST_INFO_PRINT:
          print('agent', self.index, maxValue)

      return random.choice(bestActions)

  def isPacman(self, gameState, index):
      #Returns true ONLY if we can see the agent and it's definitely a pacman
      position = gameState.getAgentPosition(index)
      print(position)
      if position is None:
          return False
      return not (gameState.isOnRedTeam(index) ^ (position[0] >= gameState.getWalls().width / 2))




#END
