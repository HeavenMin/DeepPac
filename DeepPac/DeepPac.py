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

#################
# Team creation #
#################

TEST_INFO_PRINT = True

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DeepPacOffense', second = 'DefensiveReflexAgent'):
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

# for debug, draw a debug square
def draw(agent, positions, color='r'):
    if color == 'r':
        agent.debugDraw(positions, (.9,0,0), True)
    if color == 'b':
        agent.debugDraw(positions, (0,.3,.9), True)
    if color == 'o':
        agent.debugDraw(positions, (.98,.41,.07), True)
    if color == 'g':
        agent.debugDraw(positions, (.1,.75,.7), True)
    if color == 'y':
        agent.debugDraw(positions, (1.0,0.6,0.0), True)

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

    #init
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




        ######## Test Field #########
        if False:
            print('#####Test Field#####')

            print(getEnemyPositions(gameState, self))

            print('######Test End######')
            exit()
        #############################

    #important. for choosing the action
    def chooseAction(self, gameState):
        #get legal action list
        actions = gameState.getLegalActions(self.index)
        #get num of food left
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

        #back home
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
        nodeList = util.PriorityQueueWithFunction(lambda arg:arg[0] + self.mapArea
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

    #Finds the next successor which is a grid position (location tuple).
    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    #Computes a linear combination of features and feature weights
    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    #Returns a counter of features for the state
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

class sillyAgent(basicAgent):

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)

        if len(foodList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minFoodDistance

        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}

    def chooseAction(self, gameState):


        enemyLocations = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState)]

        actions, po = self.aStarSearch(gameState, getAgentPosition(gameState, self.index),
                            getFood(gameState, self), enemyLocations)
        draw(self, po)

        return actions[0]


class DeepPacOffense(basicAgent):
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = getFood(successor, self)

        myPos = getAgentPosition(successor, self.index)
        minDisToFood = min([self.getMazeDistance(myPos, food) for food in foodList])

        features['score'] = self.getScore(successor)
        features['foodLeft'] = -len(foodList)
        features['distanceToFood'] = minDisToFood


        minEnemyDistance = self.mapArea
        enemyPositions = getEnemyPositions(gameState, self)
        for enemy in enemyPositions:
            if enemy != None:
                if self.getMazeDistance(myPos, enemy) < minEnemyDistance:
                    minEnemyDistance = self.getMazeDistance(myPos, enemy)
        print(minEnemyDistance)

        if minEnemyDistance <= 3 :
            features['escape'] = minEnemyDistance
        else:
            features['escape'] = 0






        return features

    def getWeights(self, gameState, action):
        return {'score': 1.0,
                'foodLeft': 100,
                'distanceToFood': -1.5,
                'escape': -99}









# copy code need delete
class DefensiveReflexAgent(basicAgent):
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
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


#END
