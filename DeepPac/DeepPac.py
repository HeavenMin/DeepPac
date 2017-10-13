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
               first = 'sillyAgent', second = 'DefensiveReflexAgent'):
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
def scaredTimeLeft(gameState, agent_index):
    return gameState.getAgentState(agent_index).scaredTimer

# (int, int) position(x, y)
def getAgentPosition(gameState, agent_index):
    return gameState.getAgentPosition(agent_index)

# whether two agent are across corners
def acrossCorners(pos1, pos2):
    return abs(pos1[0] - pos2[0]) == 1 and abs(pos1[1] - pos2[1])

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




        ######## Test Field #########
        if TEST_INFO_PRINT:
            print('Test')
        #############################

    #important. for choosing the action
    def chooseAction(self, gameState):
        #get legal action list
        actions = gameState.getLegalActions(self.index)

        # to evaluation time
        if TEST_INFO_PRINT:
            start_time = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        if TEST_INFO_PRINT:
            print 'eval time for agent %d: %.4f' % (self.index, time.time() - start_time)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        #get num of food left
        foodLeft = getFoodLeft(gameState, self)
        foodCarry = getFoodCarry(gameState, self.index)

        #back home
        if foodCarry >= 4:
            return self.backhome(gameState, actions)

        return random.choice(bestActions)

    def backhome(self, gameState, actions):
        bestDist = 9999
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(self.start,pos2)
            if dist < bestDist:
                bestAction = action
                bestDist = dist
        return bestAction

    def myaStarSearch(self, gameState, firstPos, goalPos):
        walls = gameState.getWalls()
        mapWidth = walls.width
        mapHeight = walls.height
        wallList = walls.asList()













    def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
        """
        Finds the distance between the agent with the given index and its nearest goalPosition
        """
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        walls = walls.asList()

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [Actions.directionToVector(action) for action in actions]
        # Change action vectors to integers so they work correctly with indexing
        actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

        # Values are stored a 3-tuples, (Position, Path, TotalCost)

        currentPosition, currentPath, currentTotal = startPosition, [], 0
        # Priority queue uses the maze distance between the entered point and its closest goal position to decide which comes first
        queue = util.PriorityQueueWithFunction(lambda entry: entry[2] +   # Total cost so far
                                               width * height if entry[0] in avoidPositions else 0 +  # Avoid enemy locations like the plague
                                               min(util.manhattanDistance(entry[0], endPosition) for endPosition in goalPositions))

        # Keeps track of visited positions
        visited = set([currentPosition])

        while currentPosition not in goalPositions:

            possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for vector, action in zip(actionVectors, actions)]
            legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]

            for position, action in legalPositions:
                if position not in visited:
                    visited.add(position)
                    queue.push((position, currentPath + [action], currentTotal + 1))

            # This shouldn't ever happen...But just in case...
            if len(queue.heap) == 0:
                return None
            else:
                currentPosition, currentPath, currentTotal = queue.pop()

        if returnPosition:
            return currentPath, currentPosition
        else:
            return currentPath


    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}

    def getnearestOurFood(self, gameState):
        our_food = self.getFoodYouAreDefending(gameState).asList()
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


        agentLocations = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState)]
        actions, po = self.aStarSearch(gameState.getAgentPosition(
                             self.index), gameState, self.getFood(gameState).asList(), agentLocations, True)
        draw(self, po)

        return actions[0]










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
