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
import random, time, util
from game import Directions
import game
from util import nearestPoint, matrixAsList
import numpy as np
from itertools import product

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Agent', second = 'Agent'):
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

class Agent(CaptureAgent):
  """
  Group 1's agent.
  """

  #Map width and height
  width  = None
  height = None

  #Moves heat map
  movesAlpha = 0.3
  moves      = np.array([])
  moveMap    = np.array([])

  #Food heat map
  foodAlpha     = 0.3
  foodMap       = np.array([])
  defendFoodMap = np.array([])

  #Avoid oponents alpha
  oppAlpha = 0.3

  #Index of team member defending
  defense = None

  @classmethod
  def heat_map(cls, gameState, agent, arr, alpha):
    """
    Return heat map.
    alpha: discounting factor, should be greater than zero.
    """
    heat_map = np.zeros((cls.width, cls.height))

    try:
      lst = arr.asList()
    except:
      w, h = arr.shape
      lst = [(i,j) for i,j in product(range(w),range(h))]

    for pos in lst:
      for pos2 in lst:
        try:
          dist = agent.getMazeDistance(pos, pos2)
          heat_map[pos] += 1.0/(1+alpha)**dist
        except:
          pass

    return heat_map/np.max(heat_map)

  @staticmethod
  def inverse_weight(agent, pos, lst, width, height, alpha):
    """
    Weight for elements in lst by
    distance to pos.
    """
    arr = np.zeros((width, height))
    for p in lst:
      dist = agent.getMazeDistance(pos, p)
      arr[p] = 1.0/(1+alpha)**dist

    return arr


  @staticmethod
  def neighbor_sum(gameState, dic):
    """
    Populates array with neighborhood sum based on dic.
    """
    walls = gameState.getWalls()
    arr = np.zeros((Agent.width, Agent.height))
    for i in range(walls.width):
      for j in range(walls.height):
        if walls[i][j]: continue
        m = dic.get((i,j), 1)
        if not walls[(i+1)][j]: m+=dic.get((i+1,j), 1)
        if not walls[(i-1)][j]: m+=dic.get((i-1,j), 1)
        if not walls[i][(j+1)]: m+=dic.get((i,j+1), 1)
        if not walls[i][(j-1)]: m+=dic.get((i,j-1), 1)
        arr[i,j] = m

    return arr/np.max(arr)

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

    '''
    Your initialization code goes here, if you need any.
    '''

    #Map width and height
    if Agent.width == None:
      walls        = gameState.getWalls()
      Agent.width  = walls.width
      Agent.height = walls.height

    #Pre-compute available moves
    if Agent.moves.size == 0:
      Agent.moves   = Agent.neighbor_sum(gameState, {})

    #Moves heat map
    if Agent.moveMap.size == 0:
      Agent.moveMap  = Agent.heat_map(gameState, self,
       Agent.moves, Agent.movesAlpha)

    #Food maps
    self.updateFoodMap(gameState)

    # import matplotlib.pyplot as plt 
    # plt.imshow(sM.T)
    # plt.colorbar()
    # plt.show()

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    stateM  = self.stateMatrix(gameState)
    bestPos = np.unravel_index(np.argmax(stateM), stateM.shape)
    bestDir = self.nextShortest(gameState, bestPos)

    return bestDir

  def stateMatrix(self, gameState):
    """
    State matrix: weighted sum of value per position
    in grid.
    """

    #Movement heat map
    sM = 0.1*Agent.moveMap

    #Food (defense/offense)
    self.updateFoodMap(gameState)
    if Agent.defense == None or Agent.defense==self.index:
      sM += Agent.defendFoodMap
      Agent.defense = self.index
    else:
      st = gameState.getAgentState(self.index)
      if not st.numCarrying:
        sM += Agent.foodMap
      else:
        sM += Agent.defendFoodMap

    #Opponents positions
    opponets  = self.getOpponents(gameState)
    opp_pos   = np.zeros(sM.shape)
    for pos, opp in [(gameState.getAgentPosition(opp),opp) for opp in opponets]:
      if pos != None:

        st = gameState.getAgentState(opp)
        if st.isPacman:
          opp_pos[pos] = 1.0
        else:
          lst = self.getFood(gameState).asList()
          width, height = opp_pos.shape
          opp_pos -= Agent.inverse_weight(self, pos, lst,
           width, height, Agent.oppAlpha)

    sM += 10.0*opp_pos

    return sM

  def nextS(self, gameState, myPos, pos, actions):
    """
    Return action to shorten distance between
    myPos and pos
    """
    myDist  = self.getMazeDistance(myPos, pos)

    #Vals
    vals = [('Stop',myDist)]
    for action in actions:
      successor = self.getSuccessor(gameState, action)
      newPos    = successor.getAgentState(self.index).getPosition()
      newDist   = self.getMazeDistance(newPos, pos)
      vals.append((action,newDist))

    return min(vals,key=lambda x: x[1])[0]


  def nextShortest(self, gameState, pos):
    """
    Return action to shorten distance between
    self and pos along shortest path.
    """

    #My position
    myPos   = gameState.getAgentState(self.index).getPosition()
    actions = gameState.getLegalActions(self.index)

    return self.nextS(gameState, myPos, pos, actions)

    

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def updateFoodMap(self, gameState):
    """
    Update food map.
    """

    food  = self.getFood(gameState)
    dfood = self.getFoodYouAreDefending(gameState)

    #Food map
    Agent.foodMap  = Agent.heat_map(gameState, self,
       food, Agent.foodAlpha)

    #Defend food map
    Agent.defendFoodMap  = Agent.heat_map(gameState, self,
       dfood, Agent.foodAlpha)