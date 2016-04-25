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
from game import Directions
import game
from util import nearestPoint, matrixAsList
import numpy as np
from itertools import product
from ai_filter import Filter

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'baseAgent', second = 'baseAgent'):
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
  enemyFilter = Filter()
  # The following line is an example only; feel free to change it.
  #return [eval(first)(firstIndex), eval(second)(secondIndex)]
  return [baseAgent(firstIndex, enemyFilter), baseAgent(secondIndex, enemyFilter)]

##########
# Agents #
##########

class baseAgent(CaptureAgent):
  """
  Group 1's agent.
  """

  def __init__(self, index, enemyFilter):
    CaptureAgent.__init__(self, index)
    self.enemyFilter = enemyFilter
    self.iter = 0

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
  opponents = None
  oppAlpha  = 0.3

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
  def inverse_weight(agent, P, lst, width, height, alpha):
    """
    Weight for elements in lst by
    distance to pos in P.
    """
    arr = np.zeros((width, height))
    for p in lst:
        for pos in P:
          pos = tuple(pos)
          dist = agent.getMazeDistance(pos, p)
          arr[p] += 1.0/(1+alpha)**dist

    return arr/np.max(arr)


  @staticmethod
  def neighbor_sum(gameState, dic):
    """
    Populates array with neighborhood sum based on dic.
    """
    walls = gameState.getWalls()
    arr = np.zeros((baseAgent.width, baseAgent.height))
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
    if baseAgent.width == None:
      walls        = gameState.getWalls()
      baseAgent.width  = walls.width
      baseAgent.height = walls.height

    #Pre-compute available moves
    if baseAgent.moves.size == 0:
      baseAgent.moves   = baseAgent.neighbor_sum(gameState, {})

    #Moves heat map
    if baseAgent.moveMap.size == 0:
      baseAgent.moveMap  = baseAgent.heat_map(gameState, self,
       baseAgent.moves, baseAgent.movesAlpha)

    #Food maps
    self.updateFoodMap(gameState)

    self.enemyFilter.addInitialGameStateInfo(self.index, gameState)

  def chooseAction(self, gameState):
    """
    Choose action.
    """

    self.enemyFilter.addNewInfo(self.index, gameState)
    '''
    To get beliefState as dict of np.arrays use:
    self.enemyFilter.getBeliefStateProb() - uniform probability over possible states
    self.enemyFilter.getBeliefStateBool() - bool (True if can be in particular state, False otherwise)
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
    sM = 0.1*baseAgent.moveMap

    #Food (defense/offense)
    self.updateFoodMap(gameState)
    if baseAgent.defense == None or baseAgent.defense==self.index:
      sM += baseAgent.defendFoodMap
      baseAgent.defense = self.index
    else:
      st = gameState.getAgentState(self.index)
      #opp_st = gameState.getAgentState(ind)
      #if opp_st.isPacman and not st.scaredTimer:
      if not st.numCarrying:
        sM += baseAgent.foodMap
      else:
        sM += baseAgent.defendFoodMap

    #Opponents positions
    try:
      baseAgent.opp_pos = self.enemyFilter.getBeliefStateProb()
    except ZeroDivisionError:
      print "ZDE"
    
    for opp in baseAgent.opp_pos.keys():
      
      st = gameState.getAgentState(opp)
      if st.isPacman:
        sM += 3*baseAgent.opp_pos[opp]
      elif baseAgent.defense!=self.index:
        P = np.transpose(baseAgent.opp_pos[opp].nonzero())
        lst = self.getFood(gameState).asList()
        width, height = sM.shape
        sM -= baseAgent.inverse_weight(self, P, lst,
         width, height, baseAgent.oppAlpha)

    return sM

  def nextS(self, gameState, myPos, pos, actions):
    """
    Return action to shorten distance between
    myPos and pos
    """
    myDist  = self.getDist(myPos, pos)

    #Vals
    vals = [('Stop',myDist)]
    for action in actions:
      successor = self.getSuccessor(gameState, action)
      newPos    = successor.getAgentState(self.index).getPosition()
      newDist   = self.getDist(newPos, pos)
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
    baseAgent.foodMap  = baseAgent.heat_map(gameState, self,
       food, baseAgent.foodAlpha)

    #Defend food map
    baseAgent.defendFoodMap  = baseAgent.heat_map(gameState, self,
       dfood, baseAgent.foodAlpha)
       
  def getDist(self, pos1, pos2):
    """
    Get Maze Distance, handle positions not in maze
    by returning inf
    """
    try:
      return self.getMazeDistance(pos1, pos2)
    except:
      return float("inf")