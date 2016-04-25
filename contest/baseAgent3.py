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
    Initialization.
    """
    CaptureAgent.registerInitialState(self, gameState)

    #Map width and height
    if baseAgent.width == None:
      walls        = gameState.getWalls()
      baseAgent.width  = walls.width
      baseAgent.height = walls.height
      
    #Blue/Red boundaries
    baseAgent.redBoundary = [(baseAgent.width/2 - 1,i) \
    for i in range(baseAgent.height)]
    
    baseAgent.blueBoundary = [(baseAgent.width/2,i) \
    for i in range(baseAgent.height)]

    #Pre-compute available moves
    if baseAgent.moves.size == 0:
      baseAgent.moves   = baseAgent.neighbor_sum(gameState, {})

    #Moves heat map
    if baseAgent.moveMap.size == 0:
      baseAgent.moveMap  = baseAgent.heat_map(gameState, self,
       baseAgent.moves, baseAgent.movesAlpha)

    #Food maps
    self.updateFoodMap(gameState)

    #Enemy filter
    self.enemyFilter.addInitialGameStateInfo(self.index, gameState)
    
  def chooseAction(self, gameState):
    """
    Choose action.
    """

    #Update filter
    self.enemyFilter.addNewInfo(self.index, gameState)

    #Find best move
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
    if baseAgent.defense!=self.index:
      sM = 0.1*baseAgent.moveMap
    else:
      sM = 0.3*baseAgent.moveMap

    #Food (defense/offense)
    self.updateFoodMap(gameState)
    if baseAgent.defense == None or baseAgent.defense==self.index:
      sM += baseAgent.defendFoodMap
      baseAgent.defense = self.index
    else:
      st = gameState.getAgentState(self.index)
      if not st.numCarrying and not st.isPacman:
        sM += baseAgent.foodMap
      else: #Capture more food or take what we have back
        posFood, distFood = self.closestFood(gameState)
        sM[posFood] = baseAgent.foodMap[posFood]
        
        posBound, distBound = self.closestBoundary(gameState)
        sM[posBound] += 0.5*st.numCarrying
        
    #Capsules
    if baseAgent.defense!=self.index:
      capsules = self.getCapsules(gameState)
    else:
      capsules = self.getCapsulesYouAreDefending(gameState)

    #Opponents positions
    try:
      baseAgent.opp_pos = self.enemyFilter.getBeliefStateProb()
    except ZeroDivisionError:
      print "ZDE"
    
    myPos = gameState.getAgentState(self.index).getPosition()
    for opp in baseAgent.opp_pos.keys():
      
      st = gameState.getAgentState(self.index)
      opp_st = gameState.getAgentState(opp)
      if opp_st.isPacman and not st.scaredTimer: #Chase if pacman and not scared
        sM += 3*baseAgent.opp_pos[opp]
      elif opp_st.isPacman: #Run away if pacman and scared
        sM -= 7*baseAgent.opp_pos[opp]
        
      #Capsules importance
      opp_pos = gameState.getAgentState(opp).getPosition()
      for pos in capsules:
        dist = self.getDist(myPos, opp_pos)
        sM[pos] += 1.0/dist
        if baseAgent.defense!=self.index:
          sM[pos] *= 1.5

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
       
  def closestFood(self, gameState):
    """
    Get distance and position of closest food.
    """
    food = self.getFood(gameState).asList()
    myPos     = gameState.getAgentState(self.index).getPosition()
    distances = [(pos, self.getDist(myPos, pos)) for pos in food]
    return min(distances, key= lambda x: x[1])
       
  def closestBoundary(self, gameState):
    """
    Get distance and position of closest boundary point.
    """
    if self.red:
      boundary = baseAgent.redBoundary
    else:
      boundary = baseAgent.blueBoundary
      
    myPos     = gameState.getAgentState(self.index).getPosition()
    distances = [(pos, self.getDist(myPos, pos)) for pos in boundary]
    
    return min(distances, key= lambda x: x[1])
       
  def getDist(self, pos1, pos2):
    """
    Get Maze Distance, handle positions not in maze
    by returning inf
    """
    try:
      return self.getMazeDistance(pos1, pos2)
    except:
      return float("inf")