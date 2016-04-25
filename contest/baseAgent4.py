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

    #Boundary
    boundary = None
    redBoundary = None
    blueBoundary = None

    #Direction encoding
    enc = {
        "Stop" : [0,0,0,0,1],
        "North": [0,0,0,1,0],
        "South": [0,0,1,0,0],
        "East" : [0,1,0,0,0],
        "West" : [1,0,0,0,0],
        }
    #
    movesDict = {
        "Stop" : (0, 0),
        "North": (0, 1),
        "South": (0, -1),
        "East" : (1, 0),
        "West" : (-1, 0)
        }

    def __init__(self, index, enemyFilter):
        CaptureAgent.__init__(self, index)
        self.enemyFilter = enemyFilter
        self.iter = 0

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
        self.computeOrder()
            
        #Map boundary
        if not baseAgent.redBoundary:
            walls = gameState.getWalls()
            baseAgent.w = walls.width
            baseAgent.h = walls.height
            baseAgent.redBoundary = filter(lambda (x,y): not gameState.hasWall(x,y),[(walls.width/2 - 1,i) for i in range(walls.height)])
            baseAgent.blueBoundary = filter(lambda (x,y): not gameState.hasWall(x,y),[(walls.width/2,i) for i in range(walls.height)])


    def chooseAction(self, gameState):
        """
        Choose action.
        """

        self.enemyFilter.addNewInfo(self.index, gameState)
        self.updateFoodPositions(gameState)
        self.updatePositions(gameState)

        carriedFood = []
        isPacman = []
        scared = []
        moveHeat = []
        for i in self.myOrderedTeam + self.enemyOrderedTeam:
            agentState = gameState.getAgentState(i)
            carriedFood.append(agentState.numCarrying)
            isPacman.append(agentState.isPacman)
            scared.append(agentState.scaredTimer)
            moveHeat.append(self.moveHeatMapValue(i))

        capsulesLeft = len(self.capsules)
        defCapsulesLeft = len(self.defCapsules)
        foodLeft = len(self.food)
        defFoodLeft = len(self.defFood)

        bannedMoves = []
        avoidWhenIn = 5
        #distance and move along the shortest path from my current agent to both enemies (1 + 5 + 1 + 5)
        dist1, direct1 = self.distanceAndDirectionToAgent(gameState, self.myOrderedTeam[0],self.enemyOrderedTeam[0])
        dist2, direct2 = self.distanceAndDirectionToAgent(gameState, self.myOrderedTeam[0],self.enemyOrderedTeam[1])
        if isPacman[0]:
            if not isPacman[2] and not (scared[2] > 5) and dist1 <= avoidWhenIn:
                bannedMoves.append(direct1)
            if not isPacman[3] and not (scared[3] > 5) and dist2 <= avoidWhenIn:
                bannedMoves.append(direct2)

        #distance and move along the shortest path to closest enemy food (1 + 5)
        dist3, direct3 = self.closestFoodFromAgent(gameState, self.myOrderedTeam[0], bannedMoves)

        dist31, direct31 = self.closestFoodFromAgentHalf(gameState, self.myOrderedTeam[0], True, bannedMoves)
        dist32, direct32 = self.closestFoodFromAgentHalf(gameState, self.myOrderedTeam[0], False, bannedMoves)
        if direct31 == "Stop":
            direct31 = direct3
        if direct32 == "Stop":
            direct32 = direct3

        #distance and move along the shortest path to closest enemy calsule (1 + 5)
        dist4, direct4 = self.closestCapsuleFromAgent(gameState, self.myOrderedTeam[0], bannedMoves)

        #distance and move along the shortest path to the boundary (1 + 5)
        dist5, direct5 = self.closestBoundaryFromAgent(gameState, self.myOrderedTeam[0], bannedMoves)
        dist50, direct50 = self.closestBoundaryFromAgent(gameState, self.myOrderedTeam[0])

        dist51, direct51 = self.closestBoundaryFromAgentHalf(gameState, self.myOrderedTeam[0], True)
        dist52, direct52 = self.closestBoundaryFromAgentHalf(gameState, self.myOrderedTeam[0], False)
        


        myBestMove = "Stop"
        movesLeft = 300 - self.iter
        score = self.getScore(gameState)
        tryToScore = 3
        print "agent", self.index
        if carriedFood[0] > 0 and movesLeft < dist5 + 5:
            #if i have food and not much time -> try to score
            myBestMove = direct5
        elif foodLeft <= 2: #
            #dangerous moment
            if defFoodLeft <= 2 and carriedFood[0] > 0:
                #try to score
                myBestMove = direct5
            else:
                #try co catch opponets with food
                if carriedFood[2] > 0:
                    myBestMove = direct1
                else:
                    myBestMove = direct2
        else:
            if score <=0:
                #play agressivly - both agents try to eat
                #TODO run away from packman
                if carriedFood[0] >= tryToScore:
                    myBestMove = direct5
                    print "try to Score"
                else:
                    if dist4!=0 and dist4 <= 5:
                        #try to eat capsule
                        print "eat capsule"
                        myBestMove = direct4
                    else:
                        #eat more
                        print "eat more"
                        if self.index < 2:
                            myBestMove = direct31
                        else:
                            myBestMove = direct32
            elif score > 5:
                #play safe - both agents defend
                #is see packman go for the one with highest food, otherwise go to the border
                if isPacman[2] or isPacman[3]:
                    myBestMove = direct1
                    if carriedFood[2]> carriedFood[2]:
                        myBestMove = direct2
                    print "defend kill pacman"
                else:
                    if self.index < 2:
                        myBestMove = direct51
                    else:
                        myBestMove = direct52
                    print "defend border"
            else:
                #one agent defending (0 or 1), one offending (2 or 3)
                if self.index < 2:
                    #TODO run away from packman
                    if carriedFood[0] >= tryToScore:
                        print "try to score"
                        myBestMove = direct5
                    else:
                        if dist4!=0 and dist4 <= 5:
                            #try to eat capsule
                            print "eat capsule"
                            myBestMove = direct4
                        else:
                            print "eat more"
                            myBestMove = direct3

                else:
                    #defend
                    #is see packman go for the one with highest food, otherwise go to the border
                    if isPacman[2] or isPacman[3]:
                        myBestMove = direct1
                        if carriedFood[2]> carriedFood[2]:
                            myBestMove = direct2
                        print "defend kill pacman"
                    else:
                        myBestMove = direct50
                        print "defend border"


        #stateM  = self.stateMatrix(gameState)
        #bestPos = np.unravel_index(np.argmax(stateM), stateM.shape)
        #bestDir = self.nextShortest(gameState, bestPos)
        
        self.iter += 1
        if myBestMove == "Stop":
            print "Stop"
        return myBestMove #bestDir


    def stateMatrix(self, gameState):
        """
        State matrix: weighted sum of value per position
        in grid.
        """

        #Movement heat map
        sM = 0.05*baseAgent.moveMap

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

    def computeOrder(self):
        #agents IDs in order of further moves
        agent1 = self.index
        enemy1 = 0 if agent1 == 3 else agent1 + 1
        agent2 = 0 if enemy1 == 3 else enemy1 + 1
        enemy2 = 0 if agent2 == 3 else agent2 + 1
        self.myOrderedTeam = [agent1, agent2]
        self.enemyOrderedTeam = [enemy1, enemy2]


    def updateFoodPositions(self, gameState):
        self.food = self.getFood(gameState).asList()
        self.defFood = self.getFoodYouAreDefending(gameState).asList()
        self.capsules = self.getCapsules(gameState)
        self.defCapsules = self.getCapsulesYouAreDefending(gameState)


    def updatePositions(self, gameState):
        self.positionByID = [None for i in range(4)]
        self.myOrderedTeamPos = []

        for agentID in self.myOrderedTeam:
            pos = gameState.getAgentState(agentID).getPosition()
            pos = (int(pos[0]), int(pos[1]))
            self.myOrderedTeamPos.append(pos)
            self.positionByID[agentID] = pos

        self.enemyOrderedTeamPos = []
        self.enemyBeliefState = self.enemyFilter.getBeliefStateBool()
        self.enemyBeliefStateAsList = {}
        exactPos = self.enemyFilter.getExactEnemyPositions()
        self.uncertainty = []
        for agentID in self.enemyOrderedTeam:
            self.enemyOrderedTeamPos.append(exactPos[agentID])
            self.positionByID[agentID] = exactPos[agentID]
            
            numOfStates = np.sum(self.enemyBeliefState[agentID])
            if numOfStates == 0:
                self.uncertainty.append(0.)
            else:
                self.uncertainty.append(1./numOfStates)
            
            stateList = []
            for x in xrange(baseAgent.w):
                for y in xrange(baseAgent.h):
                    if self.enemyBeliefState[agentID][x,y]:
                        stateList.append((x,y))
            self.enemyBeliefStateAsList[agentID] = stateList


    def closest(self, gameState, fromPos, toList, bannedMoves=[]):
        """
        Return closest distance and direction
        to closest element in toList from fromPos.
        """
        if len(toList) == 0:
            #return 0, Agent.enc["Stop"], None
            return 0, "Stop", None
        else:
            distances = [(toPos, self.getMazeDistance(fromPos, toPos)) for toPos in toList]
            toPos, dist = min(distances, key=lambda x: x[1])
            return dist, self.shortestPathMove(gameState, fromPos, toPos, bannedMoves), toPos


    def getValidMovesFromPos(self, gameState, pos):
        """
        Return array of valid moves from arbitrary position pos, example ["Stop", "North"]
        """
        validMoves = []

        if not gameState.hasWall(pos[0], pos[1]):
            for k, v in baseAgent.movesDict.items():
                x = pos[0] + v[0]
                y = pos[1] + v[1]
                inBounds = x > -1 and x < baseAgent.w and y > -1 and y < baseAgent.h
                if inBounds and not gameState.hasWall(x, y):
                    validMoves.append(k)
        return validMoves


    def shortestPathMove(self, gameState, fromPos, toPos, bannedMoves=[]):
        validMoves = self.getValidMovesFromPos(gameState, fromPos)
        dist = 1000
        bestMove = "Stop"
        for m in validMoves:
            if m not in bannedMoves:
                x = fromPos[0] + baseAgent.movesDict[m][0]
                y = fromPos[1] + baseAgent.movesDict[m][1]
                d = self.getMazeDistance((x,y), toPos)
                if d < dist:
                    dist = d
                    bestMove = m
        #return baseAgent.enc[bestMove]
        return bestMove


    def closestFoodFromAgent(self, gameState, fromAgent, bannedMoves=[]):
        """
        Return distance and direction of the closest food of other team.
        """
        fromList = []
        if self.positionByID[fromAgent] is not None:
            fromList = [self.positionByID[fromAgent]]
        else:
            fromList = self.enemyBeliefStateAsList[fromAgent]
        
        toList = []
        if fromAgent in self.myOrderedTeam:
            toList = self.food
        else:
            toList = self.defFood

        return self.closestDistanceAndMoveForLists(gameState, fromList, toList, bannedMoves)

    def closestFoodFromAgentHalf(self, gameState, fromAgent, top=True, bannedMoves=[]):
        """
        Return distance and direction of the closest food of other team.
        """
        fromList = []
        if self.positionByID[fromAgent] is not None:
            fromList = [self.positionByID[fromAgent]]
        else:
            fromList = self.enemyBeliefStateAsList[fromAgent]
        
        toList = []
        if fromAgent in self.myOrderedTeam:
            toList = self.food
        else:
            toList = self.defFood

        if top:
            toList = filter(lambda (x,y): y >(baseAgent.h/2- 1), toList)
        else:
            toList = filter(lambda (x,y): y <(baseAgent.h/2), toList)
        return self.closestDistanceAndMoveForLists(gameState, fromList, toList, bannedMoves)


    def closestCapsuleFromAgent(self, gameState, fromAgent,bannedMoves=[]):
        """
        Return distance and direction of the closest capsule of other team.
        """
        fromList = []
        if self.positionByID[fromAgent] is not None:
            fromList = [self.positionByID[fromAgent]]
        else:
            fromList = self.enemyBeliefStateAsList[fromAgent]
        
        toList = []
        if fromAgent in self.myOrderedTeam:
            toList = self.capsules
        else:
            toList = self.defCapsules
        
        return self.closestDistanceAndMoveForLists(gameState, fromList, toList, bannedMoves)


    def closestBoundaryFromAgent(self, gameState, fromAgent,bannedMoves=[]):
        """
        Return distance and direction of the closest boundary point.
        """
        fromList = []
        if self.positionByID[fromAgent] is not None:
            fromList = [self.positionByID[fromAgent]]
        else:
            fromList = self.enemyBeliefStateAsList[fromAgent]
        
        toList = []
        if fromAgent in self.myOrderedTeam:
            if self.red:
                toList = baseAgent.redBoundary
            else:
                toList = baseAgent.blueBoundary
        else:
            if self.red:
                toList = baseAgent.blueBoundary
            else:
                toList = baseAgent.redBoundary
        
        return self.closestDistanceAndMoveForLists(gameState, fromList, toList,bannedMoves)

    def closestBoundaryFromAgentHalf(self, gameState, fromAgent, top=True, bannedMoves=[]):
        """
        Return distance and direction of the closest boundary point.
        """
        fromList = []
        if self.positionByID[fromAgent] is not None:
            fromList = [self.positionByID[fromAgent]]
        else:
            fromList = self.enemyBeliefStateAsList[fromAgent]
        
        toList = []
        if fromAgent in self.myOrderedTeam:
            if self.red:
                toList = baseAgent.redBoundary
            else:
                toList = baseAgent.blueBoundary
        else:
            if self.red:
                toList = baseAgent.blueBoundary
            else:
                toList = baseAgent.redBoundary

        if top:
            toList = filter(lambda (x,y): y >(baseAgent.h/2- 1), toList)
        else:
            toList = filter(lambda (x,y): y <(baseAgent.h/2), toList)
        
        return self.closestDistanceAndMoveForLists(gameState, fromList, toList,bannedMoves)

    def distanceAndDirectionToAgent(self, gameState, fromAgent, toAgent, bannedMoves=[]):
        """
        Return distance and direction from one specified agent to another.
        If one of the agents is enemy without exact position, then computes closect possible over all possible states.
        """
        fromList = []
        toList = []
        if self.positionByID[fromAgent] is not None:
            fromList = [self.positionByID[fromAgent]]
        else:
            fromList = self.enemyBeliefStateAsList[fromAgent]

        if self.positionByID[toAgent] is not None:
            toList = [self.positionByID[toAgent]]
        else:
            toList = self.enemyBeliefStateAsList[toAgent]

        return self.closestDistanceAndMoveForLists(gameState, fromList, toList, bannedMoves)


    def distanceAndDirectionToPos(self, gameState, fromAgent, toPos, bannedMoves=[]):
        """
        Return distance and direction from one specified agent to position.
        If the agent is enemy without exact position, then computes closect possible over all possible states.
        """
        fromList = []
        toList = [toPos]
        
        if self.positionByID[fromAgent] is not None:
            fromList = [self.positionByID[fromAgent]]
        else:
            fromList = self.enemyBeliefStateAsList[fromAgent]

        return self.closestDistanceAndMoveForLists(gameState, fromList, toList, bannedMoves)


    def closestDistanceAndMoveForLists(self, gameState, fromList, toList, bannedMoves=[]):
        minDist = 1000
        #move = baseAgent.enc["Stop"]
        move = "Stop"
        for fromPos in fromList:
            dist, direc, _ = self.closest(gameState, fromPos, toList, bannedMoves)
            if dist < minDist:
                minDist = dist
                move = direc

        return minDist, move


    def moveHeatMapValue(self, agentID):
        """
        Return value of the move heat map at specified agent's position.
        If the agent is an enemy without exact position, then computes highest value over all possible states.
        """
        positions = []
        if self.positionByID[agentID] is not None:
            positions = [self.positionByID[agentID]]
        else:
            positions = self.enemyBeliefStateAsList[agentID]
        
        maxValue = 0.
        for pos in positions:
            maxValue = max(maxValue, baseAgent.moveMap[pos[0], pos[1]])

        return maxValue
     
