from baseAgent2 import baseAgent
from ai_filter import Filter
import numpy as np
import sqlite3
import pickle as pkl
from Qdb import QDB
# from sknn import mlp

#################
# Team creation #
#################

EXPLORE = 0.3
#
# 0 - play, no learning
# 1 - learning, moves based on baseAgent2
# 2 - learning, moves based on maxMinQ
#
MODE = 0

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Agent', second = 'Agent'):
    enemyFilter = Filter()
    return [Agent(firstIndex, enemyFilter), Agent(secondIndex, enemyFilter)]


class Agent(baseAgent):
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
    moves = {
        "Stop" : (0, 0),
        "North": (0, 1),
        "South": (0, -1),
        "East" : (1, 0),
        "West" : (-1, 0)
        }
  #Neural Net


    def registerInitialState(self, gameState):
        baseAgent.registerInitialState(self, gameState)
        self.enemyFilter.addInitialGameStateInfo(self.index, gameState)
        self.computeOrder()

        #Map boundary
        if not Agent.redBoundary:
            walls = gameState.getWalls()
            Agent.w = walls.width
            Agent.h = walls.height
            Agent.redBoundary = filter(lambda (x,y): not gameState.hasWall(x,y),[(walls.width/2 - 1,i) for i in range(walls.height)])
            Agent.blueBoundary = filter(lambda (x,y): not gameState.hasWall(x,y),[(walls.width/2,i) for i in range(walls.height)])


    @staticmethod
    def Q(X, a, o):
        """
        Q function (neural net)
        """
        try:
          with open("Qnet.pkl","rb") as f:
            Qnet = pkl.load(f)
          return Qnet.predict((X,Agent.enc[a],Agent.enc[o]))
        except:
          return np.random.randn() * 0.1


    @staticmethod
    def maxMinQMove(X, myActions = None, enemyActions = None):
        if myActions is None:
            myActions = Agent.enc.keys()
        if enemyActions is None:
            enemyActions = Agent.enc.keys()

        maxQ = -1000
        myMove = "Stop"
        for a in myActions:
            minQ = 1000
            for o in enemyActions:
                Qval = Agent.Q(X, a, o)
                minQ = min(minQ, Qval)
            if minQ >= maxQ:
                maxQ = minQ
                myMove = a

        return myMove


    def chooseAction(self, gameState):
        """
        Choose action.
        """
        #Update filter
        self.enemyFilter.addNewInfo(self.index, gameState)

        self.updateFoodPositions(gameState)
        self.updatePositions(gameState)

        #Features
        X = np.array([])

        food = []
        isPacman = []
        scared = []
        moveHeat = []
        for i in self.myOrderedTeam + self.enemyOrderedTeam:
            agentState = gameState.getAgentState(i)
            food.append(agentState.numCarrying)
            isPacman.append(agentState.isPacman)
            scared.append(agentState.scaredTimer)
            moveHeat.append(self.moveHeatMapValue(i))

        #moves left (1)
        X = np.hstack((X, np.array([300 - self.iter])))

        #Number of food carrying (4)
        X = np.hstack((X, np.array(food)))

        #isPacman (4)
        X = np.hstack((X, np.array(isPacman)))

        #scaredTimer (4)
        X = np.hstack((X, np.array(scared)))

        #uncertainty about enemy positions (2)
        X = np.hstack((X, np.array(self.uncertainty)))

        #move heat-map (4)
        X = np.hstack((X, np.array(moveHeat)))

        #score (1)
        X = np.hstack((X, np.array([self.getScore(gameState)])))

        #number of capsules left (2)
        X = np.hstack((X, np.array([len(self.capsules), len(self.defCapsules)])))

        #number of food left (2)
        X = np.hstack((X, np.array([len(self.food), len(self.defFood)])))

        #distance from starting position (4)
        distFromStart = []
        for i in self.myOrderedTeam + self.enemyOrderedTeam:
            pos = gameState.getInitialAgentPosition(i)
            dist, _ = self.distanceAndDirectionToPos(gameState, i, pos)
            distFromStart.append(dist)
        X = np.hstack((X, np.array(distFromStart)))

        #distance and move along the shortest path to teammate (1 + 5)
        dist, direct = self.distanceAndDirectionToAgent(gameState, self.myOrderedTeam[0],self.myOrderedTeam[1])
        X = np.hstack((X, np.array([dist] + direct)))

        #distance and move along the shortest path from my current agent to both enemies (1 + 5 + 1 + 5)
        dist, direct = self.distanceAndDirectionToAgent(gameState, self.myOrderedTeam[0],self.enemyOrderedTeam[0])
        X = np.hstack((X, np.array([dist] + direct)))
        dist, direct = self.distanceAndDirectionToAgent(gameState, self.myOrderedTeam[0],self.enemyOrderedTeam[1])
        X = np.hstack((X, np.array([dist] + direct)))

        #distance and move along the shortest path to closest enemy food (1 + 5)
        dist, direct = self.closestFoodFromAgent(gameState, self.myOrderedTeam[0])
        X = np.hstack((X, np.array([dist] + direct)))

        #distance and move along the shortest path to closest enemy calsule (1 + 5)
        dist, direct = self.closestCapsuleFromAgent(gameState, self.myOrderedTeam[0])
        X = np.hstack((X, np.array([dist] + direct)))

        #distance and move along the shortest path to the boundary (1 + 5)
        dist, direct = self.closestBoundaryFromAgent(gameState, self.myOrderedTeam[0])
        X = np.hstack((X, np.array([dist] + direct)))

        #NEXT ENEMY
        #distances end directions for the next enemy
        #TODO if exact position is unnkown move is [0 0 0 0 0] ???

        #distance and move along the shortest path to teammate (1 + 5)
        #dist, direct = self.distanceAndDirectionToAgent(gameState, self.enemyOrderedTeam[0],self.enemyOrderedTeam[1])
        #X = np.hstack((X, np.array([dist] + direct)))

        #move along the shortest path from next current enemy agent to my agents (5 + 5)
        _, direct = self.distanceAndDirectionToAgent(gameState, self.enemyOrderedTeam[0],self.myOrderedTeam[0])
        X = np.hstack((X, np.array(direct)))
        _, direct = self.distanceAndDirectionToAgent(gameState, self.enemyOrderedTeam[0],self.myOrderedTeam[1])
        X = np.hstack((X, np.array(direct)))

        #distance and move along the shortest path to closest enemy food (1 + 5)
        dist, direct = self.closestFoodFromAgent(gameState, self.enemyOrderedTeam[0])
        X = np.hstack((X, np.array([dist] + direct)))

        #distance and move along the shortest path to closest enemy calsule (1 + 5)
        dist, direct = self.closestCapsuleFromAgent(gameState, self.enemyOrderedTeam[0])
        X = np.hstack((X, np.array([dist] + direct)))

        #distance and move along the shortest path to the boundary (1 + 5)
        dist, direct = self.closestBoundaryFromAgent(gameState, self.enemyOrderedTeam[0])
        X = np.hstack((X, np.array([dist] + direct)))


        #compute best move
        legalActions = gameState.getLegalActions(self.index)
        try:
            enemyLegalActions = gameState.getLegalActions(self.enemyOrderedTeam[0])
        except:
            enemyLegalActions = None

        if MODE != 0:
            if MODE == 1:
                stateM  = self.stateMatrix(gameState)
                bestPos = np.unravel_index(np.argmax(stateM), stateM.shape)
                bestDir = self.nextShortest(gameState, bestPos)
                n = len(legalActions)
                legalActions.append(bestDir)
                prob = [EXPLORE / n] * n
                prob.append(1 - EXPLORE)
                myMove = np.random.choice(np.array(legalActions), p=prob)
            else:
                bestDir = Agent.maxMinQMove(X, legalActions, enemyLegalActions)

            n = len(legalActions)
            legalActions.append(bestDir)
            prob = [EXPLORE / n] * n
            prob.append(1 - EXPLORE)
            myMove = np.random.choice(np.array(legalActions), p=prob)

            QDB.addRow(X, myMove, self.index)
        else:
            myMove = Agent.maxMinQMove(X, legalActions, enemyLegalActions)

        self.iter += 1
        return myMove


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
            for x in xrange(Agent.w):
                for y in xrange(Agent.h):
                    if self.enemyBeliefState[agentID][x,y]:
                        stateList.append((x,y))
            self.enemyBeliefStateAsList[agentID] = stateList


    def closest(self, gameState, fromPos, toList):
        """
        Return closest distance and direction
        to closest element in toList from fromPos.
        """
        if len(toList) == 0:
            return 0, Agent.enc["Stop"], None
        else:
            distances = [(toPos, self.getMazeDistance(fromPos, toPos)) for toPos in toList]
            toPos, dist = min(distances, key=lambda x: x[1])
            return dist, self.shortestPathMove(gameState, fromPos, toPos), toPos


    def getValidMovesFromPos(self, gameState, pos):
        """
        Return array of valid moves from arbitrary position pos, example ["Stop", "North"]
        """
        validMoves = []

        if not gameState.hasWall(pos[0], pos[1]):
            for k, v in Agent.moves.items():
                x = pos[0] + v[0]
                y = pos[1] + v[1]
                inBounds = x > -1 and x < Agent.w and y > -1 and y < Agent.h
                if inBounds and not gameState.hasWall(x, y):
                    validMoves.append(k)
        return validMoves


    def shortestPathMove(self, gameState, fromPos, toPos):
        validMoves = self.getValidMovesFromPos(gameState, fromPos)
        dist = 1000
        bestMove = "Stop"
        for m in validMoves:
            x = fromPos[0] + Agent.moves[m][0]
            y = fromPos[1] + Agent.moves[m][1]
            d = self.getMazeDistance((x,y), toPos)
            if d < dist:
                dist = d
                bestMove = m
        return Agent.enc[bestMove]


    def closestFoodFromAgent(self, gameState, fromAgent):
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

        return self.closestDistanceAndMoveForLists(gameState, fromList, toList)


    def closestCapsuleFromAgent(self, gameState, fromAgent):
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

        return self.closestDistanceAndMoveForLists(gameState, fromList, toList)


    def closestBoundaryFromAgent(self, gameState, fromAgent):
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
                toList = Agent.redBoundary
            else:
                toList = Agent.blueBoundary
        else:
            if self.red:
                toList = Agent.blueBoundary
            else:
                toList = Agent.redBoundary

        return self.closestDistanceAndMoveForLists(gameState, fromList, toList)


    def distanceAndDirectionToAgent(self, gameState, fromAgent, toAgent):
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

        return self.closestDistanceAndMoveForLists(gameState, fromList, toList)


    def distanceAndDirectionToPos(self, gameState, fromAgent, toPos):
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

        return self.closestDistanceAndMoveForLists(gameState, fromList, toList)


    def closestDistanceAndMoveForLists(self, gameState, fromList, toList):
        minDist = 1000
        move = Agent.enc["Stop"]
        for fromPos in fromList:
            dist, direc, _ = self.closest(gameState, fromPos, toList)
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
            maxValue = max(maxValue, Agent.moveMap[pos[0], pos[1]])

        return maxValue

