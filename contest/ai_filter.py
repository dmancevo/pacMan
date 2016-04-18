from game import Actions
from util import manhattanDistance
from copy import deepcopy
import numpy as np


class Filter(object):
    """Class for monitoring and estimating enemy positions"""
    
    info = False
    firstIter = True
    noise = 6
    radius = 5

    @classmethod
    def addInitialGameStateInfo(cls, myID, gameState):
        if not cls.info:
            cls.info = True
            walls = gameState.getWalls() #Grid

            cls.numOfAgents = gameState.getNumAgents()
            cls.width = walls.width
            cls.height = walls.height
            cls.isRedTeam = gameState.isOnRedTeam(myID)
            cls.myTeam = gameState.getRedTeamIndices() if cls.isRedTeam else gameState.getBlueTeamIndices()
            cls.enemyTeam = gameState.getBlueTeamIndices() if cls.isRedTeam else gameState.getRedTeamIndices()

            cls.food  = gameState.getRedFood() if cls.isRedTeam else gameState.getBlueFood()

            cls.startPositions = map(lambda x: gameState.getInitialAgentPosition(x), range(cls.numOfAgents))
            cls.prevPos = [None for i in range(cls.numOfAgents)]
            cls.currentPos = deepcopy(cls.startPositions)

            cls.exactEnemyPositions = {i: None for i in cls.enemyTeam}

            #transition matrix
            cls.A = np.zeros((cls.width, cls.height, cls.width, cls.height), dtype=bool)
            for x1 in xrange(0, cls.width):
                for y1 in xrange(0, cls.height):
                    if not walls[x1][y1]:
                        for x2, y2 in Actions.getLegalNeighbors((x1, y1), walls):
                            cls.A[x1,y1,x2,y2] = True

            #observations
            cls.B = np.zeros((cls.width, cls.height, cls.width, cls.height, cls.width + cls.height + 2*cls.noise), dtype=bool)
            for x1 in xrange(0, cls.width):
                for y1 in xrange(0, cls.height):
                    if not walls[x1][y1]:
                        for x2 in xrange(0, cls.width):
                            for y2 in xrange(0, cls.height):
                                if not walls[x2][y2]:
                                    d = manhattanDistance((x1, y1), (x2, y2))
                                    for i in xrange(d - cls.noise, d + cls.noise + 1):
                                        cls.B[x1, y1, x2, y2, i + cls.noise] = True

            cls.currentBeliefState = {}
            for enemy in cls.enemyTeam:
                beliefState = np.zeros((cls.width, cls.height), dtype=bool)
                beliefState[cls.startPositions[enemy]] = True
                cls.currentBeliefState[enemy] = beliefState

    @classmethod
    def addNewInfo(cls, agentID, gameState):
        dist = gameState.getAgentDistances()
        cls.currentPos = [gameState.getAgentPosition(ind) for ind in range(cls.numOfAgents)]
        
        missingFoodPosition = cls._analizeFood(agentID, gameState)
        deadEnemies = cls._analizeDeadAgents(agentID)

        for i in cls.enemyTeam:
            exactPos = False
            cls.exactEnemyPositions[i] = None

            if i in deadEnemies:
                cls._setExactPosition(i, cls.startPositions[i])
                exactPos = True

            if i == agentID - 1 or (not cls.firstIter and agentID == 0 and i == cls.numOfAgents - 1):
                #i-th agent have just made a move
                cls._computeNewBeliefState(i)
                exactPos = False

                #check missing food
                if missingFoodPosition is not None:
                    cls._setExactPosition(i, missingFoodPosition)
                    exactPos = True

            #check if exact position is available
            if cls.currentPos[i] is not None:
                cls._setExactPosition(i, cls.currentPos[i])
            else:
                if not exactPos:
                    cls._updateCurrentStateWithObservation(i, dist[i], cls.currentPos[agentID])
                    cls._filterByFoodAndDistance(i)

        cls.firstIter = False

    @classmethod
    def getBeliefStateBool(cls):
        '''returns dict with enemy ids as keys and np.array of bools as values'''
        return cls.currentBeliefState

    @classmethod
    def getBeliefStateProb(cls):
        '''returns dict with enemy ids as keys and np.array of uniform probabilities as values'''
        probBeliefState = {}
        for enemy in cls.enemyTeam:
            beliefState = np.zeros((cls.width, cls.height))
            nPossiblePos = np.sum(cls.currentBeliefState[enemy])
            p = 1./nPossiblePos
            for x in range(cls.width):
                for y in range(cls.height):
                    if cls.currentBeliefState[enemy][x,y]:
                        beliefState[x,y] = p

            probBeliefState[enemy] = beliefState

        return probBeliefState


    @classmethod
    def getExactEnemyPositions(cls):
        '''returns dict with ememy agent ID as keys and tuple or None as value'''
        return cls.exactEnemyPositions


    @classmethod
    def _computeNewBeliefState(cls, enemyID):
        cls.currentBeliefState[enemyID] = np.tensordot(cls.currentBeliefState[enemyID], cls.A, 2)


    @classmethod
    def _setExactPosition(cls, enemyID, pos):
        cls.currentBeliefState[enemyID] = np.zeros((cls.width, cls.height), dtype=bool)
        cls.currentBeliefState[enemyID][pos] = True
        cls.exactEnemyPositions[enemyID] = pos


    @classmethod
    def _updateCurrentStateWithObservation(cls, enemyID, obs, myPos):
        cls.currentBeliefState[enemyID] = np.multiply(cls.currentBeliefState[enemyID], cls.B[:, :, myPos[0], myPos[1], obs + cls.noise])


    @classmethod
    def _filterByFoodAndDistance(cls, enemyID):
        #if there is my food or distance <= 5 there can not be any enemy agent
        for x in range(cls.width):
            for y in range(cls.height):
                if cls.food[x][y]:
                    cls.currentBeliefState[enemyID][x, y] = False
                else:
                    for agent in cls.myTeam:
                        if manhattanDistance(cls.currentPos[agent], (x,y)) < cls.radius + 1:
                            cls.currentBeliefState[enemyID][x, y] = False
                            break


    @classmethod
    def _analizeFood(cls, agentID, gameState):
        newFood  = gameState.getRedFood() if cls.isRedTeam else gameState.getBlueFood()
        pos = None
        if cls.food is not None:
            #check differnce
            for x in range(cls.width):
                for y in range(cls.height):
                    if cls.food[x][y] and not newFood[x][y]:
                        pos = (x, y)
                        break
                if pos is not None:
                    break

        cls.food = deepcopy(newFood)
        return pos


    @classmethod
    def _analizeDeadAgents(cls, agentID):
        '''returns list of ids of dead enemy agents (since last check)'''
        prevEnemy = cls.numOfAgents - 1 if agentID == 0 else agentID - 1
        prevTeammate = cls.numOfAgents - 1 if prevEnemy == 0 else prevEnemy - 1
        
        deadEnemies = []

        for enemy in cls.enemyTeam:
            if cls.prevPos[enemy] is not None and cls.prevPos[enemy] == cls.currentPos[prevTeammate]:
                deadEnemies.append(enemy)

        if prevEnemy not in deadEnemies:
            if cls.currentPos[prevEnemy] is None and cls.prevPos[prevEnemy] is not None:
                for agent in cls.myTeam:
                    if manhattanDistance(cls.prevPos[prevEnemy], cls.prevPos[agent]) <= 2:
                        deadEnemies.append(prevEnemy)
                        break

        cls.prevPos = deepcopy(cls.currentPos)

        return deadEnemies

