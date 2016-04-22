from game import Actions
from util import manhattanDistance
from copy import deepcopy
import numpy as np


class Filter(object):
    """Class for monitoring and estimating enemy positions"""

    def __init__(self):
        self.info = False
        self.firstIter = True
        self.noise = 6
        self.radius = 5


    #@classmethod
    def addInitialGameStateInfo(self, myID, gameState):
        if not self.info:
            self.info = True
            walls = gameState.getWalls() #Grid

            self.numOfAgents = gameState.getNumAgents()
            self.width = walls.width
            self.height = walls.height
            self.isRedTeam = gameState.isOnRedTeam(myID)
            self.myTeam = gameState.getRedTeamIndices() if self.isRedTeam else gameState.getBlueTeamIndices()
            self.enemyTeam = gameState.getBlueTeamIndices() if self.isRedTeam else gameState.getRedTeamIndices()

            self.food  = gameState.getRedFood() if self.isRedTeam else gameState.getBlueFood()

            self.startPositions = map(lambda x: gameState.getInitialAgentPosition(x), range(self.numOfAgents))
            self.prevPos = [None for i in range(self.numOfAgents)]
            self.currentPos = deepcopy(self.startPositions)

            self.exactEnemyPositions = {i: None for i in self.enemyTeam}
            
            #default all-possible
            self.allPossibleState = np.zeros((self.width, self.height), dtype=bool)

            #transition matrix
            self.A = np.zeros((self.width, self.height, self.width, self.height), dtype=bool)
            for x1 in xrange(0, self.width):
                for y1 in xrange(0, self.height):
                    if not walls[x1][y1]:
                        self.allPossibleState[x1,y1] = True
                        for x2, y2 in Actions.getLegalNeighbors((x1, y1), walls):
                            self.A[x1,y1,x2,y2] = True

            #observations
            self.B = np.zeros((self.width, self.height, self.width, self.height, self.width + self.height + 2*self.noise), dtype=bool)
            for x1 in xrange(0, self.width):
                for y1 in xrange(0, self.height):
                    if not walls[x1][y1]:
                        for x2 in xrange(0, self.width):
                            for y2 in xrange(0, self.height):
                                if not walls[x2][y2]:
                                    d = manhattanDistance((x1, y1), (x2, y2))
                                    for i in xrange(d - self.noise, d + self.noise + 1):
                                        self.B[x1, y1, x2, y2, i + self.noise] = True

            self.currentBeliefState = {}
            for enemy in self.enemyTeam:
                beliefState = np.zeros((self.width, self.height), dtype=bool)
                beliefState[self.startPositions[enemy]] = True
                self.currentBeliefState[enemy] = beliefState


    #@classmethod
    def addNewInfo(self, agentID, gameState):
        dist = gameState.getAgentDistances()
        self.currentPos = [gameState.getAgentPosition(ind) for ind in range(self.numOfAgents)]
        
        missingFoodPosition = self._analizeFood(agentID, gameState)
        deadEnemies = self._analizeDeadAgents(agentID)

        for i in self.enemyTeam:
            exactPos = False
            self.exactEnemyPositions[i] = None

            if i in deadEnemies:
                self._setExactPosition(i, self.startPositions[i])
                exactPos = True

            if i == agentID - 1 or (not self.firstIter and agentID == 0 and i == self.numOfAgents - 1):
                #i-th agent have just made a move
                self._computeNewBeliefState(i)
                exactPos = False

                #check missing food
                if missingFoodPosition is not None:
                    self._setExactPosition(i, missingFoodPosition)
                    exactPos = True

            #check if exact position is available
            if self.currentPos[i] is not None:
                self._setExactPosition(i, self.currentPos[i])
            else:
                if not exactPos:
                    self._updateCurrentStateWithObservation(i, dist[i], self.currentPos[agentID])
                    self._filterByFoodAndDistance(i)

        self.firstIter = False

        self._checkBeliefStateForNonZero()


    #@classmethod
    def getBeliefStateBool(self):
        '''returns dict with enemy ids as keys and np.array of bools as values'''
        return self.currentBeliefState

    #@classmethod
    def getBeliefStateProb(self):
        '''returns dict with enemy ids as keys and np.array of uniform probabilities as values'''
        probBeliefState = {}
        for enemy in self.enemyTeam:
            beliefState = np.zeros((self.width, self.height))
            nPossiblePos = np.sum(self.currentBeliefState[enemy])
            p = 1./nPossiblePos
            for x in range(self.width):
                for y in range(self.height):
                    if self.currentBeliefState[enemy][x,y]:
                        beliefState[x,y] = p

            probBeliefState[enemy] = beliefState

        return probBeliefState


    #@classmethod
    def getExactEnemyPositions(self):
        '''returns dict with ememy agent ID as keys and tuple or None as value'''
        return self.exactEnemyPositions


    #@classmethod
    def _checkBeliefStateForNonZero(self):
        for enemy in self.enemyTeam:
            nPossiblePos = np.sum(self.currentBeliefState[enemy])
            if nPossiblePos == 0:
                print "Warning: something went terribly wrong in filtering, so I assumed all-possible state"
                self.currentBeliefState[enemy] = deepcopy(self.allPossibleState)
                self._filterByFoodAndDistance(enemy)


    #@classmethod
    def _computeNewBeliefState(self, enemyID):
        self.currentBeliefState[enemyID] = np.tensordot(self.currentBeliefState[enemyID], self.A, 2)


    #@classmethod
    def _setExactPosition(self, enemyID, pos):
        self.currentBeliefState[enemyID] = np.zeros((self.width, self.height), dtype=bool)
        self.currentBeliefState[enemyID][pos] = True
        self.exactEnemyPositions[enemyID] = pos


    #@classmethod
    def _updateCurrentStateWithObservation(self, enemyID, obs, myPos):
        self.currentBeliefState[enemyID] = np.multiply(self.currentBeliefState[enemyID], self.B[:, :, myPos[0], myPos[1], obs + self.noise])


    #@classmethod
    def _filterByFoodAndDistance(self, enemyID):
        #if there is my food or distance <= 5 there can not be any enemy agent
        for x in range(self.width):
            for y in range(self.height):
                if self.food[x][y]:
                    self.currentBeliefState[enemyID][x, y] = False
                else:
                    for agent in self.myTeam:
                        if manhattanDistance(self.currentPos[agent], (x,y)) < self.radius + 1:
                            self.currentBeliefState[enemyID][x, y] = False
                            break


    #@classmethod
    def _analizeFood(self, agentID, gameState):
        newFood  = gameState.getRedFood() if self.isRedTeam else gameState.getBlueFood()
        pos = None
        if self.food is not None:
            #check differnce
            for x in range(self.width):
                for y in range(self.height):
                    if self.food[x][y] and not newFood[x][y]:
                        pos = (x, y)
                        break
                if pos is not None:
                    break

        self.food = deepcopy(newFood)
        return pos


    #@classmethod
    def _analizeDeadAgents(self, agentID):
        '''returns list of ids of dead enemy agents (since last check)'''
        prevEnemy = self.numOfAgents - 1 if agentID == 0 else agentID - 1
        prevTeammate = self.numOfAgents - 1 if prevEnemy == 0 else prevEnemy - 1
        
        deadEnemies = []
        print agentID, prevEnemy, prevTeammate

        #check if previous teammate ate an enemy
        for enemy in self.enemyTeam:
            if self.prevPos[enemy] is not None and self.prevPos[enemy] == self.currentPos[prevTeammate]:
                deadEnemies.append(enemy)

        #check if previous enemy run into my goust and was eaten
        if prevEnemy not in deadEnemies:
            if self.currentPos[prevEnemy] is None and self.prevPos[prevEnemy] is not None:
                for agent in self.myTeam:
                    print agent
                    #check if my agent died
                    myAgentDied = manhattanDistance(self.prevPos[agent], self.currentPos[agent]) > (0 if agent==agentID else 1)
                    if not myAgentDied and manhattanDistance(self.prevPos[prevEnemy], self.prevPos[agent]) <= (1 if agent==agentID else 2):
                        deadEnemies.append(prevEnemy)
                        break

        self.prevPos = deepcopy(self.currentPos)

        return deadEnemies

