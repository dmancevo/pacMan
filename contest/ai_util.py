import numpy as np


def computeClosestPosition(distancer, pos, beliefState):
    """
    distancer: distancer object
    pos: tuple  - target position
    beliefState: 2D np.array of bool or prob, or tuple if exact position is known

    Returns:
    ((x,y), distance)
    """
    closestPosition = None
    minDist = None

    if beliefState is tuple:
        closestPosition = beliefState
        minDist = distancer.getDistance(pos, closestPosition)
    else:
        w, h = beliefState.shape
        for x in range(w):
            for y in range(h):
                if beliefState[x,y]:
                    dist = distancer.getDistance(pos, (x,y))
                    if minDist is None or (dist < minDist):
                       closestPosition = (x,y)
                       minDist = dist
    return closestPosition, minDist

def computeVectorStateRepresentation(agent1, gameState, distancer, moveMap, foodMap, defendFoodMap, beliefState, foodPerTeam):
    #distancer.getDistance(pos1, pos2)
    f = []

    #agents IDs in order of further moves
    enemy1 = 0 if agent1 == 3 else agent1 + 1
    agent2 = 0 if enemy1 == 3 else enemy1 + 1
    enemy2 = 0 if agent2 == 3 else agent2 + 1
    myTeam = [agent1, agent2]
    enemyTeam = [enemy1, enemy2]
    isRed = (agent1 == 0 or agent2 == 0)

    #positions
    #myPos = [gameState.getAgentPosition(ind) for ind in myTeam]
    #enemyPos = [gameState.getAgentPosition(ind) for ind in enemyTeam]
    pos = [gameState.getAgentPosition(i) for i in myTeam + enemyTeam]

    #AgentStates
    #myAgentStates = [gameState.getAgentState(i) for i in myTeam]
    #enemyAgentStates = [gameState.getAgentState(i) for i in enemyTeam]
    agentStates = [gameState.getAgentState(i) for i in myTeam + enemyTeam]

    #FEATURES
    #move heat-map
    #TODO normilize
    f.append(moveMap[pos[0]])
    
    #number of carried food by each agent (normalized by maxPossible)
    #TODO normilize
    for state in agentStates:
        f.append(state.numCarrying/float(foodPerTeam))

    #isPacman for each agent
    for state in agentStates:
        f.append(state.isPacman)

    #scared timer (normalized)
    for state in agentStates:
        f.append(state.scaredTimer/40.)
    
    #score
    if isRed:
        f.append(gameState.getScore()/(foodPerTeam - 2.))
    else:
        f.append(-1 * gameState.getScore()/(foodPerTeam - 2.))



    return f