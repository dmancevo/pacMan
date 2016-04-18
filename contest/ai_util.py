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
