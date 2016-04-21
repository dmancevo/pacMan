from baseAgent2 import baseAgent
from ai_filter import Filter
import numpy as np
import sqlite3
import pickle as pkl

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Agent', second = 'Agent'):

  return [eval(first)(firstIndex), eval(second)(secondIndex)]


class Agent(baseAgent):
  
  #Boundary
  boundary = None
  
  #Direction encoding
  enc = {
    "Stop" : [0,0,0,0,1],
    "North": [0,0,0,1,0],
    "South": [0,0,1,0,0],
    "East" : [0,1,0,0,0],
    "West" : [1,0,0,0,0],
    }
  
  #Neural Net
  
  @staticmethod
  def addRow(X, a):
    """
    Add sample row to database, features X, reward rw
    and Q(s,a,o) Qval)
    """
    conn = sqlite.connect("Qbase.db")
    cur  = conn.cursor()
    try:
      
      #Update latest row
      cur.execute("update Qtable set o=?",(a,))
      
      #X
      X_pkl = pkl.dumps(X)
      cur.execute("insert into Qtable values (?,?,?,?,?,?)",(X_pkl,a,"None","None","None"))
      
    except sqlite3.OperationalError:
      cur.execute("create table Qtable X text, a text, o text, rw real, Qval real, target real")
      Agent.addRow(X,a)
    finally:
      conn.commit()
      conn.close()
  
  def registerInitialState(self, gameState):
    baseAgent.registerInitialState(self, gameState)
    Filter.addInitialGameStateInfo(self.index, gameState)
    
    #Map boundary
    if not Agent.boundary:
      walls = gameState.getWalls()
      Agent.boundary = [(walls.width/2,i) for i in range(walls.height)]
  
  def chooseAction(self, gameState):
    """
    Choose action.
    """
  
    #Update filter
    Filter.addNewInfo(self.index, gameState)
    
    #Features
    X = np.array([])
    
    #Closest food, boundary and capsules.
    for f in [self.closestFood, self.closestDefenseFood,
    self.closestBoundary, self.closestCapsule,
    self.closestDefenseCapsule]:
      
      dist, direc = f(gameState)
      X = np.hstack((X, np.array(Agent.enc[direc])*dist))
      
    #Closest Enemy.
    ind, dist, direc = self.closestEnemy(gameState)
    st = gameState.getAgentState(self.index)
    opp_st = gameState.getAgentState(ind)
    
    if opp_st.isPacman and not st.scaredTimer:
      X = np.hstack((X, np.array(Agent.enc[direc])*dist))
    else:
      X = np.hstack((X, np.array(Agent.enc[direc])/dist))
      
    #Number of food carrying.
    X = np.hstack((X,np.array([st.numCarrying])))
    
    print X
      
    return "Stop"
    
    
  def closest(self, gameState, M):
    """
    Return closest distance and direction
    to closest element in M from current position.
    """
    if not M: return 0, "Stop"
    myPos = gameState.getAgentState(self.index).getPosition()
    distances  = [(pos,self.getDist(myPos, pos)) for pos in M]
    pos, dist = min(distances,key=lambda x: x[1])
    return dist, self.nextShortest(gameState, pos)
  
  def closestFood(self, gameState):
    """
    Return position of closest food.
    """
    food = self.getFood(gameState).asList()
    return self.closest(gameState, food)
    
  def closestDefenseFood(self, gameState):
    """
    Return position of closest food
    we are defending.
    """
    food = self.getFoodYouAreDefending(gameState).asList()
    return self.closest(gameState, food)
  
  def closestBoundary(self, gameState):
    """
    Return position of closest boundary point.
    """
    return self.closest(gameState, Agent.boundary)
  
  def closestCapsule(self, gameState):
    """
    Return position of closest capsule.
    """
    capsules = self.getCapsules(gameState)
    return self.closest(gameState, capsules)
  
  def closestDefenseCapsule(self, gameState):
    """
    Return position of closest capsule
    we are defending.
    """
    capsules = self.getCapsulesYouAreDefending(gameState)
    return self.closest(gameState, capsules)
  
  def closestEnemy(self, gameState):
    """
    Return position and index of closest enemy.
    """
    all_pos = Filter.getBeliefStateBool()
    dist = float("inf")
    for key in all_pos.keys():
      opp_pos = []
      width, height = all_pos[key].shape
      for i in range(width):
        for j in range(height):
          if all_pos[key][i,j]:
            opp_pos.append((i,j))
            
      new_ind = key
      new_dist, new_direc = self.closest(gameState, opp_pos)
      if new_dist < dist:
        ind, dist, direc = new_ind, new_dist, new_direc
        
    return ind, dist, direc
      