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
from ai_filter import Filter
import sqlite3
import pickle as pkl

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
    
  @staticmethod
  def saveAction(X, a, firstMove):
    """
    Add sample row to database, features X, reward rw
    and Q(s,a,o) Qval)
    """
    conn = sqlite3.connect("../Qbase.db")
    cur  = conn.cursor()
    try:
      
      if not firstMove:
        #Update latest row
        cur.execute("update Qtable set o=? order by id desc limit 1",(a,))
      
      #X
      X_pkl = pkl.dumps(X)
      cur.execute("insert into Qtable (X,a) values (?,?)",(X_pkl,a))
      
    except sqlite3.OperationalError:
      cur.execute("create table Qtable (id integer primary key, X text, a text, o text, rw real, Qval real, target real)")
      Agent.saveAction(X,a,firstMove)
    finally:
      conn.commit()
      conn.close()
      
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    # Filter.addInitialGameStateInfo(self.index, gameState)

  def chooseAction(self, gameState):
    """
    Choose action.
    """

    # Filter.addNewInfo(self.index, gameState)
    '''
    To get beliefState as dict of np.arrays use:
    Filter.getBeliefStateProb() - uniform probability over possible states
    Filter.getBeliefStateBool() - bool (True if can be in particular state, False otherwise)
    '''
    
    actions = gameState.getLegalActions(self.index)

    action = np.random.choice(actions)
    
    #Store action in database
    if self.index == 0 and not self.getPreviousObservation():
      firstMove = True
    else:
      firstMove = False
    Agent.saveAction(["random move"],action,firstMove)
    
    return action

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