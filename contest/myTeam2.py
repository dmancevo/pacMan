# myTeam2.py
'''
TEST team for filtering
'''


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from ai_filter import Filter

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
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
  return [DummyAgent(firstIndex, enemyFilter), DummyAgent(secondIndex, enemyFilter)]
  # The following line is an example only; feel free to change it.
  #return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def __init__(self, index, enemyFilter):
    CaptureAgent.__init__(self, index)
    self.enemyFilter = enemyFilter

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
    self.enemyFilter.addInitialGameStateInfo(self.index, gameState)


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    self.enemyFilter.addNewInfo(self.index, gameState)
    enemyPositions = self.enemyFilter.getBeliefStateProb()
    
    cells = []
    self.debugDraw(cells, [1, 1, 1], clear=True)
    for enemy, data in enemyPositions.items():
      if enemy <2:
        for i in range(data.shape[0]):
          for j in range(data.shape[1]):
            if data[i,j]>0:
              self.debugDraw((i,j), [data[i,j], 0.1, 0], clear=False)

    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)