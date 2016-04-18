from baseAgent import baseAgent
import numpy as np

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Agent', second = 'Agent'):

  return [eval(first)(firstIndex), eval(second)(secondIndex)]


class Agent(baseAgent2):

    #Neural Net
    b1, b2 = np.random.normal(0,0.1,2)

    m, n = 10, 4
    W = np.random.normal(0,0.1,m*n).reshape((m,n))

    w = np.random.normal(0,0.1,n)

    phi = lambda x: 1.0/(1+np.exp(-x))

    @classmethod
    def forward(cls, x):
        """
        Forward pass.
        """
        w, phi, W, b1, b2 = cls.w, cls.phi, cls.W, cls.b1, cls.b2
        cls.x, cls.phiWx = x, phi(W.dot(x))
        cls.fwd = w.dot(cls.phiWx+b1)+b2
        return cls.fwd

    @classmethod
    def backward(cls, y, delta=0.01):
        """
        Backward pass and update.
        """
        fwd, y, x, phi = cls.fwd, cls.y, cls.x, cls.phi

        #Update third layer
        err = fwd-y
        cls.b2 -= err
        cls.w -= err*cls.phiWx

        #Update middle layer
        dfwd_dphi = phi*(1-phi)
        cls.b1 -= err*dfwd_dphi
        dWx_dW = np.outer(x,x)
        cls.W -= err*dfwd_dphi*dWx_dx


    def chooseAction(self, gameState):
        """
        Choose action.
        """

        Filter.addNewInfo(self.index, gameState)
        stateM  = self.stateMatrix(gameState)
        bestPos = np.unravel_index(np.argmax(stateM), stateM.shape)
        bestDir = self.nextShortest(gameState, bestPos)
    
        return bestDir

