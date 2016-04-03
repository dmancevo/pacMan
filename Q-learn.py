import numpy as np
import matplotlib.pyplot as plt

#Use a factorization machine!!

class Agent:

	def __init__(self, gamma, delta, epsilon, k):
		'''
		Docs
		'''
		#Discount factor
		self.gamma = gamma

		#Learning rate
		self.delta = delta

		#Epsilon greedy
		self.epsilon = epsilon

		#Q function
		self.W2 = {}
		self.k = k

	def Q(self,state,action):
		'''
		Docs
		'''

		skey = ('state',state)
		if skey not in self.W2:
			self.W2[skey] = np.random.normal(0,0.01,self.k)

		akey = ('action',action)
		if akey not in self.W2:
			self.W2[akey] = np.random.normal(0,0.01,self.k)

		return self.W2[skey].dot(self.W2[akey])

	def Qmax(self,state,actions):
		'''
		Docs
		'''
		vals = [self.Q(state,action) for action in actions]
		ind = np.argmax(vals)
		return actions[ind], vals[ind]

	def choice(self,state,actions):
		'''
		Docs
		'''
		if np.random.uniform(0,1)<self.epsilon:
			action = np.random.choice(actions)
			self.Qval = self.Q(state,action)
		else:
			action, self.Qval = self.Qmax(state,actions)
		self.state, self.action = state, action
		return action

	def reward(self,new_state,actions,rw):
		'''
		Docs
		'''

		e = self.delta * (rw +\
			self.gamma * self.Qmax(new_state, actions)[1] - self.Qval)

		skey = ('state',self.state)
		akey = ('action',self.action)

		self.W2[skey]  += e * self.W2[akey]
		self.W2[akey] += e * self.W2[skey]

		snorm = np.linalg.norm(self.W2[skey])
		if snorm > 1:
			self.W2[skey] = (self.W2[skey]/snorm)

		anorm = np.linalg.norm(self.W2[akey])
		if anorm > 1:
			self.W2[akey] = (self.W2[akey]/anorm)

	def setEpsilon(self,epsilon):
		'''
		Docs
		'''
		self.epsilon = epsilon

def test1():

	agent = Agent(gamma=0.95, delta=1.0, epsilon=1.0, k=1)

	actions = ['left','right']

	for _ in range(10):
		pos = 0
		for __ in range(10):
			action = agent.choice(1, actions)
			if action == 'left':
				pos -= 1
			elif action == 'right':
				pos += 1
			agent.reward(1,actions,pos)

	agent.setEpsilon(0)

	pos = 0
	for _ in range(10):
		action = agent.choice(1, actions)
		if action == 'left':
			pos -= 1
		elif action == 'right':
			pos += 1

		print action, pos

	print agent.W2

def test2():
	agent = Agent(gamma=0.9, delta=1.0, epsilon=0.5, k=2)

	goals = [43]

	for _ in range(100):
		grid = np.zeros(8*8).reshape((8,8))
		pos = (0,0)
		grid[pos] = 1

		goal = np.random.choice(goals)

		while grid.flatten()[goal] != 1:

			actions = ['up','down','right','left']
			if pos[0] == 0:
				actions.remove('up')
			elif pos[0] == 7:
				actions.remove('down')
			
			if pos[1] == 0:
				actions.remove('left')
			elif pos[1] == 7:
				actions.remove('right')

			action = agent.choice(pos,actions)

			grid[pos] = 0

			if action == 'up':
				pos = (pos[0]-1, pos[1])
			elif action == 'down':
				pos = (pos[0]+1, pos[1])
			elif action == 'right':
				pos = (pos[0], pos[1]+1)
			elif action == 'left':
				pos = (pos[0], pos[1]-1)

			grid[pos] = 1

			agent.reward(pos,actions,0)

		actions = ['up','down','right','left']
		if pos[0] == 0:
			actions.remove('up')
		elif pos[0] == 7:
			actions.remove('down')
		
		if pos[1] == 0:
			actions.remove('left')
		elif pos[1] == 7:
			actions.remove('right')

		agent.reward(pos,actions,1)

	gval = [np.zeros(8*8).reshape((8,8)) for i in range(4)]
	for i in range(8):
		for j in range(8):
			gval[0][i][j] = agent.Q((i,j),'up')
			gval[1][i][j] = agent.Q((i,j),'down')
			gval[2][i][j] = agent.Q((i,j),'right')
			gval[3][i][j] = agent.Q((i,j),'left')

	plt.matshow(np.hstack(tuple(gval)))
	plt.show()

	import time

	agent.setEpsilon(0)

	for goal in goals:

		grid = np.zeros(8*8).reshape((8,8))
		pos = (0,0)
		grid[pos] = 1

		plt.ion()
		plt.matshow(grid,fignum=1,cmap=plt.cm.gray)
		plt.show()

		while grid.flatten()[goal] != 1:

			actions = ['up','down','right','left']
			if pos[0] == 0:
				actions.remove('up')
			elif pos[0] == 7:
				actions.remove('down')
			
			if pos[1] == 0:
				actions.remove('left')
			elif pos[1] == 7:
				actions.remove('right')

			action = agent.choice(pos,actions)

			grid[pos] = 0

			if action == 'up':
				pos = (pos[0]-1, pos[1])
			elif action == 'down':
				pos = (pos[0]+1, pos[1])
			elif action == 'right':
				pos = (pos[0], pos[1]+1)
			elif action == 'left':
				pos = (pos[0], pos[1]-1)

			grid[pos] = 1

			plt.matshow(grid,fignum=1,cmap=plt.cm.gray)
			plt.draw()

			time.sleep(0.1)


if __name__ == '__main__':

	test2()