import os
from Qreward import saveScore, giveRewards, updateQValues

command = "python capture.py -r Qagent -b Qagent -Q"
nOfGames = 5

# this shoud be done in cycle, after each iteration train neural net
for i in range(nOfGames):
	os.system(command)
	saveScore()

giveRewards()
updateQValues()

#train neural net
