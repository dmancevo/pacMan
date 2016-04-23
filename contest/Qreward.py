import numpy as np
import sqlite3
import pickle as pkl

def Q(X, a, o):
    #TODO replace with neural net 
    return np.random.randn() * 0.1

def maxMinQ(X, myActions = None, enemyActions = None):
    enc = {
        "Stop" : [0,0,0,0,1],
        "North": [0,0,0,1,0],
        "South": [0,0,1,0,0],
        "East" : [0,1,0,0,0],
        "West" : [1,0,0,0,0],
        }

    if myActions is None:
        myActions = enc.keys()
    if enemyActions is None:
        enemyActions = enc.keys()

    maxQ = -1000
    for a in myActions:
        minQ = 1000
        for o in enemyActions:
            Qval = Q(X, a, o)
            minQ = min(minQ, Qval)
        maxQ = max(maxQ, minQ)

    return maxQ

def giveRewards(recomputeAll = False):
    conn = sqlite3.connect("Qbase.db")
    cur  = conn.cursor()
    if recomputeAll:
        sql = "select rowid, gameID, agentID, X, a, o from Qtable"
    else:
        sql = "select rowid, gameID, agentID, X, a, o from Qtable where rw='None'"
    data = cur.execute(sql) #iterator
    row1 = None
    row2 = None
    id1 = None
    id2 = None
    X1 = None
    X2 = None
    endgame = {}
    rewards = []
    game = 0
    for row in data:
        agentID = row[2]
        currGame = row[1]
        if currGame != game:
            if game != 0:
                #save game and ids for endgame scoring
                endgame[game] = [id1, id2, row1, row2]
                
            row1 = None
            row2 = None
            id1 = None
            id2 = None
            X1 = None
            X2 = None
            game = currGame

        X3 = pkl.loads(row[3])
        if row1 is not None:
            reward  = calculateReward(X1, X2, X3)
            rewards.append([reward, row1])
        row1 = row2
        row2 = row[0]
        id1 = id2
        id2 = agentID
        X1 = np.copy(X2)
        X2 = X3

    if game != 0:
        endgame[game] = [id1, id2, row1, row2]
    #add endgame scores
    if len(endgame.keys()) > 0:
        sql = "select gameID, score from score where gameID in (" + (",".join(map(lambda x: str(x),endgame.keys()))) + ")"
        cur.execute(sql)
        for row in cur.fetchall():
            agent1 = endgame[row[0]][0]
            agent2 = endgame[row[0]][1]
            rowid1 = endgame[row[0]][2]
            rowid2 = endgame[row[0]][3]
            score = 100 if row[1] > 0 else -100 if row[1] < 0 else 0
            
            reward1 = score if (agent1 == 0 or agent1 == 2) else -score
            reward2 = score if (agent2 == 0 or agent2 == 2) else -score
            rewards.append([reward1, rowid1])
            rewards.append([reward2, rowid2])

    cur.executemany("update Qtable set rw=? where rowid=?", rewards)
    conn.commit()
    conn.close()


def calculateReward(X1, X2, X3):
    '''
    X1 state before my move
    X2 state after my move - from opponent's perspective
    X3 stare after opponent's move - from teammate's perspective
    '''
    foodICarriedBefore = X1[1]
    foodICarryAfter = X3[2]

    foodTeammateCarriedBefore = X1[2]
    foodTeammateCarryAfter = X3[1]

    foodEnemyCarriedBefore = X1[3]
    foodEnemyCarriedAfter = X3[4]

    foodEnemy2CarriedBefore = X1[4]
    foodEnemy2CarriedAfter = X3[3]
    #
    scoreBeforeMyMove = X1[19]
    scoreAfterMyMove = - X2[19]
    scoreAfterEnemyMove = X3[19]
    #
    iScored = (scoreAfterMyMove - scoreBeforeMyMove) # > 0
    enemyScored = (scoreAfterEnemyMove - scoreAfterMyMove) # < 0
    #
    powerFoodBefore = X1[20] #before my move
    powerFoodAfter = X2[21] #after my move
    defencePowerFoodBefore = X2[20] #before opponent's move
    defencePowerFoodAfter = X3[21] #after opponent's move
    #
    iAtePowerFood = powerFoodAfter < powerFoodBefore
    enemyAtePowerFood = defencePowerFoodAfter < defencePowerFoodBefore

    #coefficients
    c_score = 2.
    c_powerFood = 1.
    c_food = 1.

    reward = 0
    reward += (iScored + enemyScored) * c_score
    reward += (int(iAtePowerFood) - int(enemyAtePowerFood)) * c_powerFood

    if not iScored:
        reward += (foodICarryAfter - foodICarriedBefore) * c_food

    if not enemyScored:
        reward += (foodEnemyCarriedAfter - foodEnemyCarriedBefore) * (-c_food)

    reward += (foodTeammateCarryAfter - foodTeammateCarriedBefore) * c_food
    reward += (foodEnemy2CarriedAfter - foodEnemy2CarriedBefore) * (-c_food)

    return reward


def updateQValues(recomputeAll = True, gamma = 0.5):
    conn = sqlite3.connect("Qbase.db")
    cur  = conn.cursor()
    if recomputeAll:
        sql = "select rowid, gameID, agentID, X, a, o, rw from Qtable"
    else:
        sql = "select rowid, gameID, agentID, X, a, o, rw from Qtable where Qval='None'"
    data = cur.execute(sql) #iterator
    row1 = None
    row2 = None
    rw1 = None
    rw2 = None
    X1 = None
    X2 = None
    Qvals = []
    endgame = []
    game = 0
    for row in data:
        agentID = row[2]
        currGame = row[1]
        reward  = row[6]
        if currGame != game:
            if game != 0:
                #endgame
                Qvals.append([rw1, row1])
                Qvals.append([rw2, row2])
                pass
                
            row1 = None
            row2 = None
            rw1 = None
            rw2 = None
            X1 = None
            X2 = None
            game = currGame

        X3 = pkl.loads(row[3])
        if row1 is not None:
            Q = rw1 + gamma * maxMinQ(X3)
            Qvals.append([Q, row1])
        row1 = row2
        row2 = row[0]
        rw1 = rw2
        rw2 = reward
        X1 = np.copy(X2)
        X2 = X3

    if game != 0:
        Qvals.append([rw1, row1])
        Qvals.append([rw2, row2])
        cur.executemany("update Qtable set Qval=? where rowid=?", Qvals)
    conn.commit()
    conn.close()


def saveScore():
    #read score of the last game
    f = open('score', 'r')
    score = int(f.read())
    
    conn = sqlite3.connect("Qbase.db")
    cur  = conn.cursor()
    try:
        cur.execute("create table score (gameID int, score int)")
        gameID = 1
    except:
        cur.execute("select gameID from score order by rowid desc limit 1")
        gameID = cur.fetchone()[0] + 1

    cur.execute("insert into score values (?,?)", (gameID, score))

    conn.commit()
    conn.close()


if __name__ == '__main__':
    giveRewards()
    updateQValues()