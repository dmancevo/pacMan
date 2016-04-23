import os
import sqlite3
import pickle as pkl
import numpy as np
from sknn.mlp import Regressor, Layer
from sknn import ae, mlp
from sklearn.metrics import mean_squared_error as mse
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def play(n):
  """
  Play n more games.
  """
  for _ in range(n):
    os.system("capture.py -r Qagent2 -b Qagent2 -b -Q")
  
def dbQ(Q):
  """
  Update Q column in DB
  """
  conn = sqlite3.connect("../Qbase.db")
  cur  = conn.cursor()
  cur.execute("select id from Qtable order by id desc")
  N = cur.fetchone()[0]
  
  for _id in range(1,N+1):
    cur.execute("select X, a, o from Qtable where id=?",(_id,))
    row = cur.fetchone()
    X, a, o = pkl.loads(row[0]), row[1], row[2]
    Qval = Q.predict(list(X)+[a,o])
    cur.execute("update Qtable set Qval=? where id=?",(Qval,_id))
  
  conn.commit()
  conn.close()
  
def dbRw():
  """
  Update reward column.
  """
  pass

def dbTarget(gamma):
  """
  Update target in DB
  """
  conn = sqlite3.connect("../Qbase.db")
  cur  = conn.cursor()
  cur.execute("update Qtable set target = rw+?*Qval", (gamma,))
  conn.commit()
  conn.close()
  
def fromDB():
  """
  Return dataset from database
  """
  conn = sqlite3.connect("../Qbase.db")
  cur  = conn.cursor()
  cur.execute("select * from Qtable")

  X, A, O, Rws, Qvals, targets = [],[],[],[],[],[]
  for row in cur:
    X.append(pkl.loads(row[1]))
    A.append(row[2])
    O.append(row[3])
    Rws.append(row[4])
    Qvals.append(row[5])
    targets.append(row[6])
    
  conn.close()
  
  return X, A, O, Rws, Qvals, targets

# Q = Regressor(
#     layers=[
#         Layer("Rectifier", units=2),
#         Layer("Linear")],
#     learning_rate=0.02,
#     n_iter=5)
    
# X = np.hstack((np.random.uniform(-3,3,300).reshape((300,1)),np.random.uniform(-1,1,300).reshape(300,1)))
# y = np.array([0.3*x[0]+0.4*x[0]**2+2.3*x[1]+np.random.normal(0,0.3) for x in X])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
# Q.fit(X_train, y_train)

# y_pred = Q.predict(X_test)

# print mse(y_test, y_pred)