import sqlite3
import pickle as pkl

class QDB(object):
    """docstring for ClassName"""
    firstMove = True

    @classmethod
    def addRow(cls, X, a, agentID):
        """
        Add sample row to database, features X, reward rw
        and Q(s,a,o) Qval)
        """

        conn = sqlite3.connect("Qbase.db")
        cur  = conn.cursor()

        if cls.firstMove:
            try:
                cur.execute("create table Qtable (gameID int, agentID int, X text, a text, o text, rw real, Qval real, target real)")
                cls.gameID = 1
            except:
                cur.execute("select gameID from Qtable order by rowid desc limit 1")
                cls.gameID = cur.fetchone()[0] + 1
        else:
            cur.execute("update Qtable set o=? where rowid=?", (a, cls.last_id))
        
        X_pkl = pkl.dumps(X)
        cur.execute("insert into Qtable values (?,?,?,?,?,?,?,?)", (cls.gameID, agentID, X_pkl, a, "Stop", "None", "None", "None"))
        cls.last_id = cur.lastrowid
        conn.commit()
        conn.close()

        cls.firstMove = False
