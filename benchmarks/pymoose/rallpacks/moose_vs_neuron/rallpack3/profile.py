"""config.py: 

    Global variables.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sqlite3 as sql 

dbFile = '_profile.sqlite'
conn_ = sql.connect(dbFile)
cur_ = conn_.cursor()
tableName = 'rallpack1'

cur_.execute(
        """CREATE TABLE IF NOT EXISTS {} ( time DATETIME 
        , no_of_compartment INTEGER 
        , simulator TEXT NOT NULL
        , runtime REAL DEFAULT -1
        , coretime REAL DEFAULT -1
        , comment TEXT
        )""".format(tableName)
        )

def insert(**values):
    simulator = values['simulator']
    keys = []
    vals = []
    for k in values: 
        keys.append(k)
        vals.append("'%s'"%values[k])
    keys.append("time")
    vals.append("datetime('now')")

    keys = ",".join(keys)
    vals = ",".join(vals)
    
    query = """INSERT INTO {} ({}) VALUES ({})""".format(tableName, keys, vals)
    print("Excuting: %s" % query)
    cur_.execute(query)
    conn_.commit()

def main():
    insert({ 'no_of_compartment': 100, 'coretime' : 0.0001, 'simulator' : 'moose' })
    insert({ 'no_of_compartment': 100, 'coretime' : 0.0001, 'simulator' : 'neuron' })
    for c in cur_.execute("SELECT * from %s"%tableName):
        print c

if __name__ == '__main__':
    main()
