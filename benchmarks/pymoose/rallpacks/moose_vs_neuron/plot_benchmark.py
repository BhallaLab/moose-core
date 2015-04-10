#!/usr/bin/env python
"""plot_benchmark.py: 

    Plot the benchmarks.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
sys.path.append(sys.argv[1])
import _profile
import pylab
import sqlite3 as sql
from collections import defaultdict
import os

benchmark = defaultdict(list)
dbFile = _profile.dbFile
tableName = _profile.tableName

def plotBenchmark(d):
    print("INFO: Connection to %s" % dbFile)
    db = sql.connect(os.path.join(d, dbFile))
    cur = db.cursor()
    for sim in ['moose', 'neuron']:
        query = """SELECT coretime, no_of_compartment FROM {} WHERE
        simulator='{}'""".format(tableName, sim)
        for c in cur.execute(query):
            benchmark[sim].append(c)
    for k in benchmark:
        vals = benchmark[k]
        time, compt = zip(*vals)
        pylab.plot(compt, time, '.', label=k)
        pylab.legend(loc='best', framealpha=0.4)
        pylab.xlabel("No of compartment")
        pylab.ylabel("Time taken (sec)")

    filename = "{}_benchmark.png".format(d)
    print("Saving benchmark to %s" % filename)
    pylab.savefig(filename)

def main():
    plotBenchmark(sys.argv[1])

if __name__ == '__main__':
    main()
