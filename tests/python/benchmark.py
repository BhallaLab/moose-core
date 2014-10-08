# import os
# import subprocess
# import uuid
import sys
from datetime import datetime

sys.path.append('../../python')
try:
    import moose
except ImportError:    
    print 'Please include the directory containing moose.py and _moose.so in your PYTHONPATH environmental variable.'
    sys.exit(1)

def time_creation(n=1000):
    elist = []
    start = datetime.now()
    for ii in range(n):
        elist.append(moose.Neutral('a_%d' % (ii)))
    end = datetime.now()
    delta = end - start
    print 'total time to create %d Neutral elements: %g' % (n, delta.days * 86400 + delta.seconds + delta.microseconds * 1e-6)
    return delta

if __name__ == '__main__':
    time_creation()
