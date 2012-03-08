# test_moose_thread.py --- 
# 
# Filename: test_moose_thread.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Thu Mar  8 09:38:02 2012 (+0530)
# Version: 
# Last-Updated: Thu Mar  8 12:32:07 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 121
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Example of using multithreading to run a MOOSE simulation in
# parallel with querying MOOSE objects involved.
# 

# Change log:
# 
# 2012-03-08 12:31:46 (+0530) Initial version by Subha
# 

# Code:

import sys
from datetime import datetime
import threading
import Queue
import moose

worker_queue = Queue.Queue()
status_queue = Queue.Queue()

class WorkerThread(threading.Thread):
    def __init__(self, runtime):
        threading.Thread.__init__(self)
        self.runtime = runtime
        print 'Created WorkerThread of name', self.name
    def run(self):
        print self.name, 'Starting run for', self.runtime, ' seconds'
        moose.reinit()
        moose.start(self.runtime)
        print self.name, 'Finishing'
        worker_queue.put(self.name)

class StatusThread(threading.Thread):        
    def __init__(self, tab):
        threading.Thread.__init__(self)
        self.table = tab
        print 'Created status thread of name', self.name
        
    def run(self):
        while True:
            try:
                worker_queue.get(False)
                print self.name, 'Finishing'
                status_queue.put(self.name)
                return
            except Queue.Empty:
                print self.name, 'Queue is empty'
                print self.name, 'table length', len(tab.vec)
                pass
        
if __name__ == '__main__':
    pg = moose.PulseGen('pg')
    pg.firstDelay = 10.0
    pg.firstLevel = 10.0
    pg.firstWidth = 5.0
    tab = moose.Table('tab')
    moose.connect(tab, 'requestData', pg, 'get_output')
    moose.useClock(0, 'pg,tab', 'process')
    t1 = WorkerThread(100000)
    t2 = StatusThread(tab)
    t2.start()
    t1.start()
    status_queue.get()
    tab.xplot('threading_demo.dat', 'pulsegen_output')
    print 'Ending threading_demo: final length of table', len(tab.vec) 

# 
# threading_demo.py ends here
