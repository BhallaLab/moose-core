# simulation.py --- 
# 
# Filename: simulation.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Thu Apr 30 02:25:06 2009 (+0530)
# Version: 
# Last-Updated: Tue May 19 16:15:14 2009 (+0530)
#           By: subhasis ray
#     Update #: 17
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# A manager class to handle a whole simulation.
# 
# 

# Change log:
#
# 2009-04-30 02:25:59 (+0530) - factored out the Simulation class from
# test.py to a separate simulation.py.
# 
# 
# 
 

# Code:
import os
from datetime import datetime
import moose
import config


class Simulation:
    """This class is a wrapper to control a whole simulation."""
    def __init__(self):
        self.model = moose.Neutral('model')
        self.data = moose.Neutral('data')
        self.start_t = None
        self.end_t = None
        self.simtime = 50e-3

    def schedule(self):
        config.context.setClock(0, config.simdt)
        config.context.setClock(1, config.simdt)
        config.context.setClock(2, config.plotdt)
        config.context.useClock(0, self.model.path + '/##,'+self.data.path + '/#')
        config.context.useClock(1, self.model.path + '/##[Type=Compartment]', 'init')

    def run(self, time=None):
        if time is None or time <= 0.0:
            time = self.simtime
        else:
            self.simtime = time
        t1 = datetime.now()
        config.context.reset()
        t2 = datetime.now()
        delta = t2 - t1
        print 'reset time: ', delta.seconds + 1e-6 * delta.microseconds
        self.start_t = datetime.now()
        config.context.step(float(time))
        self.end_t = datetime.now()

    def dump_data(self, directory, time_stamp=False):
        """Save the data in directory. It creates a subdirectory with
        the date of start_t. The files are prefixed with start time in
        HH.MM.SS_ format if time_stamp is True."""
        path = directory
        tables = []
        if not os.access(directory, os.W_OK):
            print 'data directory:', directory, 'is not writable'
            return
        else:
            path = directory + '/' + self.start_t.strftime('%Y_%m_%d') + '/'
            if not os.access(path, os.F_OK):
                os.mkdir(path)
        if time_stamp:
            path = path + self.start_t.strftime('%H.%M.%S') + '_'
        for table_id in self.data.children():
            table = moose.Table(table_id)
            tables.append(table)
            file_path = path + table.name + '.plot'
            table.dumpFile(file_path)
            print 'Dumped data in ', file_path
        return tables



# 
# simulation.py ends here
