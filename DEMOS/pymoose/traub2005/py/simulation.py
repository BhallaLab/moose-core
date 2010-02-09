# simulation.py --- 
# 
# Filename: simulation.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Thu Apr 30 02:25:06 2009 (+0530)
# Version: 
# Last-Updated: Tue Feb  9 14:20:31 2010 (+0100)
#           By: Subhasis Ray
#     Update #: 57
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
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 

# Code:
import os
from datetime import datetime
import moose
import config

channel_lib = {}


class Simulation:
    """This class is a wrapper to control a whole simulation."""
    def __init__(self, name):
        self.name = name
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
        # print 'reset time: ', delta.seconds + 1e-6 * delta.microseconds
        self.start_t = datetime.now()
        config.context.step(float(time))
        self.end_t = datetime.now()
        delta = self.end_t - t1 # include reset time
	config.BENCHMARK_LOGGER.info('%s : RUNTIME [reset + step] (second): %g' %  (self.name, delta.seconds + 1e-6 * delta.microseconds))

    def dump_data(self, directory, time_stamp=False):
        """Save the data in directory. It creates a subdirectory with
        the date of start_t. The files are prefixed with start time in
        HH.MM.SS_ format if time_stamp is True."""
        path = directory
        tables = []
        if not os.access(directory, os.W_OK):
            config.LOGGER.warning('data directory: ' + directory + 'is not writable')
            return
        else:
            path = directory + '/' # + self.start_t.strftime('%Y_%m_%d') + '/'
            if not os.access(path, os.F_OK):
                os.mkdir(path)
        if time_stamp:
            path = path + self.start_t.strftime('%H.%M.%S') + '_'
        for table_id in self.data.children():
            table = moose.Table(table_id)
            tables.append(table)
            file_path = path + table.name + '.plot'
            table.dumpFile(file_path)
            config.LOGGER.info('Dumped data in %s' % (file_path))
        return tables

# 
# simulation.py ends here
