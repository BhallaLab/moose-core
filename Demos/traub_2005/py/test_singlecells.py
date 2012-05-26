# test_singlecells.py --- 
# 
# Filename: test_singlecells.py
# Description: 
# Author: 
# Maintainer: 
# Created: Thu May 24 17:45:52 2012 (+0530)
# Version: 
# Last-Updated: Fri May 25 18:40:13 2012 (+0530)
#           By: subha
#     Update #: 78
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 
# 
# Code:

import sys
import os
os.environ['NUMPTHREADS'] = '1'
sys.path.append('../../../python')
import moose
from init import init
import cells

init()
# prototypes = cells.init_prototypes()
container = moose.Neutral('/test')
moose.setClock(0, 25e-6)
moose.setClock(1, 25e-6)
moose.setClock(2, 25e-6)
moose.setClock(3, 0.25e-3)

def testSupPyrRS():    
    cell =  moose.copy(cells.SupPyrRS.prototype, container, 'SupPyrRS')
    cell = cells.SupPyrRS(cell)
    print 'Created cell', cell.path
    vmtable = moose.Table('%s/vmSupPyrRS' % (container.path))
    injection = moose.PulseGen('%s/injection' % (container.path))
    injection.firstDelay = 50e-3
    injection.firstWidth = 500e-3
    injection.firstLevel = 1e-9
    moose.connect(injection, 'outputOut', cell.soma, 'injectMsg')
    moose.connect(vmtable, 'requestData', cell.soma, 'get_Vm')
    moose.useClock(0, '%s,%s/##' % (cell.path, cell.path), 'init')
    moose.useClock(1, '%s,%s/##' % (cell.path, cell.path), 'process')
    moose.useClock(3, '%s' % (vmtable.path), 'process')
    moose.reinit()
    print 'reinit done'
    moose.start(1.0)
    print 'Simulation finished'
    vmtable.xplot(cell.name + '.dat', 'Vm')

if __name__ == '__main__':
    testSupPyrRS()
    
# 
# test_singlecells.py ends here
