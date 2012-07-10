# test_hsolve_single_comp.py --- 
# 
# Filename: test_hsolve_single_comp.py
# Description: 
# Author: 
# Maintainer: 
# Created: Tue Jul 10 16:16:55 2012 (+0530)
# Version: 
# Last-Updated: Tue Jul 10 20:36:10 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 194
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
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('../python')
import moose
import uuid
import unittest
import numpy as np
import matplotlib.pyplot as plt

def make_testcomp(containerpath):
    comp = moose.Compartment('%s/testcomp' % (containerpath))
    comp.Em = -65e-3
    comp.initVm = -65e-3
    comp.Cm = 1e-12
    comp.Rm = 1e9
    comp.Ra = 1e5
    return comp

def make_pulsegen(containerpath):
    pulsegen = moose.PulseGen('%s/testpulse' % (containerpath))
    pulsegen.firstLevel = 1e-9
    pulsegen.firstDelay = 50e-3
    pulsegen.firstWidth = 100e-3
    pulsegen.secondLevel = -1e-12
    pulsegen.secondDelay = 150e-3
    pulsegen.secondWidth = 100e-3
    pulsegen.count = 3
    pulsegen.delay[2] = 1e9
    return pulsegen

def setup_single_compartment(container_path, channel_proto, Gbar):
    comp = make_testcomp(container_path)
    channel = moose.copy(channel_proto, comp, channel_proto.name)[0]
    moose.connect(channel, 'channel', comp, 'channel')
    channel.Gbar = Gbar
    pulsegen = make_pulsegen(container_path)
    moose.connect(pulsegen, 'outputOut', comp, 'injectMsg')
    vm_table = moose.Table('%s/Vm' % (container_path))
    moose.connect(vm_table, 'requestData', comp, 'get_Vm')
    gk_table = moose.Table('%s/Gk' % (container_path))
    moose.connect(gk_table, 'requestData', channel, 'get_Gk')
    ik_table = moose.Table('%s/Ik' % (container_path))
    moose.connect(ik_table, 'requestData', channel, 'get_Ik')
    moose.setClock(0, simdt)
    moose.setClock(1, simdt)
    moose.setClock(2, simdt)
    moose.useClock(0, '%s/##[TYPE=Compartment]' % (container_path), 'init')
    moose.useClock(1, '%s/##[TYPE=Compartment]' % (container_path), 'process')
    moose.useClock(2, '%s/##[TYPE!=Compartment]' % (container_path), 'process')

    return {'compartment': comp,
            'stimulus': pulsegen,
            'channel': channel,
            'Vm': vm_table,
            'Gk': gk_table,
            'Ik': ik_table}
    
def run_single_channel(channelname, Gbar, simtime):
    testId = uuid.uuid4().int
    container = moose.Neutral('test%d' % (testId))
    params = setup_single_compartment(
        container.path,
        channelbase.prototypes[channelname],
        Gbar)
    vm_data = params['Vm']
    gk_data = params['Gk']
    ik_data = params['Ik']
    moose.reinit()
    print 'Starting simulation', testId, 'for', simtime, 's'
    moose.start(simtime)
    print 'Finished simulation'
    vm_file = 'data/%s_Vm.dat' % (channelname)
    gk_file = 'data/%s_Gk.dat' % (channelname)
    ik_file = 'data/%s_Ik.dat' % (channelname)
    tseries = np.array(range(len(vm_data.vec))) * simdt
    print 'Vm:', len(vm_data.vec), 'Gk', len(gk_data.vec), 'Ik', len(ik_data.vec)
    data = np.c_[tseries, vm_data.vec]
    np.savetxt(vm_file, data)
    print 'Saved Vm in', vm_file
    data = np.c_[tseries, gk_data.vec]
    np.savetxt(gk_file, data)
    print 'Saved Gk in', gk_file
    data = np.c_[tseries, ik_data.vec]
    np.savetxt(ik_file, data)
    print 'Saved Gk in', ik_file
    return params

def compare_channel_data(series, channelname, param, simulator, x_range=None, plot=False):
    if simulator == 'moose':
        ref_file = 'testdata/%s_%s.dat.gz' % (channelname, param)
    elif simulator == 'neuron':
        ref_file = '../nrn/data/%s_%s.dat.gz' % (channelname, param)
    else:
        raise ValueError('Unrecognised simulator: %s' % (simulator))
    ref_series = np.loadtxt(ref_file)
    if plot:
        plt.figure()
        plt.title(channelname)
    return compare_data_arrays(ref_series, series, relative='meany', x_range=x_range, plot=plot)

def compare_data_arrays(left, right, relative='maxw', plot=False, x_range=None):
    """compare two data arrays. They must have the same number of
    dimensions (1 or 2) and represent the same range of x values. In
    case they are 1 dimensional, we take x values as relative position
    of that data point in the total x-range.

    We interpolate the y values for the x-values of the series with
    lower resolution using the heigher resolution series as the
    interpolation table.

    The error is calculated as the maximum difference between the
    interpolated values and the actual values in the lower resolution
    array divided by the difference between the maximum and minimum y
    values of both the series.

    If plot is True, left, right and their difference at common points
    are plotted.

    relative: `rms` - return root mean square of the error values
    `taxicab` - mean of the absolute error values
    `maxw` - max(abs(error))/(max(y) - min(y))    
    `meany` - rms(error)/mean(y)

    x_range : (minx, maxx) range of X values to consider for comparison

    """
    if len(left.shape) != len(right.shape):
        print left.shape, right.shape
        raise ValueError('Arrays to be compared must have same dimensions.')
    # y is the intrepolation result for x array using xp and fp when xp and x do not match.
    # xp and fp are interpolation table's independent and dependent variables
    # yp is a view of the original y values
    x = None
    y = None
    xp = None
    fp = None
    yp = None
    # arbitrarily keep series with more datapoint as left
    if left.shape[0] < right.shape[0]:
        tmp = left
        left = right
        right = tmp
    if len(right.shape) == 1:
        x = np.arange(right.shape[0]) * 1.0 / right.shape[0]
        yp = right
        xp = np.arange(left.shape[0]) * 1.0 / left.shape[0]
        fp = left
    elif len(right.shape) == 2:
        x = right[:,0]
        yp = right[:,1]
        xp = left[:,0]
        fp = left[:,1]
    else:
        raise ValueError('Cannot handle more than 2 dimensional arrays.')
    if left.shape[0] != right.shape[0]:
        print 'Array sizes not matching: (%d <> %d) - interpolating' % (left.shape[0], right.shape[0])
        y = np.interp(x, xp, fp)
    else: # assume we have the same X values when sizes are the same
        y = np.array(fp)
    if x_range:
        indices = np.nonzero((x > x_range[0]) & (x <= x_range[1]))[0]
        y = np.array(y[indices])
        yp = np.array(yp[indices])
        x = np.array(x[indices])
        # We update xp and fp to have the same plotting x-range
        indices = np.nonzero((xp > x_range[0]) & (xp <= x_range[1]))[0]
        xp = xp[indices]
        fp = fp[indices]
    err = y - yp
    print min(err), max(err), min(y), max(y), min(yp), max(yp)
    # I measure a conservative relative error as maximum of all the
    # errors between pairs of points with
    all_y = np.r_[y, yp]
    if plot:        
        plt.plot(x, yp, 'b-.', label='right')
        plt.plot(xp, fp, 'g--', label='left')
        plt.plot(x, err, 'r:', label='error')
        plt.legend()
        plt.show()
    if relative == 'rms':
        return np.sqrt(np.mean(err**2))
    elif relative == 'taxicab':
        return np.mean(np.abs(err))
    elif relative == 'maxw':
        return max(np.abs(err))/(max(all_y) - min(all_y))
    elif relative == 'meany':
        return np.sqrt(np.mean(err**2)) / np.mean(all_y)
    else:
        return err

def setup_single_comp_model(model_container, data_container, simdt):
    cell = moose.Neuron(model_container.path + '/neuron')
    soma = make_testcomp(cell.path)
    pulsegen = make_pulsegen(model_container.path)
    pulsegen.firstDelay = simdt
    moose.connect(pulsegen, 'outputOut', soma, 'injectMsg')
    vm_table = moose.Table(data_container.path + '/Vm')
    moose.connect(vm_table, 'requestData', soma, 'get_Vm')
    return {'pulsegen': pulsegen,
            'soma': soma,
            'cell': cell,
            'vm_table': vm_table}

def setup_hsolve_single_comp_model(model_container, data_container, simdt):
    elements = setup_single_comp_model(model_container, data_container, simdt)
    solver = moose.HSolve(model_container.path + '/hsolve')
    solver.dt = simdt
    solver.target = elements['cell'].path
    elements['solver'] = solver
    return elements

def setup_passive_cable(model_container, data_container, comp_count, simdt):
    """We are considering a passive cable with identical
    compartments. Stimulus is applied on first compartment, data is
    recorded from last compartment."""
    cell = moose.Neuron(model_container.path + '/neuron')
    comp = [moose.Compartment('%s/comp_%d' % (cell.path, ii)) for ii in range(comp_count)]
    for c in comp:
        c.Em = -65e-3
        c.initVm = -65e-3
        c.Cm = 1e-12
        c.Rm = 1e9
        c.Ra = 1e5
    for ii in range(1, comp_count):
        moose.connect(comp[ii-1], 'axial', comp[ii], 'raxial')
    pulsegen = make_pulsegen(model_container.path)
    pulsegen.firstDelay = simdt
    moose.connect(pulsegen, 'outputOut', comp[0], 'injectMsg')
    vm_table = moose.Table(data_container.path + '/Vm')
    moose.connect(vm_table, 'requestData', comp[-1], 'get_Vm')
    return {'pulsegen': pulsegen,
            'soma': comp[-1],
            'cell': cell,
            'vm_table': vm_table}

def setup_hsolve_passive_cable(model_container, data_container, comp_count, simdt):
    elements = setup_passive_cable(model_container, data_container, comp_count, simdt)
    solver = moose.HSolve(model_container.path + '/hsolve')
    solver.dt = simdt
    solver.target = elements['cell'].path
    elements['solver'] = solver
    return elements    

def run_simulation(container, simdt, simtime):
        moose.setClock(0, simdt)
        moose.setClock(1, simdt)
        moose.setClock(2, simdt)
        moose.setClock(3, simdt)
        moose.useClock(0, container.path + '/##[TYPE=Compartment]', 'init')
        moose.useClock(1, container.path + '/##[TYPE=Compartment]', 'process')
        moose.useClock(2, container.path + '/##[TYPE!=Compartment]', 'process')
        moose.reinit()
        moose.start(simtime)
        
    
class TestSingleCompPassive(unittest.TestCase):
    """Test hsolve with a single compartment"""
    def setUp(self):
        self.simdt = 1e-4
        self.testId = uuid.uuid4().int
        test_container = moose.Neutral('test%d' % (self.testId))
        hsolve_data = moose.Neutral('%s/hsolvedata' % (test_container.path))
        hsolve_model = moose.Neutral('%s/hsolvemodel' % (test_container.path))
        self.hsolve_elements = setup_hsolve_single_comp_model(hsolve_model, hsolve_data, self.simdt)
        fwdeuler_model = moose.Neutral('%s/fwdeulermodel' % (test_container.path))
        fwdeuler_data = moose.Neutral('%s/fwdeulerdata' % (test_container.path))
        self.fwdeuler_elements = setup_single_comp_model(fwdeuler_model, fwdeuler_data, self.simdt)
        run_simulation(test_container, self.simdt, 1000*self.simdt)
        print self.hsolve_elements['vm_table'].vec
        print self.fwdeuler_elements['vm_table'].vec

    def testHSolveSingleComp(self):
        err = compare_data_arrays(self.hsolve_elements['vm_table'].vec, self.fwdeuler_elements['vm_table'].vec, plot=True)
        print 'Difference:', err
        self.assertLess(err, 0.01)

class TestPassiveCable(unittest.TestCase):
    """Test a passive cable in forward Euler and hsolve"""
    def setUp(self):
        self.comp_count = 100
        self.simdt = 1e-6
        self.testId = uuid.uuid4().int
        test_container = moose.Neutral('test%d' % (self.testId))
        hsolve_data = moose.Neutral('%s/hsolvedata' % (test_container.path))
        hsolve_model = moose.Neutral('%s/hsolvemodel' % (test_container.path))
        self.hsolve_elements = setup_hsolve_passive_cable(hsolve_model, hsolve_data, self.comp_count, self.simdt)
        fwdeuler_model = moose.Neutral('%s/fwdeulermodel' % (test_container.path))
        fwdeuler_data = moose.Neutral('%s/fwdeulerdata' % (test_container.path))
        self.fwdeuler_elements = setup_passive_cable(fwdeuler_model, fwdeuler_data, self.comp_count, self.simdt)
        # self.hsolve_elements['pulsegen'].firstLevel = 1e-9
        # self.fwdeuler_elements['pulsegen'].firstLevel = 1e-9
        run_simulation(test_container, self.simdt, 1000*self.simdt)
        print self.hsolve_elements['vm_table'].vec
        print self.fwdeuler_elements['vm_table'].vec
        

    def testHSolvePassiveCable(self):
        compare_data_arrays(self.hsolve_elements['vm_table'].vec, self.fwdeuler_elements['vm_table'].vec, plot=True)
        
        
        
if __name__ == '__main__':
    unittest.main()


# 
# test_hsolve_single_comp.py ends here
