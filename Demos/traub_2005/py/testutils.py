# test_utils.py --- 
# 
# Filename: test_utils.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat May 26 10:41:37 2012 (+0530)
# Version: 
# Last-Updated: Fri Jun  1 10:45:36 2012 (+0530)
#           By: subha
#     Update #: 287
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

# Code:

import os
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('../../../python')

import uuid
import numpy as np
from matplotlib import pyplot as plt
import unittest

import moose
import config
import channelbase

simtime = 350e-3

simdt = 1e-6

lib = moose.Neutral(config.modelSettings.libpath)

def make_testcomp(containerpath):
    comp = moose.Compartment('%s/testcomp' % (containerpath))
    comp.Em = -65e-3
    comp.initVm = -65e-3
    comp.Cm = 1e-12
    comp.Rm = 1e9
    comp.Ra = 1e5
    comp.Vm = comp.initVm
    return comp

def make_pulsegen(containerpath):
    pulsegen = moose.PulseGen('%s/testpulse' % (containerpath))
    pulsegen.firstLevel = 1e-12
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
    moose.useClock(0, '%s/##[TYPE=Compartment]' % (container_path), 'init')
    moose.useClock(1, '%s/##' % (container_path), 'process')
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
        ref_file = '../nrn/data/%s_%s.dat' % (channelname, param)
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
        # plt.subplot(221)
        plt.plot(x, yp, 'b-.', label='right')
        # plt.legend()
        # plt.subplot(222)
        plt.plot(xp, fp, 'g--', label='left')
        # plt.legend()
        # plt.subplot(223)
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
    

class ChannelTestBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def compare_Vm(self, filename):
        print 'Comparing Vm ...'
        self.ref_vm_data = np.loadtxt(filename)
        err = compare_data_arrays(self.ref_vm_data, np.array(self.vm_data.vec), plot=True)
        self.assertAlmostEqual(err, 0.0)
        print 'OK'

    def compare_Gk(self, filename):
        print 'Comparing Gk ...'
        self.ref_gk_data = np.loadtxt(filename)
        err = compare_data_arrays(self.ref_gk_data, np.array(self.gk_data.vec), plot=True)
        self.assertAlmostEqual(err, 0.0)
        print 'OK'

    
# 
# test_utils.py ends here
