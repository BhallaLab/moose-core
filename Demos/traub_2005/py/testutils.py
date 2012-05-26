# test_utils.py --- 
# 
# Filename: test_utils.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat May 26 10:41:37 2012 (+0530)
# Version: 
# Last-Updated: Sat May 26 16:13:35 2012 (+0530)
#           By: subha
#     Update #: 129
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

import numpy as np
from matplotlib import pyplot as plt

import moose

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
    pulsegen.secondLevel = -1e-9
    pulsegen.secondDelay = 150e-3
    pulsegen.secondWidth = 100e-3
    return pulsegen

def setup_single_compartment(container_path, channel_proto, Gbar=1.0e-12):
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
    moose.setClock(0, 1e-6)
    moose.setClock(1, 1e-6)
    moose.useClock(0, '%s/##' % container_path, 'init')
    moose.useClock(1, '%s/##' % container_path, 'process')
    return {'compartment': comp,
            'stimulus': pulsegen,
            'channel': channel,
            'Vm': vm_table,
            'Gk': gk_table}
    

def compare_data_arrays(left, right, plot=False):
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

    """
    if len(left.shape) != len(right.shape):
        print left.shape, right.shape
        raise ValueError('Arrays to be compared must have same dimensions.')
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
        xp = np.arange(left.shape[0]) * 1.0 / left.shape[0]
        fp = left
        yp = right
    elif len(right.shape) == 2:
        print right.shape
        x = right[:,0]
        xp = left[:,0]
        fp = left[:,1]
        yp = right[:,1]
    else:
        raise ValueError('Cannot handle more than 2 dmensional arrays.')
    if left.shape[0] != right.shape[0]:
        print 'Array sizes not matching: (%d <> %d) - interpolating' % (left.shape[0], right.shape[0])
        y = np.interp(x, xp, fp)
    else:
        y = fp
    err = y - yp
    # I measure a conservative relative error as maximum of all the
    # errors between pairs of points with
    all_y = np.r_[fp, yp]
    if plot:
        # plt.subplot(221)
        plt.plot(x, yp, 'bx', label='right')
        # plt.legend()
        # plt.subplot(222)
        plt.plot(xp, fp, 'g+', label='left')
        # plt.legend()
        # plt.subplot(223)
        plt.plot(x, err, 'r^', label='error')
        plt.legend()
        plt.show()
    return max(err)/(max(all_y) - min(all_y))
    
    
# 
# test_utils.py ends here
