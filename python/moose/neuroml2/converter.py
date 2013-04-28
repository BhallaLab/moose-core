# converter.py --- 
# 
# Filename: mtoneuroml.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Apr 22 12:15:23 2013 (+0530)
# Version: 
# Last-Updated: Sun Apr 28 22:25:35 2013 (+0530)
#           By: subha
#     Update #: 543
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Utility for converting a MOOSE model into NeuroML2. This uses Python
# libNeuroML.
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

#!!!!! TODO: unit conversion !!!!

from collections import deque
import numpy as np
from scipy.optimize import curve_fit

import moose
from moose.utils import autoposition
import neuroml 


def convert_morphology(root, positions='auto'):
    """Convert moose neuron morphology contained under `root` into a
    NeuroML object. The id of the return object is
    {root.name}_morphology. Each segment object gets the numeric value
    of the moose id of the object. The name of the segments are same
    as the corresponding moose compartment.

    Parameters
    ----------
    root : a moose element containing a single cell model.
    
    positions : string
    flag to indicate if the positions of the end points of the
    compartments are explicitly available in the compartments or
    should be automatically generated.  Possible values:
    
    `auto` - automatically generate z coordinates using length of the
    compartments.
    
    `explicit` - model has explicit coordinates for all compartments.

    Return
    ------
    a neuroml.Morphology instance.

    """
    if positions == 'auto':
        queue = deque([autoposition(root)])
    elif positions == 'explicit':
        compartments = moose.wildcardFind('%s/##[TYPE=Compartment]' % (root.path))    
        queue = deque([compartment for compartment in map(moose.element, compartments)
                  if len(compartment.neighbours['axial']) == 0])
        if len(queue) != 1:
            raise Exception('There must be one and only one top level compartment. Found %d' % (len(topcomp_list)))
    else:
        raise Exception('allowed values for keyword argument positions=`auto` or `explicit`')
    comp_seg = {}
    parent = None
    while len(queue) > 0:
        compartment = queue.popleft()
        proximal = neuroml.Point3DWithDiam(x=compartment.x0,
                                           y=compartment.y0, 
                                           z=compartment.z0,
                                           diameter=compartment.diameter)
        distal = neuroml.Point3DWithDiam(x=compartment.x,
                                         y=compartment.y,
                                         z=compartment.z, 
                                         diameter=compartment.diameter)
        plist = map(moose.element, compartment.neighbours['axial'])
        try:            
            parent = neuroml.SegmentParent(segments=comp_seg[moose.element(plist[0])].id)
        except (KeyError, IndexError) as e:
            parent = None
        segment = neuroml.Segment(id=compartment.id_.value,
                                  proximal=proximal, 
                                  distal=distal, 
                                  parent=parent)        
        # TODO: For the time being using numerical value of the moose
        # id for neuroml id.This needs to be updated for handling
        # array elements
        segment.name = compartment.name
        comp_seg[compartment] = segment
        queue.extend([comp for comp in map(moose.element, compartment.neighbours['raxial'])])
    morph = neuroml.Morphology(id='%s_morphology' % (root.name))
    morph.segments.extend(comp_seg.values())
    return morph

###########################################
# function defs for curve fitting H-H-Gates
def exponential(x, a, k, x0):
    return a * np.exp(k * (x - x0))

def sigmoid(x, a, k, x0):
    return a / (np.exp(k * (x - x0)) + 1.0)

def linoid(x, a, k, x0):
    """The so called linoid function. Called explinear in neurml.""" 
    denominator = np.exp(k * (x - x0)) - 1.0
    # Linoid often includes a zero denominator
    zero_idx = np.flatnonzero(denominator == 0)
    print 'Zeroidx', zero_idx
    # Do something to avoid zero div
    # if len(zero_idx) > 0:
    #     for ii in zero_idx:
    #         denominator[ii] = 1e-10
    ret = a * (x - x0) / denominator
    infidx = np.flatnonzero(ret == np.inf)
    print 'infindex', infidx
    if len(infidx) > 0:
        for ii in infidx:
            if ii == 0:
                ret[ii] = ret[ii+1] - (ret[ii+2] - ret[ii+1])
            elif ii == len(ret):
                ret[ii] = ret[ii-1] + (ret[ii-1] - ret[ii-2])
            else:
                ret[ii] = (ret[ii+1] + ret[ii+2]) * 0.5
    return ret

def double_exp(x, a,  k1, x1, k2, x2):
    """For functions of the form:

    a / (exp(k1 * (x - x1)) + exp(k2 * (x - x2)))

    """
    return a / (np.exp(k1 * (x - x1)) + np.exp(k2 * (x - x2)))

# Map from the above functions to corresponding neuroml class
fn_rate_map = {
    exponential: 'HHExpRate',
    sigmoid: 'HHSigmoidRate',
    linoid: 'HHExpLinearRate',
}

# end: function defs for curve fitting H-H-Gates
###########################################

# import pylab

def find_ratefn(x, y):
    """Find the function that fits the rate function best. This will try
    exponential, sigmoid and linoid and return the best fit.

    Needed until NeuroML2 supports tables or MOOSE supports
    functions.

    Parameters
    ----------
    x: 1D array
    independent variable

    y: 1D array
    function values

    """
    functions = [exponential, sigmoid, linoid]
    rms_error = 1.0 # arbitrarily setting this
    best_fn = None
    best_p = None
    for fn in functions:
        try:
            popt, pcov = curve_fit(fn, x, y)
        except RuntimeError as e:
            print fn, e
            # This can be reached in case maxfev is reached
            continue
        error = y - fn(x, *popt)
        erms = np.sqrt(np.mean(error**2))
        # print erms, rms_error, fn
        # pylab.plot(x, y, 'b-')
        # pylab.plot(x, fn(x, *popt), 'r-.')
        # pylab.show()
        if erms < rms_error:
            rms_error = erms
            best_fn = fn
            best_p = popt
    return (best_fn, best_p)    

def convert_hhgate(gate):
    """Convert a MOOSE gate into GateHHRates in NeuroML"""
    hh_rates = neuroml.GateHHRates(id=gate.id_.value, name=gate.name)
    alpha = gate.tableA.copy()
    beta = gate.tableB - alpha
    vrange = np.linspace(gate.min, gate.max, len(alpha))
    afn, ap = find_ratefn(vrange, alpha)
    bfn, bp = find_ratefn(vrange, beta)
    if afn is None:
        raise Exception('could not find a fitting function for `alpha`')
    if bfn is  None:
        raise Exception('could not find a fitting function for `alpha`')
    hh_rates.forward_rate = neuroml.HHRate(type=fn_rate_map[afn], 
                                           midpoint='%gmV' % (ap[2]),
                                           scale='%gmV' % (ap[1]),
                                           rate='%gper_ms' % (ap[0]))
    hh_rates.reverse_rate = neuroml.HHRate(type=fn_rate_map[bfn], 
                                           midpoint='%gmV' % (bp[2]),
                                           scale='%gmV' % (bp[1]),
                                           rate='%gper_ms' % (bp[0]))
    return hh_rates
                                           
    
def convert_hhchannel(channel):
    """Convert a moose HHChannel object into a neuroml element.

    TODO: need to check useConcentration option for Ca2+ and V
    dependent gates. How to handle generic expressions???

    """
    nml_channel = neuroml.IonChannel(id=channel.id_.value,
                                     name=channel.name,
                                     type='ionChannelHH',
                                     conductance = channel.Gbar)
    if channel.Xpower > 0:
        hh_rate_x = convert_hhgate(channel.gateX[0])
        hh_rate_x.instances = channel.Xpower
        nml_channel.gate.append(hh_rate_x)
    if channel.Ypower > 0:
        hh_rate_y = convert_hhgate(channel.gateY[0])
        hh_rate_y.instances = channel.Ypower
        nml_channel.gate.append(hh_rate_y)
    if channel.Zpower > 0:
        hh_rate_z = convert_hhgate(channel.gateZ[0])
        hh_rate_y.instances = channel.Zpower
        nml_channel.gate.append(hh_rate_z)
    return nml_channel

# 
# converter.py ends here
