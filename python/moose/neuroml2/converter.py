# converter.py --- 
# 
# Filename: mtoneuroml.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Apr 22 12:15:23 2013 (+0530)
# Version: 
# Last-Updated: Mon May 20 23:06:36 2013 (+0530)
#           By: subha
#     Update #: 700
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
from matplotlib import pyplot as plt

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
def exponential(x, a, k, x0, y0=0):
    return a * np.exp(k * (x - x0)) + y0

def sigmoid(x, a, k, x0, y0=0):
    return a / (np.exp(k * (x - x0)) + 1.0) + y0

def linoid(x, a, k, x0, y0=0):
    """The so called linoid function. Called explinear in neurml.""" 
    denominator = np.exp(k * (x - x0)) - 1.0
    # Linoid often includes a zero denominator - we need to fill those
    # points with interpolated values (interpolation is simpler than
    # finding limits).
    ret = a * (x - x0) / denominator
    infidx = np.flatnonzero((ret == np.inf) | (ret == -np.inf))
    if len(infidx) > 0:
        for ii in infidx:
            if ii == 0:
                ret[ii] = ret[ii+1] - (ret[ii+2] - ret[ii+1])
            elif ii == len(ret):
                ret[ii] = ret[ii-1] + (ret[ii-1] - ret[ii-2])
            else:
                ret[ii] = (ret[ii+1] + ret[ii+2]) * 0.5
    return ret + y0

def double_exp(x, a, k1, x1, k2, x2, y0=0):
    """For functions of the form:

    a / (exp(k1 * (x - x1)) + exp(k2 * (x - x2)))

    """
    ret = np.zeros(len(x))
    try:
        ret = a / (np.exp(k1 * (x - x1)) + np.exp(k2 * (x - x2))) + y0
    except RuntimeWaring as e:
        print 'Double exponential:', e
        print 'params:', a, k1, x1, k2, x2, y0
    # plt.plot(x, ret, label='%g, %g, %g, %g, %g, %g' % (a, k1, x1, k2, x2, y0))
    return ret

# Map from the above functions to corresponding neuroml class
fn_rate_map = {
    exponential: 'HHExpRate',
    sigmoid: 'HHSigmoidRate',
    linoid: 'HHExpLinearRate',
    double_exp: None,    
}

# These are default starting parameter values
fn_p0_map = {
    exponential: (1.0, -100, 20e-3, 0.0),
    sigmoid: (1.0, 1.0, 0.0, 0.0),
    linoid: (1.0, 1.0, 0.0, 0.0),
    double_exp: (1e-3, -1.0, 0.0, -1.0, 0.0, 0.0),
}


# end: function defs for curve fitting H-H-Gates
###########################################

# import pylab

def randomized_curve_fit(fn, x, y, maxiter=100, best=False):
    """Repeatedly search for a good fit with randomly generated initial
    parameter set. This function first tries with default p0 for
    fn. If that fails to find a good fit, (correlation coeff returned
    by curve_fit being inf is an indication of this), it goes on to
    generate random p0 arrays and try scipy.optimize.curve_fit using
    this p0 until it finds a good fit or the number of iterations
    reaches maxiter.

    Ideally we should be doing something like stochastic gradient
    descent, but I don't know if that might have performance issue in
    pure python. uniform random is good enough for now.

    Parameters
    ----------
    x: ndarray
    values of the independent variable

    y: ndarray
    sample values of the dependent variable

    maxiter: int
    maximum number of iterations

    best: bool
    if true, repeat curve_fit for maxiter and return the case of least
    rms error.

    Returns
    -------
    The return value of scipy.optimize.curve_fit which succeed, or the
    last call to it if maxiter iterations is reached..

    """
    good = True
    p = None
    try:
        p = curve_fit(fn, x, y,  p0=fn_p0_map[fn], full_output=True)
    except (RuntimeError, RuntimeWarning) as e:
        good = False
    # The last entry returned by scipy.optimize.leastsq used by
    # curve_fit is 1, 2, 3 or 4 if it succeeds.
    if (p is None) or np.any(p[1] == np.inf) or (p[-1] not in [1, 2, 3, 4]):
        good = False
    min_err = 1e10
    p_best = None
    if good: 
        if best:
            min_err = sum((y - fn(x, *tuple(p[0])))**2)
            p_best = p
        else:
            return p
    for ii in range(maxiter):
        good = True
        p0 = np.random.uniform(low=min(x), high=max(x), size=len(fn_p0_map[fn]))
        try:
            p = curve_fit(fn, x, y,  p0=p0, full_output=True)
        except (RuntimeError, RuntimeWarning) as e:
            good = False
        if (p is None) or (p[-1] not in [1, 2, 3, 4]) or np.any(p[1] == np.inf):
            good = False
        if good:
            if best:
                err = sum((y - fn(x, *tuple(p[0])))**2)
                if err < min_err:
                    min_err = err 
                    p_best = p
            else:
                print 'Found good', fn.func_name, p[0], p[1]
                break
    if ii == maxiter:
        print fn.func_name, 'reached maxiter'
    return p_best


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
    rms_error = 1e10 # arbitrarily setting this
    best_fn = None
    best_p = None
    for fn in fn_rate_map.keys():
        p = randomized_curve_fit(fn, x, y, best=True)
        if p is None:
            continue
        popt = p[0]
        pcov = p[1]        
        error = y - fn(x, *popt)
        erms = np.sqrt(np.mean(error**2))
        print fn, 'e_rms', erms
        # if fn == double_exp:
        #     plt.plot(x, fn(x, *popt), label='%s' % (fn))
        # print erms, rms_error, fn
        # pylab.plot(x, y, 'b-')
        # pylab.plot(x, fn(x, *popt), 'r-.')
        # pylab.show()
        print fn.func_name, 'rms error:', erms, 'pcov:', pcov
        if erms < rms_error:
            rms_error = erms
            best_fn = fn
            best_p = popt
    # plt.plot(x, y, label='original')
    # plt.legend()
    # plt.show()
    return (best_fn, best_p)    

def define_vdep_rate(fn, name):
    """Define new component type with generic expressions for voltage
    dependent rate.

    """
    ctype = neuroml.ComponentType(name)
    # This is going to be ugly ...
    
    
    
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
    afn_type = fn_rate_map[afn]
    afn_component_type = None
    if afn_type is None:
        afn_type, afn_component_type = define_component_type(afn)
    hh_rates.forward_rate = neuroml.HHRate(type=afn_type, 
                                           midpoint='%gmV' % (ap[2]),
                                           scale='%gmV' % (ap[1]),
                                           rate='%gper_ms' % (ap[0]))
    bfn_type = fn_rate_map[bfn]
    bfn_component_type = None
    if bfn_type is None:
        bfn_type, bfn_component_type = define_component_type(bfn)
    hh_rates.reverse_rate = neuroml.HHRate(type=bfn_type, 
                                           midpoint='%gmV' % (bp[2]),
                                           scale='%gmV' % (bp[1]),
                                           rate='%gper_ms' % (bp[0]))
    return hh_rates, afn_component_type, bfn_component_type
                                           
    
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
