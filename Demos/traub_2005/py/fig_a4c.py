# fig_a4c.py --- 
# 
# Filename: fig_a4c.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Jun 24 18:43:47 2013 (+0530)
# Version: 
# Last-Updated: Tue Jun 25 18:29:26 2013 (+0530)
#           By: subha
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

"""Experiment to reproduce figure A4C from Traub et al 2005"""

import numpy as np
from scipy.optimize import curve_fit
import pylab

import moose
from moose import utils
from cells import TuftedIB
import config

simtime = 100e-3
simdt = 1e-6
plotdt=1e-4

d1 = 'comp_44'
d2 = 'comp_40'

scale_x = 20e-3/150.0 # 150 pixels = 20 ms scalebar
scale_y = 0.5e-9 / 40 # 40 pixels = 0.5 nA scalebar

# These are the computed parameters for the stimulus current in fig A4C
params = np.array([ -2.95506969e-01,   1.88994285e-01,  -1.00969140e-02,
         2.22528805e-04,  -1.82993456e-06])

def poly4(x, a, b, c, d, e):
    return a + x * (b + x * ( c + x * (d + x * e)))

def get_stimulus_fit(filename='curve_toscale.txt', scale_x=1.0, scale_y=1.0):
    data = np.loadtxt(filename)
    popt, pcov = curve_fit(poly4, data[:, 0]*scale_x, data[:, 1]*scale_y)
    return popt

def get_stimulus(path, start=10e-3, end=40e-3, dt=simdt):
    x = np.arange(start, end, dt)
    y = poly4(x, *get_stimulus_fit())
    stim = moose.StimulusTable(path)
    stim.startTime = start
    stim.stopTime = end
    stim.stepSize = dt
    stim.vec = y
    return stim

def create_stim(stimpath, starttime, stoptime, dt):
    stim = moose.StimulusTable(stimpath)
    # stim_data = np.loadtxt(filepath)
    # y = stim_data[:, 1]*1e-9
    y = poly4(np.arange(0, stoptime-starttime, dt)*1e3, *params)*1e-9
    # The paper says the soma was held at -0.2 nA, assume that means a -0.2 nA current was being injected beforehand
    y = y - y.min()
    y[0] = -0.2e-9
    y[-1] = -0.2e-9
    # The data columns are in ms and nA units respectively.
    stim.vec = y
    stim.startTime = starttime
    # stim.stopTime = starttime + (stim_data[:,0][-1] - stim_data[:,0][0])*1e-3
    stim.stopTime = stoptime
    stim.stepSize = dt
    return stim

def alpha_stimulus(stimpath, peak, peaktime, starttime, simtime, dt):
    """Creates a stimulus in the alpha-function form with peak at
    peaktime"""
    t = np.arange(0, simtime, dt)
    y = peak * t * np.exp(1 - t/peaktime) / peaktime
    y1 = np.zeros(len(t))
    indices = np.flatnonzero(t>starttime)
    y1[indices[0]:] = y[:len(t)-indices[0]]
    pylab.plot(t, y1)
    pylab.show()
    stim = moose.StimulusTable(stimpath)
    stim.vec = y1
    stim.startTime = 0
    stim.stopTime = simtime
    stim.stepSize = dt
    return stim
    

def run_model():
    model = moose.Neutral('/model')
    data = moose.Neutral('/data')
    cell = TuftedIB('/model/TuftedIB')
    # stim = get_stimulus('/model/stim')
    # The following, though loading the ImageJ fitted curve in scale, does not reproduce spiking in soma.
    # stim = create_stim('/model/stim', 2e-3, 42e-3, simdt)
    stim = alpha_stimulus('/model/stim', 1.0e-9, 15e-3, 2e-3, simtime, simdt)
    comp_d1 = moose.element('%s/%s' % (cell.path, d1))
    comp_d2 = moose.element('%s/%s' % (cell.path, d2))
    comp_soma = moose.element('%s/%s' % (cell.path, 'comp_1'))
    moose.connect(stim, 'output', comp_d1, 'injectMsg')
    tab_d1 = moose.Table('%s/d1_Vm' % (data.path))
    tab_d2 = moose.Table('%s/d2_Vm' % (data.path))
    tab_soma = moose.Table('%s/soma_Vm' % (data.path))
    tab_stim = moose.Table('%s/stim' % (data.path))
    moose.connect(tab_d1, 'requestData', comp_d1, 'get_Vm')
    moose.connect(tab_d2, 'requestData', comp_d2, 'get_Vm')
    moose.connect(tab_soma, 'requestData', comp_soma, 'get_Vm')
    moose.connect(stim, 'output', tab_stim, 'input')
    utils.setDefaultDt(elecdt=simdt,plotdt2=plotdt)
    utils.assignDefaultTicks()
    moose.reinit()
    utils.stepRun(simtime, 1e5*simdt, logger=config.logger)
    pylab.subplot(211)
    pylab.plot(np.linspace(0, simtime, len(tab_d1.vec)), tab_d1.vec * 1e3, label='D1 Vm (mV)')
    pylab.plot(np.linspace(0, simtime, len(tab_d2.vec)), tab_d2.vec * 1e3, label='D2 Vm (mV)')
    pylab.plot(np.linspace(0, simtime, len(tab_soma.vec)), tab_soma.vec * 1e3, label='SOMA Vm (mV)')
    pylab.legend()
    pylab.subplot(212)
    pylab.plot(np.linspace(0, simtime, len(tab_stim.vec)), tab_stim.vec * 1e9, label='Stimulus (nA)')
    pylab.legend()
    pylab.show()

if __name__ == '__main__':
    run_model()
        

# 
# fig_a4c.py ends here
