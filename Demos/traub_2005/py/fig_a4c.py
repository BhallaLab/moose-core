# fig_a4c.py --- 
# 
# Filename: fig_a4c.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Jun 24 18:43:47 2013 (+0530)
# Version: 
# Last-Updated: Mon Jun 24 23:39:18 2013 (+0530)
#           By: subha
#     Update #: 70
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

def poly4(x, a, b, c, d, e):
    return a + x * (b + x * ( c + x * (d + x * e)))

def get_stimulus_fit(filename='curve.txt'):
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

def run_model():
    model = moose.Neutral('/model')
    cell = TuftedIB('/model/TuftedIB')
    stim = get_stimulus('/model/stim')
    comp_d1 = moose.element('%s/%s' % (cell.path, d1))
    comp_d2 = moose.element('%s/%s' % (cell.path, d2))
    comp_soma = moose.element('%s/%s' % (cell.path, 'comp_1'))
    moose.connect(stim, 'output', comp_d1, 'injectMsg')
    data = moose.Neutral('/data')
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
    for tab in moose.wildcardFind('/data/##[ISA=Table]'):
        vm = moose.element(tab).vec
        pylab.plot(np.linspace(0, simtime, len(vm)), vm, label=tab[0].name)
    pylab.show()

if __name__ == '__main__':
    run_model()
        

# 
# fig_a4c.py ends here
