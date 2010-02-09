# trbutil.py --- 
# 
# Filename: trbutil.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Jun  5 13:59:40 2009 (+0530)
# Version: 
# Last-Updated: Tue Feb  9 14:52:04 2010 (+0100)
#           By: Subhasis Ray
#     Update #: 57
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
import config
from subprocess import call
import pylab
import gzip 
def almost_equal(left, right, epsilon=1e-6):
    """check if two floats are almost equal"""
    if left == right:
        return True
    if abs(left) > abs(right):
        return (1 - right / left) < epsilon
    else:
        return ( 1 - left / right) < epsilon
#!almost_equal

def read_nrn_data(filename, hoc_script=None):
    data = None
    filepath = '../nrn/data/' + filename
    try:
        data = pylab.loadtxt(filepath)
    except IOError:
        try:
            data = pylab.loadtxt(filepath + '.gz')
        except IOError:
            call([config.neuron_bin, hoc_script], cwd='../nrn')
            data = pylab.loadtxt(filepath)
    return data

def do_plot(class_name, mus_t, mus_ca, mus_vm, nrn_t=None, nrn_ca=None, nrn_vm=None):
    '''Plot the membrane potential over time in both moose and neuron.'''
    if nrn_vm is None or len(nrn_vm) is 0:
        nrn_t = pylab.array()
        nrn_vm = pylab.array() 
        nrn_ca  = pylab.array()
    pylab.subplot(211)
    pylab.plot(nrn_t, nrn_vm, 'y-', label='NEURON')
    pylab.plot(mus_t, mus_vm, 'g-.', label='MOOSE')
    pylab.title('Vm in presynaptic compartment of %s' % class_name)
    pylab.legend()
    pylab.subplot(212)
    pylab.plot(nrn_t, nrn_ca, 'r-', label='NEURON')
    pylab.plot(mus_t, mus_ca, 'b-.', label='MOOSE')
    pylab.title('[Ca2+] in soma of %s' % class_name)
    pylab.legend()
    pylab.show()


if __name__ == '__main__':
    read_nrn_data('asa')

# 
# trbutil.py ends here
