#!/usr/bin/env python
# -*- coding: utf-8 -*-

### This program plots a channel's state variables / hinf, htau etc. as a function of voltage.

mechanisms = {
'H_STG': ['minf','mtau'],
'CaS_STG': ['minf','mtau','hinf','htau'],
'CaT_STG': ['minf','mtau','hinf','htau'],
'KA_STG': ['minf','mtau','hinf','htau'],
'Kd_STG': ['ninf','ntau'],
'Na_STG': ['minf','mtau','hinf','htau']
}

import sys
if len(sys.argv)<2:
    print "Please print a channel name to be plotted from", mechanisms.keys()
    sys.exit(1)
channel_name = sys.argv[1]
if channel_name in mechanisms:
    mechanism_vars = mechanisms[channel_name]
else:
    print "Undefined channel, please use one of", mechanisms.keys()
    sys.exit(1)

import math

# The PYTHONPATH should contain the location of moose.py and _moose.so
# files.  Putting ".." with the assumption that moose.py and _moose.so
# has been generated in ${MOOSE_SOURCE_DIRECTORY}/pymoose/ (as default
# pymoose build does) and this file is located in
# ${MOOSE_SOURCE_DIRECTORY}/pymoose/examples
try:
    import moose
    from moose.neuroml import *
except ImportError:
    print "ERROR: Could not import moose."
    print "Please add the directory containing moose.py in your PYTHONPATH"
    import sys
    sys.exit(1)

CELSIUS = 35 # degrees Centigrade
CML = ChannelML({'temperature':CELSIUS})
CML.readChannelMLFromFile('../channels/'+channel_name+'.xml')

from pylab import *
                        
if __name__ == "__main__":
    for varidx in range(len(mechanism_vars)/2): # loop over each inf and tau
        var = ['X','Y','Z'][varidx]

        gate = moose.element('/library/'+channel_name+'/gate'+var)
        VMIN = gate.min
        VMAX = gate.max
        NDIVS = gate.divs
        dv = (VMAX-VMIN)/NDIVS
        # will use same VMIN, VMAX and dv for A and B tables.
        vrange = array([VMIN+i*dv for i in range(NDIVS+1)])
        figure()
        plot(vrange*1000,gate.tableA/gate.tableB,'b-,') # Assume A and B have corresponding number of entries
        xlabel('Voltage (mV)')
        ylabel('steady state value')
        title('state variable '+mechanism_vars[2*varidx]+' of '+channel_name+' vs Voltage (mV)')
        figure()
        plot(vrange*1000,1./gate.tableB*1000.,'b-,')
        xlabel('Voltage (mV)')
        ylabel('tau (ms)')
        title('state variable '+mechanism_vars[2*varidx+1]+' of '+channel_name+' vs Voltage (mV)')
    show()

