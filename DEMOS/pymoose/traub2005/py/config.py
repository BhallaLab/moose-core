# config.py --- 
# 
# Filename: config.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Apr 17 14:36:30 2009 (+0530)
# Version: 
# Last-Updated: Fri Oct 21 16:08:03 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 127
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# This contains global variables used in other modules.
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

# # These are for memory profiling with heapy
# from guppy import hpy
# heapy = hpy()
# # Up to here

import sys
import os
from datetime import datetime
import ConfigParser as configparser
import logging
import numpy
# Try to import matplotlib. Code using matplotlib should be excluded
# based on has_pylab flag.
has_pylab = True
try:
    import pylab
except ImportError:
    has_pylab = False

import moose

timestamp = datetime.now()
mypid = os.getpid()
rngseed = None
stochastic = False # If set to True, the synapses are stochastic, otherwise the deterministic ones are used.

def reseed(seed):
    """This is intended to be a single point of reseeding the
    RNG. numpy.random.seed should not be called anywhere else in the
    simulation setup.


    seed -- seed to be used for the simulation.
    """
    global rngseed
    if rngseed is not None:
        raise Warning('Random number generator already seeded with: %s' % (str(seed)))
    rngseed = seed
    numpy.random.seed(rngseed)

moose_rngseed = None

def moose_reseed(seed):
    """This is for changing the moose' built-in RNG"""
    global moose_rngseed
    if moose_rngseed is not None:
        raise Warning('Random number generator already seeded with: %s' % (str(seed)))
    moose_rngseed = seed
    moose.PyMooseBase.getContext().srandom(seed)
    
#---------------------------------------------------------------------
# configuration for saving simulation data
#---------------------------------------------------------------------

data_dir = os.path.join('data', timestamp.strftime('%Y_%m_%d'))
if not os.access(data_dir, os.F_OK):
    os.mkdir(data_dir)

filename_suffix = '_%s_%d' % (timestamp.strftime('%Y%m%d_%H%M%S'), mypid)
DATA_FILENAME = os.path.join(data_dir, 'data%s.h5' % (filename_suffix))
MODEL_FILENAME = os.path.join(data_dir, 'network%s.h5' % (filename_suffix))
#---------------------------------------------------------------------
# moose components
#---------------------------------------------------------------------
context = moose.PyMooseBase.getContext()
lib = moose.Neutral('/library')
root = moose.Neutral("/")
clockjob = moose.ClockJob('sched/cj')
clockjob.autoschedule = 1
solver = 'hsolve'
simdt = 0.5e-4
plotdt = 1e-3
gldt = 1e-2
vmin = -120e-3
vmax = 40e-3
ndivs = 640
dv = (vmax - vmin) / ndivs
channel_name_list = ['AR','CaPool','CaL','CaT','CaT_A','K2','KA','KA_IB','KAHP','KAHP_DP','KAHP_SLOWER','KC','KC_FAST','KDR','KDR_FS','KM','NaF','NaF2','NaF_TCR','NaP','NaPF','NaPF_SS','NaPF_TCR', 'NaF2_nRT']

channel_map = {'AR': 'ar',
	       'CaPool': 'cad',
	       'CaL': 'cal',
	       'CaT': 'cat',
	       'CaT_A': 'cat_a',
	       'K2': 'k2',
	       'KA': 'ka',
	       'KA_IB': 'ka_ib',
	       'KAHP': 'kahp',
	       'KAHP_DP': 'kahp_deeppyr',
	       'KAHP_SLOWER': 'kahp_slower',
	       'KC': 'kc',
	       'KC_FAST': 'kc_fast',
	       'KDR': 'kdr',
	       'KDR_FS': 'kdr_fs',
	       'KM': 'km',
	       'NaF': 'naf',
	       'NaF2': 'naf2',
	       'NaF2_nRT': 'naf2',
	       'NaF_TCR': 'naf_tcr',
	       'NaP': 'nap',
	       'NaPF': 'napf',
	       'NaPF_SS': 'napf_spinstell',
	       'NaPF_TCR': 'napf_tcr'	      
	       }

channel_lib = {}

#---------------------------------------------------------------------
# Logging
#---------------------------------------------------------------------
def handleError(self, record):
    raise

LOG_FILENAME = os.path.join(data_dir, 'traub2005%s.log' % (filename_suffix))
LOG_LEVEL = logging.DEBUG
logging.Handler.handleError = handleError
logging.basicConfig(filename=LOG_FILENAME, level=LOG_LEVEL, format='%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s: %(message)s', filemode='w')
# logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s: %(message)s', filemode='w')

LOGGER = logging.getLogger('traub2005')
BENCHMARK_LOGGER = logging.getLogger('traub2005.benchmark')
BENCHMARK_LOGGER.setLevel(logging.DEBUG)
benchmarking=True # Dump benchmarking information

# Unit Conversion Factors
uS = 1e-6 # micro Siemens to Siemens
ms = 1e-3 # milli second to second
mV = 1e-3 # milli Volt to Volt


#---------------------------------------------------------------------
# NEURON 
#---------------------------------------------------------------------    
# Locate the neuron binaries
import subprocess
neuron_bin = None
try:
    which_neuron = subprocess.Popen(['which', 'nrngui'], stdout=subprocess.PIPE, close_fds=True)
    for neuron_bin in which_neuron.stdout:
        neuron_bin = neuron_bin.strip()
        if neuron_bin:
            LOGGER.info('nrngui fount at: %s' % (neuron_bin))
            break
except Exception, e:
    print e
    print 'config.py: could not locate nrngui. Set the full path to it in this file.'
finally:
    if not neuron_bin:
        print 'config.py: could not locate nrngui. Set variable neuron_bin to the full path to it in this file. Otherwise the .plot files in the data directory will be used for plotting'

#---------------------------------------------------------------------
# Try to read runcontrol file using ConfigParser module. This should
# be in a format similar to MS Windows .INI files.
#---------------------------------------------------------------------

# The following are known only at runtime. Not to be in configuration file.
#
# 'timestamp': timstamp,
# 'mypid': mypid,
runconfig = configparser.SafeConfigParser()
runconfig.optionxform = str
runconfig.read(['defaults.ini', 'custom.ini'])
for section in runconfig.sections():    
    for key, value in runconfig.items(section):
        LOGGER.debug('%s: %s = %s' % (section, key, value))
        
try:
    _rngseed = int(runconfig.get('numeric', 'rngseed'))
except ValueError:
    rngseed = None
try:
    __seed = int(runconfig.get('numeric', 'moose_rngseed'))
    moose_reseed(__seed)
    LOGGER.info('MOOSE RNG SEED SET TO %d' % (moose_rngseed))
except ValueError:
    moose_rngseed = None

to_reseed = runconfig.get('numeric', 'reseed') in ['Yes', 'yes', 'True', 'true', '1']
stochastic = runconfig.get('numeric', 'stochastic') in ['Yes', 'yes', 'True', 'true', '1']
solver = runconfig.get('numeric', 'solver')
simtime = float(runconfig.get('scheduling', 'simtime'))
simdt = float(runconfig.get('scheduling', 'simdt'))
plotdt = float(runconfig.get('scheduling', 'plotdt'))
clockjob.autoschedule = runconfig.get('scheduling', 'autoschedule') in ['Yes', 'yes', 'True', 'true', '1']
default_releasep = float(runconfig.get('synapse', 'releasep')) 
if to_reseed:
    if _rngseed is not None:
        reseed(_rngseed)
        LOGGER.info('NUMPY RNG SEED SET TO %d' % (rngseed))
    else:
        reseed(mypid)
        LOGGER.info('NUMPY RNG SEED SET TO %d' % (rngseed))


# 
# config.py ends here
