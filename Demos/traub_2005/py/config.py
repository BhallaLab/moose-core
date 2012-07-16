# config.py --- 
# 
# Filename: config.py
# Description: 
# Author: 
# Maintainer: 
# Created: Wed May 23 11:31:40 2012 (+0530)
# Version: 
# Last-Updated: Mon Jul 16 16:47:11 2012 (+0530)
#           By: subha
#     Update #: 91
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
import settings

# These settings are to imitate sedml entities for configuring simulation
simulationSettings = settings.SimulationSettings()
modelSettings = settings.ModelSettings()
analogSignalRecordingSettings = settings.DataSettings()
spikeRecordingSettings = settings.DataSettings()
changeSettings = settings.ChangeSettings()

simulationSettings.endTime = 10.0

modelSettings.container = '/network'
modelSettings.libpath = '/library'
modelSettings.protodir = 'proto'

modelSettings.morph_has_postion = False
modelSettings.populationSize['SupPyrRS'] = 1000
modelSettings.populationSize['SupPyrFRB'] = 50
modelSettings.populationSize['SupBasket'] = 90       
modelSettings.populationSize['SupAxoaxonic'] = 90
modelSettings.populationSize['SupLTS'] = 90
modelSettings.populationSize['SpinyStellate'] =	240
modelSettings.populationSize['TuftedIB'] = 800
modelSettings.populationSize['TuftedRS'] = 200
modelSettings.populationSize['DeepBasket'] = 100
modelSettings.populationSize['DeepAxoaxonic'] = 100
modelSettings.populationSize['DeepLTS'] = 100
modelSettings.populationSize['NontuftedRS'] = 500
modelSettings.populationSize['TCR'] = 100
modelSettings.populationSize['nRT'] = 100       

analogSignalRecordingSettings.targets = [
    '/network/SupPyrRS/#[NAME=comp_0]',
    '/network/SupPyrFRB/#[NAME=comp_0]',
    '/network/SupBasket/#[NAME=comp_0]',
    '/network/SupAxoaxonic/#[NAME=comp_0]',
    '/network/SupLTS/#[NAME=comp_0]',
    '/network/SpinyStellate/#[NAME=comp_0]',
    '/network/TuftedIB/#[NAME=comp_0]',
    '/network/TuftedRS/#[NAME=comp_0]',
    '/network/DeepBasket/#[NAME=comp_0]',
    '/network/DeepAxoaxonic/#[NAME=comp_0]',
    '/network/DeepLTS/#[NAME=comp_0]',
    '/network/NontuftedRS/#[NAME=comp_0]',
    '/network/TCR/#[NAME=comp_0]',
    '/network/nRT/#[NAME=comp_0]',
    ]

analogSignalRecordingSettings.fractions = 0.1
analogSignalRecordingSettings.fields = {
    'Vm': 'AnalogSignal',
    'CaPool/conc': 'AnalogSignal',
    }

spikeRecordingSettings.targets = [
    '/network/SupPyrRS/#[NAME=comp_0]',
    '/network/SupPyrFRB/#[NAME=comp_0]',
    '/network/SupBasket/#[NAME=comp_0]',
    '/network/SupAxoaxonic/#[NAME=comp_0]',
    '/network/SupLTS/#[NAME=comp_0]',
    '/network/SpinyStellate/#[NAME=comp_0]',
    '/network/TuftedIB/#[NAME=comp_0]',
    '/network/TuftedRS/#[NAME=comp_0]',
    '/network/DeepBasket/#[NAME=comp_0]',
    '/network/DeepAxoaxonic/#[NAME=comp_0]',
    '/network/DeepLTS/#[NAME=comp_0]',
    '/network/NontuftedRS/#[NAME=comp_0]',
    '/network/TCR/#[NAME=comp_0]',
    '/network/nRT/#[NAME=comp_0]',
    ]

spikeRecordingSettings.fractions = 1.0
spikeRecordingSettings.fields = {
    'Vm': 'Event',
    }

#---------------------------------------------------------------------
# Logging
#---------------------------------------------------------------------
import os
from datetime import datetime
import logging

timestamp = datetime.now()
mypid = os.getpid()
data_dir = os.path.join('data', timestamp.strftime('%Y_%m_%d'))

if not os.access(data_dir, os.F_OK):
    os.mkdir(data_dir)

filename_suffix = '_%s_%d' % (timestamp.strftime('%Y%m%d_%H%M%S'), mypid)

def handleError(self, record):
    raise

LOG_FILENAME = os.path.join(data_dir, 'traub2005%s.log' % (filename_suffix))
LOG_LEVEL = logging.DEBUG
logging.Handler.handleError = handleError
logging.basicConfig(filename=LOG_FILENAME, level=LOG_LEVEL, format='%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s: %(message)s', filemode='w')
# logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s: %(message)s', filemode='w')

logger = logging.getLogger('traub2005')

import os
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('../../../python')
import moose
############################################################
# Initialize library and other containers.
############################################################
library = moose.Neutral(modelSettings.libpath)


# 
# config.py ends here
