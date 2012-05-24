# config.py --- 
# 
# Filename: config.py
# Description: 
# Author: 
# Maintainer: 
# Created: Wed May 23 11:31:40 2012 (+0530)
# Version: 
# Last-Updated: Thu May 24 16:56:27 2012 (+0530)
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



# 
# config.py ends here
