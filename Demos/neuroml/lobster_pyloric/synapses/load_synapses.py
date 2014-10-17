#!/usr/bin/env python
# -*- coding: utf-8 -*-

import moose
from GluSyn_STG import GluSyn_STG
from AchSyn_STG import AchSyn_STG

def load_synapses():
    syn = GluSyn_STG("/library/DoubExpSyn_Glu")
    synhandler = moose.SimpleSynHandler("/library/DoubExpSyn_Glu_handler")
    ## connect the SimpleSynHandler to the SynChan (double exp)
    moose.connect( synhandler, 'activationOut', syn, 'activation' )

    syn = AchSyn_STG("/library/DoubExpSyn_Ach")
    synhandler = moose.SimpleSynHandler("/library/DoubExpSyn_Ach_handler")
    ## connect the SimpleSynHandler to the SynChan (double exp)
    moose.connect( synhandler, 'activationOut', syn, 'activation' )
