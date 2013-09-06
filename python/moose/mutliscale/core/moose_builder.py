#!/usr/bin/env python

# Filename       : moose_builder.py 
# Created on     : Fri 06 Sep 2013 03:23:57 PM IST
# Author         : Dilawar Singh
# Email          : dilawars@ncbs.res.in
#
# Description    :
#   Maps the given XML element onto moose's internal data-structures.
#
# Logs           :

import sys
import debug.debug as debug

sys.path.append("../../../python/")
import moose 

# Let's build a primary model of given neuron. We'll populate it after building
# models of each XML model given in XML file.
# TODO : This is left here to draw attention of programmar.
model = moose.Neutral('/model')

def buildMooseObjects(listOfXMLElements) :
    """
    Given parsed XML of one or mode models, build moose storehouse.
    """
    assert(len(listOfXMLElements) > 0)
    nmlModel = moose.Neutral('/nmlModel')


