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
import xml.get_elements as xml 
import core.neuroml_to_moose as nml2moose

try:
    from lxml import etree
    debug.printDebug("DEBUG", "running with lxml.etree")
except ImportError:
    try:
        # Python 2.5
        import xml.etree.cElementTree as etree
        debug.printDebug("DEBUG", "running with cElementTree")
    except ImportError:
        try:
            # Python 2.5
            import xml.etree.ElementTree as etree
            debug.printDebug("DEBUG", "running with ElementTree")
        except ImportError:
            try:
              # normal cElementTree install
              import cElementTree as etree
              debug.printDebug("DEBUG", "running with cElementTree")
            except ImportError:
                try:
                    # normal ElementTree install
                    import elementtree.ElementTree as etree
                    debug.printDebug("DEBUG", "running with ElementTree")
                except ImportError:
                    debug.prefix("FATAL", "Failed to import ElementTree")
                    os._exit(1)


sys.path.append("../../../python/")
import moose 

# This is a primray model of a given neuron.
model = moose.Neutral('/model')

def buildMooseObjects(dictOfXMLElements) :
    """
    Given parsed XML of one or mode models, build moose storehouse.
    """
    assert(len(dictOfXMLElements) > 0)
    nmlF = dictOfXMLElements['nml']
    if nmlF :
      objNmlReader = nml2moose.NML2Reader()
      objNmlReader.read(nmlF)
      nmlModel = moose.Neutral('/nmlModel')
      # get the morphology of neuron 
      #neuromlPathFind = etree.ETXPath("//{"+namespace+"}neuroml")
      #print neuromlPathFind(nmlXml)
      cellElem = xml.getElementIgnoreNM(nmlXml, "cell")
      print cellElem
      print xml.getElementNM(nmlXml, ["neuroml"])

