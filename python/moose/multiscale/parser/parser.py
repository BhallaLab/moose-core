# This file contains functions to parse neuroML and sbml.

import debug.debug as debug 
import logging 
import os

logger = logging.getLogger('multiscale')
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


def parseModels(models) :

    """
    Parses given xml models. We can pass either one or two models; one described in
    neuroML and the other in sbml. 
    
    Notes: Document is properly. See 

      http://www.biomedcentral.com/1752-0509/7/88/abstract

    sent by Aditya Girla. It a online composition tool for SBML. In its
    references, some other tools are mentioned.

    Args :

    Raises :

    Return 
    return a list of elementTree of given models.

    """
    elemDict = dict()
    if models.nml :
        # Get the schema 
        with open(models.nml, "r") as nmlFile :
            elemDict['nml'] = models.nml
        
    if models.sbml :
        elemDict['sbml'] = models.sbml
    return elemDict

