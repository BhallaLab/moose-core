#!/usr/bin/env python

# Filename       : nml2moose.py 
# Created on     : 2013-09-09
# Author         : Dilawar Singh
# Email          : dilawars@ncbs.res.in
#
# Description    : parse NML2 models and build moose.
#
# Logs           :



import logging
import os
import debug.logger 
from IPython import embed

logger = logging.getLogger('multiscale')

try:
    from lxml import etree
    logger.info("Running with lxml.etree")
except ImportError:
    try:
        # Python 2.5
        import xml.etree.cElementTree as etree
        logger.info("Running with cElementTree")
    except ImportError:
        try:
            # Python 2.5
            import xml.etree.ElementTree as etree
            logger.info("Running with xml.etree.ElementTree")
        except ImportError:
            try:
              # normal cElementTree install
              import cElementTree as etree
              logger.info("Running with cElementTree")
            except ImportError:
                try:
                    # normal ElementTree install
                    import elementtree.ElementTree as etree
                    logger.info("Running with elementtree.ElementTree")
                except ImportError:
                    logger.error("Failed to import ElementTree in usual places")
                    os._exit(1)


# class to read NML2
import moose 
import moose_xml.get_elements as _xml
class NML2Reader(object) :

  def __init__(self):
    self.lunit = 1e-6 # micron is the default lenght unit 
    self.xmlDoc = None
    self.nmlDoc = None # NML2 Document
    self.nmlToMoose = dict() # NeuroML object to moose object
    self.includes = dict()   # Included files in nml
    self.lib = moose.Neutral('/library')


  def read(self, nmlFilePath) :
    logger.info("Parsing NML2 file.")
    self.nmlDoc = etree.parse(nmlFilePath)
    if self.nmlDoc :
      includesElem =  _xml.getElementNM(self.nmlDoc, ["neuroml", "include"])
      self.importIncludes(nmlFilePath, includesElem)
    else :
      logger.error("Could not parse {0}".format(nmlFilePath))
      sys.exit(0)
    

  def importIncludes(self, rootFilePath, includeElem) :
    logger.info("Adding include path to moose")
    for iE in includeElem :
      paths = []
      p = iE.values()[0]
      paths.append(p)
      paths.append(os.path.join(os.path.dirname(rootFilePath), p))
      # We should be able to find at least one path 
      for tryP in paths :
        if os.path.exists(tryP) :
          logger.info("Parse the include path and append it to XML")
          # parse it and append to self.doc 
        else : pass 

