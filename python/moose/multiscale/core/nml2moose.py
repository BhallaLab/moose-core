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

from lxml import etree
from lxml import objectify 

# class to read NML2
import moose 
import moose_xml.get_elements as _xml

class NML2Reader(object) :

  def __init__(self):
    self.lunit = 1e-6 # micron is the default lenght unit 
    self.xmlObj = None
    self.nmlObj = None # NML2 Document
    self.nmlToMoose = dict() # NeuroML object to moose object
    self.includes = dict()   # Included files in nml
    self.lib = moose.Neutral('/library')


  def read(self, nmlFilePath) :
    logger.info("Parsing NML2 file.")
    self.nmlObj = objectify.parse(nmlFilePath)
    if self.nmlObj :
      self.nmlObjRoot = self.nmlObj.getroot()
      objPath = objectify.ObjectPath(".include")
      includesElem = objPath(self.nmlObj.getroot())
      self.importIncludes(nmlFilePath, includesElem)
      # Import ion channels 
      self.importIonChannels()
      # Add concentration models
      self.createConcentrationModels()
      # and build moose now 
      self.buildMoose()
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
          logger.info("Parse the include path {0} and append it to XML".format(tryP))
          self.appendIncludeToDoc(tryP)
          # parse it and append to self.doc 
        else : pass 

  def appendIncludeToDoc(self, includedXMLPath) :
    """
    Store all included files in a dictionary. When parsing the main XML model,
    we should search the included file to find the models.

    """
    includePathObj = objectify.parse(includedXMLPath)
    assert os.path.exists(includedXMLPath)
    includeElemObj = objectify.parse(includedXMLPath)
    self.includes[includedXMLPath] = includeElemObj
  
  def createConcentrationModels(self) :
    logger.info("Create concentration models.")

    

  def buildMoose(self) :
    """
    Now build moose.
    """
    logger.info("Build moose now.")
    
