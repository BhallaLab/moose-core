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
        schemaFile = "./moose_xml/NeuroML_v2beta1.xsd"
        with open(schemaFile, "r") as f :
            nmlSchemaTxt = etree.XML(f.read())

        nmlSchema = etree.XMLSchema(nmlSchemaTxt)
        nmlParser = etree.XMLParser(schema = nmlSchema)

        with open(models.nml, "r") as nmlFile :
            nmlText = nmlFile.read()
            elemDict['nml'] = models.nml
        try :
            rootElementOfNml = etree.fromstring(nmlText, nmlParser)
        except :
            logger.warn("Validation with schema {0} failed. Continuing \
                nonetheless.".format(schemaFile))
            rootElementOfNml = etree.parse(models.nml)
            elemDict['nmlXml'] = rootElementOfNml

    if models.sbml :
        # TODO : parse it in the same way as we are parsing neuroML
        rootElementOfSbml = etree.parse(models.sbml)
        elemDict['sbmlXml'] = rootElementOfSbml
        elemDict['sbml'] = models.sbml
    return elemDict

