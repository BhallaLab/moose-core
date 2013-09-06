#!/usr/bin/env python

# Filename       : get_elements.py 
# Created on     : 2013
# Author         : Dilawar Singh
# Email          : dilawars@ncbs.res.in
#
# Description    : Functions to get elements from XML tree.
#
# Logs           :

import debug.debug as debug 

namespace = 'http://www.neuroml.org/schema/neuroml2'

def getElement(xmlElem, path) :
  """ 
  Get elements as described in XPATH path.
  """
  return xmlElem.xpath(path)


def getElementNM(xmlElem, listOfNames) :
  """
  Given a list of words e.g. [a, b, c] construct a path a/b/c with namespace and
  query the xmlElem.
  """
  global namespace
  path = ''
  for p in listOfNames :
    path += ('/a:' + p )
  try :
    return xmlElem.xpath(path, namespaces = {'a' : namespace})
  except :
    debug.printDebug("WARN", "Failed to search XML for XPATH {0}".format(path))

def getElementIgnoreNM(xmlElem, name) :
  """
  Ignore namespace. But this function only takes name and not XPATH
  """
  return xmlElem.xpath("//*[local-name()='"+name+"']")


