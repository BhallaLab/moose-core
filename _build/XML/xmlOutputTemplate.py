
from lxml import builder
from lxml import etree
import lxml
import sys

#path to saved file
save_path_file =  open("./outxml.xml", 'w')

#create XML
sys.stdout=save_path_file

#attributes are printed in reverse order
SBML 					= etree.Element("sbml",xmlns="http://www.sbml.org/sbml/level3/version1/core", level="3", version="1") 
MODEL 					= etree.SubElement(SBML,"model",metaid="_case00001", id="case00001", name="case00001", timeUnits="time")
LISTOFUNITDEFINITIONS 	= etree.SubElement(MODEL, "listOfUnitDefinitions")
UNITDEFINITION 			= etree.SubElement(LISTOFUNITDEFINITIONS, "unitDefinition", id="volume")
LISTOFUNITS 			= etree.SubElement(UNITDEFINITION, "listOfUnits")
UNIT 					= etree.SubElement(LISTOFUNITS, "unit", kind="litre", exponent="1", scale="0", multiplier="1")

#pretty string
s = etree.tostring(SBML, pretty_print = True)
print s
print SBML.attrib



