#!/usr/bin/env python
## -*- coding: utf-8 -*-


# #required libraries
# from xml.dom import minidom				
# import os

# #open a new document
# root = minidom.Document()				
# #create xml file
# xml = root.createElement('root') 		

# root.appendChild(xml)
# #to give the indentation
# xml_str = root.toprettyxml(indent="\t")	
# # path to saved file
# with open(save_path_file, "w") as f:
# 	f.write

# save_path_file.encode('UTF-8')			



# from lxml import etree as ET
# from lxml import builder
# from lxml.builder import E
# import lxml.etree
# import lxml
from lxml import builder
from lxml import etree
import lxml
import sys



#path to saved file
save_path_file =  open("./outxml.xml", 'w')
# with open(save_path_file,"w") as f:
# 	f.write
#create XML
sys.stdout=save_path_file


# root = etree.Element('root')
# root.append(etree.Element('child'))
# #another child with text
# child = etree.Element('child')
# grandchild = etree.Element('grandchild')
# child.append(etree.Element('grandchild'))
# grandchild.text= "some more text"
# # child.append(grandchild)
# child.text = 'some text'
# root.append(child)
# doc = etree.SubElement(root, "doc")
# # child.text.encode(UTF-8)

# E= lxml.builder.ElementMaker()
# Sbml=E.sbml
# Model=E.model
# Listofunitdef=E.listOfUnitDefinitions
# UnitDef=E.unitDefinition
# Listofunits=E.listOfUnitDefinitions
# Unit=E.unit

# # Sbml()

# the_doc = (
# 		Sbml(
# 			Model(
# 				Listofunitdef(
# 					UnitDef(
# 						Listofunits(
# 							Unit(kind="litre", exponent="1", scale="0", multiplier="1")
# 							)
# 						)
# 					)
# 				)
# 			)
# 		)


#prints the attributes in reveerse order
SBML = etree.Element("sbml",xmlns="http://www.sbml.org/sbml/level3/version1/core", level="3", version="1") 
MODEL = etree.SubElement(SBML,"model",metaid="_case00001", id="case00001", name="case00001", timeUnits="time")
LISTOFUNITDEFINITIONS = etree.SubElement(MODEL, "listOfUnitDefinitions")
UNITDEFINITION = etree.SubElement(LISTOFUNITDEFINITIONS, "unitDefinition", id="volume")
LISTOFUNITS = etree.SubElement(UNITDEFINITION, "listOfUnits")
UNIT = etree.SubElement(LISTOFUNITS, "unit", kind="litre", exponent="1", scale="0", multiplier="1")







# the_doc.append(Unit('another value again', name='hithere'))

#pretty string
s = etree.tostring(SBML, pretty_print = True)
print s

# print __name__

