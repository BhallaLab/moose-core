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

E= lxml.builder.ElementMaker()
Sbml=E.sbml()
Model=E.model
Listofunitdef=E.listOfUnitDefinitions
UnitDef=E.unitDefinition
Listofunits=E.listOfUnitDefinitions
Unit=E.unit

# Sbml(xmlns="http://www.sbml.org/sbml/level3/version1/core", level="3", version="1")

the_doc = (
		Sbml( 
		Model(
		Listofunitdef(	
		UnitDef(
		Listofunits(
		Unit(kind="litre", exponent="1", scale="0", multiplier="1")
									)	
								)		
							)	
						)
				)
			) 



#pretty string
s = etree.tostring(the_doc, pretty_print = True)
print s

# print __name__

