#!/usr/bin/env python
## -*- coding: utf-8 -*-



# from xml.dom import minidom				#required libraries
# import os

# root = minidom.Document()				#to open a new document

# xml = root.createElement('root') 		#to create xml file

# root.appendChild(xml)



# xml_str = root.toprettyxml(indent="\t")	#to give the indentation

			#path to saved file

# with open(save_path_file, "w") as f:
# 	f.write

# save_path_file.encode('UTF-8')

from lxml import etree
import sys



#path to saved file
save_path_file =  open("./outxml.xml", 'w')
	# with open(save_path_file,"w") as f:
	# 	f.write
#create XML
sys.stdout=save_path_file
root = etree.Element('root')
root.append(etree.Element('child'))
#another child with text
child = etree.Element('child')
child.text = 'some text'
root.append(child)



#pretty string
s = etree.tostring(root, pretty_print = True)
print s



