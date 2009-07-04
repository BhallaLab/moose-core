# filetypeutil.py --- 
# 
# Filename: filetypeutil.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Thu Jun 25 00:35:30 2009 (+0530)
# Version: 
# Last-Updated: Sat Jul  4 18:49:10 2009 (+0530)
#           By: subhasis ray
#     Update #: 139
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 
# 
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 

# Code:
import re

class FileTypeChecker:
    """This class provides a very simple file type checking based on
    regular expressions. 

    It first assumes that if a file has extension '.g' then it must
    be a genesis file. Now it looks for the statement 'include
    kkit'. If that is present, the file is assumed to be a kinetikit
    file. 

    If the file extension is '.xml', then it looks for the string
    '<sbml' in the file contents. If found, this is assumed to be an
    sbml file. On the other hand, if it has the string '<neuroml', it
    is assumed to be a neuroml file. 

    This is not robust - but given a correct file, it will correctly
    recognise its type.
    """
    type_sbml = 'SBML'
    type_kkit = 'KKIT'
    type_neuroml = 'NEUROML'
    type_genesis = 'GENESIS'
    def __init__(self, fileName):
	if fileName.endswith('.g') or fileName.endswith('.G'):
	    fileType = self.getGenesisFileType(fileName)
	elif fileName.endswith('.xml') or fileName.endswith('.XML'):
	    fileType = self.getXMLFileType(fileName)
	else:
	    fileType = None
	self.fileType_ = fileType

    def fileType(self):
	return self.fileType_

    def getGenesisFileType(self, fileName):
	fileObj = open(fileName, 'r')
	kkit_pattern = 'include *kkit'
	ret = FileTypeChecker.type_genesis
	in_comment = False
	for line in fileObj:
	    line = line.strip()
	    if line.find('//') == 0: # skip comments
		continue
	    if line.find('/*') >= 0: # block comment
		in_comment = True
	    if in_comment:
		pos = line.find('*/')
		if  pos >= 0: # check for end of block comment
		    in_comment = False 
		    line = line[pos+2:]  # check only rest of the line
		else:
		    continue
	    if in_comment:
		continue
	    if re.search(kkit_pattern, line):
		ret = FileTypeChecker.type_kkit
		break
	fileObj.close()
	return ret

    def getXMLFileType(self, fileName):
	ret = None
	fileObj = open(fileName, 'r')
	sbml_pattern = '<sbml'
	neuroml_pattern = '<neuroml'
	in_comment = False
	for line in fileObj:
	    line = line.strip()
	    if line.startswith('<!--'):
		in_comment = True
	    if in_comment:
		pos = line.find('-->')
		if pos >= 0:
		    line = line[pos + 3:]
		    in_comment = False
	    if in_comment:
		continue
	    if re.search(sbml_pattern, line):
		ret = FileTypeChecker.type_sbml
		break
	    elif re.search(neuroml_pattern, line):
		ret = FileTypeChecker.type_neuroml
		break
	return ret

# The following is test code
if __name__ == '__main__':
    kkit_fname = 'kkit_tmp.g'
    kkit = open(kkit_fname, 'w')
    kkit.write(
	"""//genesis
/* this is a block comment in a single line include kkit*/
""")
    kkit.close()
    fcheck = FileTypeChecker(kkit_fname)
    assert(fcheck.fileType() == FileTypeChecker.type_genesis)

    kkit_fname = 'kkit_tmp.g'
    kkit = open(kkit_fname, 'w')
    kkit.write(
	"""//genesis
/* this is a block comment in a multiple lines
 include kkit
ends here
*/
""")
    kkit.close()
    fcheck = FileTypeChecker(kkit_fname)
    assert(fcheck.fileType() == FileTypeChecker.type_genesis)
    kkit = open(kkit_fname, 'w')    
    kkit.write(
	"""//genesis
/* this is a block comment in a multiple lines
 include kkit
ends here
*/
include kkit
""")
    kkit.close()
    fcheck = FileTypeChecker(kkit_fname)
    assert(fcheck.fileType() == FileTypeChecker.type_kkit)
    kkit = open(kkit_fname, 'w')
    kkit.write(
	"""//include kkit
/* this is a block comment in a multiple lines
 include kkit
ends here
*/
""")

    kkit.close()
    fcheck = FileTypeChecker(kkit_fname)
    assert(fcheck.fileType() == FileTypeChecker.type_genesis)
    kkit = open(kkit_fname, 'w')
    kkit.write(
	"""//include kkit
/* this is a block comment in a multiple lines
 include kkit
ends here
*/
include kkit
""")
    kkit.close()
    fcheck = FileTypeChecker(kkit_fname)
    assert(fcheck.fileType() == FileTypeChecker.type_kkit)
    
    print 'success with genesis file type'

    xml_fname = 'test.xml'
    xml_file = open(xml_fname, 'w')
    xml_file.write("""
<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
""")
    xml_file.close()
    fcheck = FileTypeChecker(xml_fname)
    assert(fcheck.fileType() == FileTypeChecker.type_sbml)
    
    xml_file = open(xml_fname, 'w')
    xml_file.write("""
<?xml version="1.0" encoding="UTF-8"?>
<!--<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">-->
""")
    xml_file.close()
    fcheck = FileTypeChecker(xml_fname)
    assert(fcheck.fileType() == None)
    
    xml_file = open(xml_fname, 'w')
    xml_file.write("""
<?xml version="1.0" encoding="UTF-8"?>
<!--
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
-->
""")
    xml_file.close()
    fcheck = FileTypeChecker(xml_fname)
    assert(fcheck.fileType() == None)
		
    xml_file = open(xml_fname, 'w')
    xml_file.write("""
<?xml version="1.0" encoding="UTF-8"?>
<!--<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">-->
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
""")
    xml_file.close()
    fcheck = FileTypeChecker(xml_fname)
    assert(fcheck.fileType() == FileTypeChecker.type_sbml)
    print 'SBML check success'

    xml_file = open(xml_fname, 'w')
    xml_file.write('<?xml version="1.0" encoding="UTF-8"?>\n<neuroml lengthUnits="micron" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://morphml.org/neuroml/schema" xmlns:meta="http://morphml.org/metadata/schema" xmlns:mml="http://morphml.org/morphml/schema" xsi:schemaLocation="http://morphml.org/neuroml/schema /home/subha/src/neuroml/Schemata/Level1/NeuroML_Level1_v1.7.3.xsd">')
    xml_file.close()
    fcheck = FileTypeChecker(xml_fname)
    assert(fcheck.fileType() == FileTypeChecker.type_neuroml)
    xml_file = open(xml_fname, 'w')
    xml_file.write("""
<?xml version="1.0" encoding="UTF-8"?>
<!-- This is a test block comment

he he he
-->
<neuroml lengthUnits="micron" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://morphml.org/neuroml/schema" xmlns:meta="http://morphml.org/metadata/schema" xmlns:mml="http://morphml.org/morphml/schema" xsi:schemaLocation="http://morphml.org/neuroml/schema /home/subha/src/neuroml/Schemata/Level1/NeuroML_Level1_v1.7.3.xsd">
""")
    xml_file.close()
    fcheck = FileTypeChecker(xml_fname)
    assert(fcheck.fileType() == FileTypeChecker.type_neuroml)    
    xml_file = open(xml_fname, 'w')
    xml_file.write("""
<?xml version="1.0" encoding="UTF-8"?>
<!-- another comment to confuse the checker.
<neuroml lengthUnits="micron" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://morphml.org/neuroml/schema" xmlns:meta="http://morphml.org/metadata/schema" xmlns:mml="http://morphml.org/morphml/schema" xsi:schemaLocation="http://morphml.org/neuroml/schema /home/subha/src/neuroml/Schemata/Level1/NeuroML_Level1_v1.7.3.xsd">

evil laughter ....-->
""")
    xml_file.close()
    fcheck = FileTypeChecker(xml_fname)
    assert(fcheck.fileType() == None)	    
    xml_file = open(xml_fname, 'w')
    xml_file.write("""
<?xml version="1.0" encoding="UTF-8"?>
<!-- another comment to confuse the checker.
<neuroml lengthUnits="micron" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://morphml.org/neuroml/schema" xmlns:meta="http://morphml.org/metadata/schema" xmlns:mml="http://morphml.org/morphml/schema" xsi:schemaLocation="http://morphml.org/neuroml/schema /home/subha/src/neuroml/Schemata/Level1/NeuroML_Level1_v1.7.3.xsd">

evil laughter ....-->
<neuroml lengthUnits="micron" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://morphml.org/neuroml/schema" xmlns:meta="http://morphml.org/metadata/schema" xmlns:mml="http://morphml.org/morphml/schema" xsi:schemaLocation="http://morphml.org/neuroml/schema /home/subha/src/neuroml/Schemata/Level1/NeuroML_Level1_v1.7.3.xsd">
""")
    xml_file.close()
    fcheck = FileTypeChecker(xml_fname)
    assert(fcheck.fileType() == FileTypeChecker.type_neuroml)	    

    print 'NeuroML check success'

# 
# filetypeutil.py ends here
