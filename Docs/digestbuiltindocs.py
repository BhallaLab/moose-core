# digestbuiltindocs.py --- 
# 
# Filename: digestbuiltindocs.py
# Description: 
# Author: 
# Maintainer: 
# Created: Thu Oct  4 11:33:31 2012 (+0530)
# Version: 
# Last-Updated: Thu Oct  4 15:15:34 2012 (+0530)
#           By: subha
#     Update #: 144
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Script to extract the builtin documentation of MOOSE and save it as
# org text file.
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
import sys
import cStringIO
import re
sys.path.append('../python')
import moose

# We assume any non-word-constituent character in the start of C++
# type name to be due to the name-mangling done by compiler on
# templated or user defined types.
type_mangling_regex = re.compile('^[^a-zA-Z]+')

config = {'heading1': '* ',
	  'heading2': '** ',
	  'heading3': '*** ',
	  'ulist1': '+ ',
	  'ulist2': '- ',
	  'nlist': 1}


def extract_ftype_doc(cinfo, finfotype, docio, indent):
    """Extract field documentation for all fields of type `finfotype`
    in class `cinfo` into `docio`.

    Parameters
    ----------
    cinfo: moose.Cinfo 
	class info object in MOOSE.
    
    finfotype: str
	finfo type (valueFinfo/srcFinfo/destFinfo/lookupFinfo/sharedFinfo

    docio: StringIO
	IO object to write the documentation into
    """
    indent2 = indent + ' ' * len(config['ulist2'])
    numfinfo = moose.getField(cinfo, 'num_'+finfotype, 'unsigned')
    finfo = moose.ematrix('%s/%s' % (cinfo.path, finfotype))
    for ii in range(numfinfo):
	fid = moose.melement(finfo, 0, ii, 0)
	if finfotype == 'destFinfo' and (fid.name.startswith('get_') or
					 fid.name.startswith('set_')):
	    continue
	dtype = type_mangling_regex.sub('', fid.type)
	docio.write('\n%s%s%s/%s/ ::\n' % (indent, config['ulist2'], fid.name, dtype))
	doc = fid.docs.replace('\n', '\n%s' % (indent2))
	docio.write(indent2+doc)

def extract_class_doc(name, docio, indent):
    """Extract documentation for Cinfo object at path
    
    Parameters
    ----------
    name: str
	path of the class.

    docio: StringIO
	output object to write the documentation into.
    """
    cinfo = moose.Cinfo('/classes/%s' % (name))
    docs = cinfo.docs.replace('\n', '\n'+indent)
    if docs.strip():
	docio.write('\n%s%s' % (indent, docs))
    for finfotype, readablename in moose.finfotypes:
        docio.write('\n\n%s%s*%s*' % (indent, config['ulist1'], readablename.capitalize()))
	extract_ftype_doc(cinfo, finfotype, docio, indent + ' ' * len(config['ulist1']))

def extract_all_class_doc(docio):
    for cinfo in moose.element('/classes').children:
	docio.write('\n\n%s%s' % (config['heading1'], cinfo[0].name))
	extract_class_doc(cinfo.name, docio, ' ' * len(config['heading1']))

if __name__ == '__main__':
    docio = cStringIO.StringIO()
    extract_all_class_doc(docio)
    print docio.getvalue()

# 
# digestbuiltindocs.py ends here
