# digestbuiltindocs.py --- 
# 
# Filename: digestbuiltindocs.py
# Description: 
# Author: 
# Maintainer: 
# Created: Thu Oct  4 11:33:31 2012 (+0530)
# Version: 
# Last-Updated: Thu Oct  4 16:53:43 2012 (+0530)
#           By: subha
#     Update #: 234
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
import inspect
from datetime import datetime

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


def extract_ftype_doc(cinfo, finfotype, docio, indent, lmarker):
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
	docio.write('\n%s%s%s:\t/%s/ ::\n' % (indent, lmarker, fid.name, dtype))
	doc = fid.docs.replace('\n', '\n%s' % (indent2))
	docio.write(indent2+doc)

def extract_class_doc(name, docio, indent, lmarker):
    """Extract documentation for Cinfo object at path
    
    Parameters
    ----------
    name: str
	path of the class.

    docio: StringIO
	output object to write the documentation into.
    """
    cinfo = moose.Cinfo('/classes/%s' % (name))
    docs = cinfo.docs.strip()
    if docs:	
	docs = docs.replace('\n', '\n'+indent)
	docio.write('\n%s%s' % (indent, docs))
    for finfotype, readablename in moose.finfotypes:
        docio.write('\n\n%s%s*%s*' % (indent, lmarker, readablename.capitalize()))
	extract_ftype_doc(cinfo, finfotype, docio, indent + ' ' * len(lmarker), lmarker)

def extract_all_class_doc(docio, hmarker, lmarker):
    for cinfo in moose.element('/classes').children:
	docio.write('\n\n%s%s' % (hmarker, cinfo[0].name))
	extract_class_doc(cinfo.name, docio, ' ' * len(hmarker), lmarker)

def extract_all_func_doc(docio, hmarker):
    for fname, fdef in inspect.getmembers(moose, inspect.isbuiltin) + inspect.getmembers(moose, inspect.isfunction):
	docio.write('\n\n%s%s\n' % (hmarker, fname))
	doc = inspect.getdoc(fdef)
	doc = doc.replace('\n', '\n'+' '*len(hmarker))
	docio.write(' '*len(hmarker)+doc)

if __name__ == '__main__':
    if len(sys.argv)  > 1:
	docio = open(sys.argv[1], 'w')
    else:
	docio = cStringIO.StringIO()
    docio.write(config['heading1'] + 'MOOSE Classes\n')
    ts = datetime.now()
    extract_all_class_doc(docio, config['heading2'], config['ulist1'])
    docio.write('\n\n'+config['heading1'] + 'MOOSE Functions\n')
    extract_all_func_doc(docio, config['heading2'])
    docio.write('/automatically extracted on %s/\n' % (ts.isoformat()))
    if isinstance(docio, cStringIO.OutputType):
	print docio.getvalue()
    else:
	docio.close()

# 
# digestbuiltindocs.py ends here
