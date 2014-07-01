# create_rest_doc.py --- 
# 
# Filename: create_rest_doc.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Mon Jun 30 21:35:07 2014 (+0530)
# Version: 
# Last-Updated: 
#           By: 
#     Update #: 0
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
"""Dump reStructuredText of moose class docs"""
import sys
import cStringIO
import re
import inspect
from datetime import datetime
import moose

# We assume any non-word-constituent character in the start of C++
# type name to be due to the name-mangling done by compiler on
# templated or user defined types.
type_mangling_regex = re.compile('^[^a-zA-Z]+')

finfotypes = dict(moose.finfotypes)

def extract_finfo_doc(cinfo, finfotype, docio, indent='   '):
    """Extract field documentation for all fields of type `finfotype`
    in class `cinfo` into `docio`.

    Parameters
    ----------
    cinfo: moose.Cinfo 
        class info object in MOOSE.
    
    ftype: str
        finfo type (valueFinfo/srcFinfo/destFinfo/lookupFinfo/sharedFinfo

    docio: StringIO
        IO object to write the documentation into
    """
    print 'FINDENT: {0}'.format(len(indent))
    data = []
    try:
        finfo = moose.element('%s/%s' % (cinfo.path, finfotype)).vec
    except ValueError:
        return
    for field_element in finfo:
        dtype = type_mangling_regex.sub('', field_element.type)
        if finfotype.startswith('dest'):
            name = '.. py:method:: {0}'.format(field_element.fieldName)
            dtype = ''
        else:
            name = '.. py:attribute:: {0}'.format(field_element.fieldName)
            dtype = '{0}'.format(dtype)
        doc = field_element.docs
        docio.write('{0}{1}\n\n{0}   {2} (*{3}*) {4}\n\n'.format(indent, name, dtype, finfotypes[finfotype], doc))

def extract_class_doc(name, docio, indent='   '):
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
    docio.write('{0}.. py:class:: {1}\n\n'.format(indent, cinfo.name))
    if docs:                    
        docs = docs.split('\n')
        # We need these checks to avoid mis-processing `:` within
        # description of the class
        name_done = False
        author_done = False
        descr_done = False
        for doc in docs:
            if not doc:
                continue
            if not (name_done and author_done and descr_done):
                pos = doc.find(':')         
                field = doc[:pos].strip()
                content = doc[pos+1:]
            else:
                content = doc
            if field.lower() == 'name':
                name_done = True
                continue
            elif field.lower() == 'author':
                author_done = True
                continue          
            elif field.lower() == 'description':
                descr_done = True
                content = content.lstrip()                
            docio.write('{0}   {1}\n\n'.format(indent, content))
    for finfotype in finfotypes.keys():
	extract_finfo_doc(cinfo, finfotype, docio, indent + '   ')

def extract_all_class_doc(docio, indent='   '):
    print 'INDENT "{0}"'.format(len(indent))
    for cinfo in moose.element('/classes').children:
	extract_class_doc(cinfo.name, docio, indent=indent)

def extract_all_func_doc(docio, indent='   '):
    for fname, fdef in (inspect.getmembers(moose, inspect.isbuiltin) +
                        inspect.getmembers(moose, inspect.isfunction)):
	docio.write('\n{}.. py:func:: {}\n'.format(indent, fname))
	doc = inspect.getdoc(fdef)
	doc = doc.split('\n')
	drop = []
	for i in range(len(doc)):
	    di = doc[i]
	    doc[i] = di.strip()
	    hyphen_count = di.count('-')
	    if hyphen_count > 0 and hyphen_count == len(di) and i > 0:
		drop.append(i)
		doc[i-1] = indent + doc[i-1]
	for i in range(len(doc)):
	    if i not in drop:
		docio.write(doc[i] + '\n\n')


if __name__ == '__main__':
    if len(sys.argv)  > 1:
	docio = open(sys.argv[1], 'w')
    else:
	docio = cStringIO.StringIO()
    
    ts = datetime.now()
    docio.write('.. Documentation for all MOOSE classes and functions\n')
    docio.write('.. As visible in the Python module\n')
    docio.write(ts.strftime('.. Auto-generated on %B %d, %Y\n'))
    
    docio.write('''
==================
MOOSE Classes
==================
''')
    extract_all_class_doc(docio, indent='')
#     docio.write('''
# =================
# MOOSE Functions
# =================
# ''')
#     extract_all_func_doc(docio, indent='')
    if isinstance(docio, cStringIO.OutputType):
	print docio.getvalue()
    else:
	docio.close()



# 
# create_rest_doc.py ends here
