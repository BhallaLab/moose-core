# digestbuiltindocs.py --- 
# 
# Filename: digestbuiltindocs.py
# Description: 
# Author: 
# Maintainer: 
# Created: Thu Oct  4 11:33:31 2012 (+0530)
# Version: 
# Last-Updated: Mon Dec  31 17:08:00 2012 (+0530)
#           By: niraj
#     Update #: 249
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
import moose

# We assume any non-word-constituent character in the start of C++
# type name to be due to the name-mangling done by compiler on
# templated or user defined types.
type_mangling_regex = re.compile('^[^a-zA-Z]+')

config = {'heading1': '# ',
	  'heading2': '## ',
	  'heading3': '### ',
	  'heading4': '#### ',
	  'ulist1': '+ ',
	  'ulist2': '- ',
	  'nlist': 1}


def extract_ftype_doc(cinfo, finfotype, docio):
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
    numfinfo = moose.getField(cinfo, 'num_'+finfotype, 'unsigned')
    finfo = moose.ematrix('%s/%s' % (cinfo.path, finfotype))
    data = []
    inf = float( 'Inf' )
    max_width_name = -inf
    max_width_type = -inf
    for ii in range(numfinfo):
	fid = moose.melement(finfo, 0, ii, 0)
	
	if finfotype == 'destFinfo' and (fid.name.startswith('get_') or
					 fid.name.startswith('set_')):
	    continue
	
	dtype = type_mangling_regex.sub('', fid.type)
	name = '**`{0}`**'.format( fid.name )
	dtype = '`{0}`'.format( dtype )
	doc = fid.docs
	
	data.append( [ name, dtype, doc ] )
	
	if len( name ) > max_width_name:
	    max_width_name = len( name )
	
	if len( dtype ) > max_width_type:
	    max_width_type = len( dtype )
    
    if len( data ) == 0:
	return
    
    padding = 2
    format_string = '{{0: <{width}}}'
    format_strings = (
	format_string.format( width = max_width_name + padding ),
	format_string.format( width = max_width_type + padding ),
	'{0}'
    )
    col3_alignment = format_strings[0].format('') + format_strings[1].format('')
    for row in data:
	row[2] = row[2].replace( '\n', '\n' + col3_alignment )
    
    column_headings = ( 'Field', 'Type', 'Description' )
    docio.write( '\n' )
    for ( column_heading, format_string ) in zip( column_headings, format_strings ):
	docio.write( format_string.format( column_heading ) )
    docio.write( '\n' )
    for format_string in format_strings:
	docio.write( format_string.format( '----' ) )
    docio.write( '\n' )
    
    for row in data:
	for ( field, format_string ) in zip( row, format_strings ):
	    docio.write( format_string.format( field ) )
	docio.write( '\n' )

def extract_class_doc(name, docio, hmarker):
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
	docs = docs.split( '\n' )
	for doc in docs:
	    if not doc:
		continue
	    
	    pos = doc.find( ':' )
	    
	    field = doc[ :pos ]
	    content = doc[ pos: ]
	    if field.lower() == 'name':
		continue
	    
	    if pos == -1 or content.strip() == ':':
		line = '\n' + doc + '\n'
	    else:
		line = '\n**{field}**{content}\n'.format(
		    field=field,content=content)
	    
	    docio.write(line)
    
    for finfotype, readablename in moose.finfotypes:
        docio.write( '\n\n{0} {1}\n'.format(
	    hmarker, readablename.capitalize() + 's' ) )
	extract_ftype_doc(cinfo, finfotype, docio)

def extract_all_class_doc(docio, hmarker1, hmarker2):
    for cinfo in moose.element('/classes').children:
	docio.write('\n\n%s%s' % (hmarker1, cinfo[0].name))
	extract_class_doc(cinfo.name, docio, hmarker2)

def extract_all_func_doc(docio, hmarker1, hmarker2):
    for fname, fdef in inspect.getmembers(moose, inspect.isbuiltin) + inspect.getmembers(moose, inspect.isfunction):
	docio.write('\n\n%s%s\n' % (hmarker1, fname))
	doc = inspect.getdoc(fdef)
	doc = doc.split( '\n' )
	drop = []
	for i in range( len( doc ) ):
	    di = doc[i]
	    doc[i] = di.strip()
	    hyphen_count = di.count( '-' )
	    if hyphen_count > 0 and hyphen_count == len( di ) and i > 0:
		drop.append( i )
		doc[i-1] = hmarker2 + doc[i-1]
	for i in range( len( doc ) ):
	    if i not in drop:
		docio.write(doc[i] + '\n\n')

def generate_compact_index(docio, content, ncols, padding=2):
    """Generate a compact index, in the form of the table.
    
    Parameters
    ----------
    docio: StringIO
	output object to write the documentation into.
    
    content: list
	List of sections that need to be linked to.
    
    ncols: int
	Number of columns in the table.
    
    padding: int
	Separation between columns, in number of characters. This applies to the
	generated Markdown file, and not any HTML/PDF/etc. generated
	subsequently.
    """
    # Case-insensitive sort.
    content.sort(key=str.lower)
    
    # Generating formatted list of table entries. This consists of 2 things:
    #    - Titles of subsections, linking to the subsections' hrefs.
    #    - Entries in the table get grouped by letter of alphabet.
    #      Hence, for every group ('A', 'B', etc.) we put in a heading as well.
    entries = []
    for i in range( len( content ) ):
	current = content[i]
	previous = content[i-1] if i else None
	if not previous or previous[0].lower() != current[0].lower():
	    letter=current[0].upper()
	    entries.append('**{0}**'.format(letter))
	# Monospace text, linking to subsection.
	entries.append('[`{title}`](#{href})'.format(
	    title=current, href=current.lower()))
    
    # Finding out how wide each column should be in the Markdown.
    # We could just select a large number, but then some entries in the 'Type'
    # column can get unexcpectedly big (easily >100 chars), so should keep it
    # flexible if some of the entries become even bigger in the future.
    widths = []
    nrows = len(entries)/ncols
    if nrows * ncols != len(entries):
	nrows += 1
    for i in range(ncols):
	start = i*nrows
	end = start+nrows
	col_entries = entries[start:end]
	entry_widths = [ len(entry) for entry in col_entries ]
	widths.append( max(entry_widths) )
    
    format_string = ['{{0: <{0}}}'.format(width+padding) for width in widths]
    
    # Dumping entries in the Markdown text, row-by-row.
    # However, in the table, entries fill in alphabetically, column-by-column.
    for j in range(ncols):
	docio.write(format_string[j].format('----'))
    docio.write('\n')
    for i in range(nrows):
	for j in range(ncols):
	    entry_index=j*nrows+i
	    try:
		text=entries[entry_index]
		text=format_string[j].format(text)
	    except IndexError:
		text = ''
	    
	    docio.write(text)
	docio.write('\n')
    for j in range(ncols):
	docio.write(format_string[j].format('----'))
    docio.write('\n')

def generate_class_index(docio,ncols):
    content = []
    for cinfo in moose.element('/classes').children:
	content.append(cinfo[0].name)
    
    generate_compact_index(docio,content,ncols)

def generate_func_index(docio,ncols):
    content = []
    for fname, fdef in inspect.getmembers(moose, inspect.isbuiltin) + inspect.getmembers(moose, inspect.isfunction):
	content.append(fname)
    
    generate_compact_index(docio,content,ncols)

if __name__ == '__main__':
    if len(sys.argv)  > 1:
	docio = open(sys.argv[1], 'w')
    else:
	docio = cStringIO.StringIO()
    
    ts = datetime.now()
    docio.write('% Documentation for all MOOSE classes and functions\n')
    docio.write('% As visible in the Python module\n' )
    docio.write( ts.strftime( '% Auto-generated on %B %d, %Y\n' ) )
    
    ncols=6
    docio.write('\n' + config['heading1'] + 'Index for MOOSE Classes\n')
    generate_class_index(docio,ncols)
    docio.write('\n\n'+config['heading1'] + 'Index for MOOSE Functions\n')
    generate_func_index(docio,ncols)
    
    docio.write('\n' + config['heading1'] + 'MOOSE Classes\n')
    extract_all_class_doc(docio, config['heading2'], config['heading4'])
    docio.write('\n\n'+config['heading1'] + 'MOOSE Functions\n')
    extract_all_func_doc(docio, config['heading2'], config['heading4'])
    if isinstance(docio, cStringIO.OutputType):
	print docio.getvalue()
    else:
	docio.close()

# 
# digestbuiltindocs.py ends here
