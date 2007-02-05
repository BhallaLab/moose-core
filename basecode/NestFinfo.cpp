/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "stdlib.h"
#include "DynamicFinfo.h"
#include "NestFinfo.h"

/**
 * This function has to match either the name itself, or if there
 * are indices in the name, to also match the array index.
 * This is a little messy because it builds the indirection syntax
 * into the base code.
 * The match function recurses all the way down to the final field
 * definition, whether it is Value, Src or Dest finfo.
 * On the way it builds up a stack of inderection pairs, and at the
 * terminal point it saves the appropriate finfo as the origFinfo
 * of the DynamicFinfo.
 */
const Finfo* NestFinfo::match(
				Element* e, const string& s ) const 
{
	std::string::size_type openpos = name().length();
	if ( s.substr( 0, openpos ) != name() )
		return 0;

	vector< IndirectType > v;
	return parseName( v, s );
}

//
// Kchan->X_A->xmin
//
// Kchan is the element name
// X_A is the NestFinfo name for a gate.
// It traverses further down to get other fields.

const Finfo* NestFinfo::parseName( vector< IndirectType >& v,
				const string& path ) const 
{
	if ( path == name() ) {
		/// \todo return a dynamic Finfo set up for this field alone.
		DynamicFinfo* ret = 
			new DynamicFinfo(
				fullpath,



	}
		return 0;

 	if ( path.find( name() ) == 0 ) {
		std::string::size_type pos = path.find( "[" );
		unsigned long length = name().length();
		if ( maxIndex_ != 0 && pos == name().length() ) {
			std::string::size_type endpos = path.find( "]" );
			if ( endpos == std::string::npos ) {
				cout << "PtrFinfo::match: no closing ]: " <<
						path << endl;
				return 0;
			}
			int index = atoi( path.substr( pos+1, endpos - pos - 1 ) );
			if ( index <= 0 ) {
				cout << "Error: PtrFinfo::match: Negative index in: " <<
						path << endl;
				return 0;
			}
			length = 1 + endpos - path.begin();
		}

		//  foo->bar or foo[23]->bar
		if ( path.length() > length + 2 ) {
			if ( path.substr( length, 2 ) == "->" ) {
				// recurse.
				return findFinfo( e, path.substr( length + 2 ) );
				return parseName( v, path.substr( length + 2 ) );
			}
		}
			IndirectType ind( ptrFunc, index );
	}

 	if ( path.find( name() ) == 0 &&
		path.substr( name().length(), 2 ) == "->" ) {
		string nest_name = path.substr( name().length() + 2 );
		const Finfo* ret = cinfo_->findFinfo( e, const string& path );
		if ( ret ) {
			DynamicFinfo* df = new DynamicFinfo(
						this, ret->setFunc() , ret );
			return df;
		}
	}
}
