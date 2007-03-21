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
#include "ArrayFinfo.h"

/**
 * For now we just require that the name be followed immediately
 * by square braces, and that there is a single index in there.
 * Later may consider wildcards.
 */
const Finfo* ArrayFinfo::match( Element* e, const string& s ) const
{
	std::string::size_type openpos = name().length();
	if ( s.substr( 0, openpos ) != name() )
		return 0;

	if ( s.length() < name().length() + 3 ) {
		cout << "Error: ArrayFinfo::match: bad indexing: " <<
				s << endl;
		return 0;
	}

	std::string::size_type closepos = s.length() - 1;

	if ( s[openpos] == '[' && s[closepos] == ']' ) {
		int index = atoi( 
			s.substr( openpos + 1, closepos - openpos - 1).c_str() );
		if ( index < 0 ) {
			cout << "Error: ArrayFinfo::match: -ve index: " <<
				s << endl;
			return 0;
		} else {
			char num[40];
			sprintf( num, "%d", index );
			string n = name() + "[" + num + "]";
			DynamicFinfo* ret = 
				new DynamicFinfo(
					n,
					this, 
					set_, get_,
					ftype()->recvFunc(), ftype()->trigFunc(),
					index
				);
			e->addFinfo( ret );
			return ret;
		}
	}
	return 0;
}

/// Dummy function: DynamicFinfo should handle
bool ArrayFinfo::add( 
	Element* e, Element* destElm, const Finfo* destFinfo
	) const 
{
		assert( 0 );
		return 0;
}
			
/// Dummy function: DynamicFinfo should handle
bool ArrayFinfo::respondToAdd(
		Element* e, Element* src, const Ftype *srcType,
		FuncList& srcFl, FuncList& returnFl,
		unsigned int& destIndex, unsigned int& numDest
) const
{
		assert( 0 );
		return 0;
}

/// Dummy function: DynamicFinfo should handle
void ArrayFinfo::dropAll( Element* e ) const
{
		assert( 0 );
}

/// Dummy function: DynamicFinfo should handle
bool ArrayFinfo::drop( Element* e, unsigned int i ) const
{
		assert( 0 );
		return 0;
}
