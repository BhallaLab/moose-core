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
#include "MsgDest.h"
#include "DestFinfo.h"

/**
 * Check the request to form a message, and return true with 
 * the necessary information if all is well.
 */
bool DestFinfo::respondToAdd(
					Element* e, Element* src, const Ftype *srcType,
					FuncList& srcFl, FuncList& returnFl,
					unsigned int& destIndex, unsigned int& numDest
) const
{
	assert ( srcType != 0 );

	if ( ftype()->isSameType( srcType ) && srcFl.size() == 0 ) {
		assert ( src != 0 && e != 0 );
		assert ( returnFl.size() == 0 );
		returnFl.push_back( rfunc_ );
		destIndex = destIndex_;
		numDest = 1;
		return 1;
	}
	return 0;
}
			
/// \Todo: implement the lookup of the list.
unsigned int DestFinfo::srcList(
					const Element* e, vector< Conn >& list ) const
{
	return list.size();
}

unsigned int DestFinfo::destList(
					const Element* e, vector< Conn >& list ) const
{
	return list.size();
}

/**
 * Directly call the recvFunc on the element with the string argument
 * typecast appropriately.
 */
bool DestFinfo::strSet( Element* e, const std::string &s ) const
{
	/**
	 * \todo Here we will ask the Ftype to do the string conversion
	 * and call the properly typecast rfunc.
	 */
	return 0;
}

const Finfo* DestFinfo::match( 
	const Element* e, unsigned int connIndex ) const
{
	return ( e->isConnOnDest( destIndex_, connIndex ) ? this : 0 );
}

bool DestFinfo::inherit( const Finfo* baseFinfo )
{
	const DestFinfo* other =
			dynamic_cast< const DestFinfo* >( baseFinfo );
	if ( other && ftype()->isSameType( baseFinfo->ftype() ) ) {
			destIndex_ = other->destIndex_;
			return 1;
	} 
	return 0;
}
