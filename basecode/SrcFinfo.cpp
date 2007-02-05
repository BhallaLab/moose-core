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
#include "SrcFinfo.h"
#include "MsgSrc.h"

bool SrcFinfo::add(
	Element* e, Element* destElm, const Finfo* destFinfo
) const
{
	FuncList srcFl;
	FuncList destFl;
	unsigned int destIndex;
	unsigned int numDest;
	if ( destFinfo->respondToAdd( destElm, e, ftype(),
							srcFl, destFl,
							destIndex, numDest ) )
	{
		// All these assertions say that this is a single message,
		// not a shared one.
		assert( srcFl.size() == 0 );
		assert( destFl.size() == 1 );
		assert( numDest == 1 );

		unsigned int srcConn = 
				e->insertConnOnSrc( srcIndex_, destFl, 0, 0);
		unsigned int destConn = 
				destElm->insertConnOnDest( destIndex, 1 );
		e->connect( srcConn, destElm, destConn );
		return 1;
	}
	return 0;
}

bool SrcFinfo::respondToAdd(
					Element* e, Element* src, const Ftype *srcType,
					FuncList& srcfl, FuncList& returnFl,
					unsigned int& destIndex, unsigned int& numDest
) const
{
	return 0; // for now we cannot handle this.
}
			
/**
 * Here we look up all the inputs to this MsgSrc, especially the 
 * internal messages that deal with traversal.
 * Return the list size.
 */
unsigned int SrcFinfo::srcList(
				const Element* e, vector< Conn >& list ) const
{
	return list.size();
}

/**
 * Here we look up all the targets on this MsgSrc, and fill up the 
 * list. Return the list size.
 */
unsigned int SrcFinfo::destList(
				const Element* e, vector< Conn >& list ) const
{
	return list.size();
}

bool SrcFinfo::strSet( Element* e, const std::string &s ) const
{
		return 0;
}


const Finfo* SrcFinfo::match( 
				const Element* e, unsigned int connIndex ) const
{
	return ( e->isConnOnSrc( srcIndex_, connIndex ) ? this : 0 );
}
