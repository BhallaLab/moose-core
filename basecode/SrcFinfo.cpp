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
 * Disconnects all messages out of this SrcFinfo.
 */
void SrcFinfo::dropAll( Element* e ) const
{
	vector< Conn >::const_iterator i;
	unsigned int begin;
	unsigned int end;
	if ( srcIndex_ > 0 ) {
		begin = e->connSrcBegin( srcIndex_ )->sourceIndex( e );
		end = e->connSrcVeryEnd( srcIndex_ )->sourceIndex( e );
		for ( unsigned int j = end; j > begin; j-- )
			e->disconnect( j - 1 );
	}
}

/**
 * Delete a specific message emerging from this SrcFinfo. Returns True
 * if the index is valid and the operation succeeds.
 */
bool SrcFinfo::drop( Element* e, unsigned int i ) const
{
	unsigned int begin = e->connSrcBegin( srcIndex_ )->sourceIndex( e );
	unsigned int end = e->connSrcVeryEnd( srcIndex_ )->sourceIndex( e );
	i += begin;
	if ( i < end ) {
		e->disconnect( i );
		return 1;
	}
	return 0;
}

/**
 * for now numIncoming does not look at possible future internal
 * messages or relayed messages.
 */
unsigned int SrcFinfo::numIncoming( const Element* e ) const
{
		return 0;
}

unsigned int SrcFinfo::numOutgoing( const Element* e ) const
{
	return ( e->connSrcVeryEnd( srcIndex_ ) - e->connSrcBegin( srcIndex_ ));
}
			
/**
 * Here we look up all the inputs to this MsgSrc, especially the 
 * internal messages that deal with traversal.
 * \todo: We don't yet have internal messages handled.
 * Return the list size.
 */
unsigned int SrcFinfo::incomingConns(
				const Element* e, vector< Conn >& list ) const
{
	return list.size();
}

/**
 * Here we look up all the targets on this MsgSrc, and fill up the 
 * list. Return the list size.
 */
unsigned int SrcFinfo::outgoingConns(
				const Element* e, vector< Conn >& list ) const
{
	list.resize( 0 );
	list.insert( list.end(), e->connSrcBegin( srcIndex_ ),
					e->connSrcVeryEnd( srcIndex_ ) );
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

bool SrcFinfo::inherit( const Finfo* baseFinfo )
{
	const SrcFinfo* other =
			dynamic_cast< const SrcFinfo* >( baseFinfo );
	if ( other && ftype()->isSameType( baseFinfo->ftype() ) ) {
			srcIndex_ = other->srcIndex_;
			return 1;
	} 
	return 0;
}


bool SrcFinfo::getSlotIndex( const string& name, unsigned int& ret ) const
{
	if ( name != this->name() ) return 0;
	ret = srcIndex_;
	return 1;
}
