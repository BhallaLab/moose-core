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

DestFinfo::DestFinfo( const string& name, const Ftype *f, 
							RecvFunc rfunc, unsigned int destIndex )
	: Finfo( name, f ), 
	rfunc_( rfunc ), 
	destIndex_( destIndex )
{
	// Save the function data.
	getFunctionDataManager()->add( rfunc, f );
}

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

/**
 * Deletes all connections for this DestFinfo by iterating through the
 * list of connections. This is non-trivial, because the connection
 * indices change as we delete them.
 * We use two attributes of the connections: 
 * First, they are sequential.
 * Second, deleting higher connections does not affect lower connection
 * indices.
 * So we just find the range and delete them in reverse order.
 * Note that we do not notify the target Finfo or Element that these
 * connections are being deleted. This operation must be guaranteed to 
 * have no other side effects.
 */
void DestFinfo::dropAll( Element* e ) const
{
	vector< Conn >::const_iterator i;

	i = e->connDestBegin( destIndex_ );
	unsigned int begin = i->sourceIndex( e );
	i = e->connDestEnd( destIndex_ );
	unsigned int end = i->sourceIndex( e );

	for ( unsigned int j = end; j > begin; j-- )
		e->disconnect( j - 1 );
}

/**
 * Deletes a specific connection into this DestFinfo. The index is 
 * numbered within this Finfo because the most common use case is to
 * pick a specific index from a vector of Conns coming into this
 * Finfo.
 */
bool DestFinfo::drop( Element* e, unsigned int i ) const
{
	vector< Conn >::const_iterator k;

	k = e->connDestBegin( destIndex_ );
	unsigned int begin = k->sourceIndex( e );
	k = e->connDestEnd( destIndex_ );
	unsigned int end = k->sourceIndex( e );

	i += begin;
	if ( i < end ) {
		e->disconnect( i );
		return 1;
	}
	return 0;
}

unsigned int DestFinfo::numIncoming( const Element* e ) const
{
	return ( e->connDestEnd( destIndex_ ) - e->connDestBegin( destIndex_ ) );
}

// Doesn't handle outgoing messages yet, probably a DynamicFinfo task.
unsigned int DestFinfo::numOutgoing( const Element* e ) const
{
	return 0;
}
			
/**
 * incomingConns does a simple insertion of the dest Conn list for this
 * Finfo into the vector. It does NOT check for duplications as
 * that would turn it into an n^2 operation.
 */
unsigned int DestFinfo::incomingConns(
					const Element* e, vector< Conn >& list ) const
{
	list.resize( 0 );
	list.insert( list.end(), e->connDestBegin( destIndex_ ),
					e->connDestEnd( destIndex_ ) );
	return list.size();
}

/**
 * does not do anything: this is a job for DynamicFinfo.
 */
unsigned int DestFinfo::outgoingConns(
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
	return ftype()->strSet( e, this, s );
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

bool DestFinfo::getSlotIndex( const string& name, unsigned int& ret ) const
{
	if ( name != this->name() ) return 0;
	ret = destIndex_;
	return 1;
}
