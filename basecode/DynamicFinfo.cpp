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
#include <iostream>
#include <map>
#include "Cinfo.h"
#include "MsgSrc.h"
#include "MsgDest.h"
#include "SimpleElement.h"
#include "send.h"
#include "DynamicFinfo.h"
#include "DerivedFtype.h"
#include "SharedFtype.h"

/**
 * Here the DynamicFinfo intercepts add requests that would have
 * initiated from the original ValueFinfo or whatever it is 
 * substituting for.
 * This function is identical to that from SrcFinfo.
 */
bool DynamicFinfo::add( 
		Element* e, Element* destElm, const Finfo* destFinfo) const
{
		FuncList srcFl;
		FuncList destFl;
		unsigned int destIndex;
		unsigned int numDest;

		if ( destFinfo->respondToAdd( destElm, e, ftype(),
								srcFl, destFl,
								destIndex, numDest ) )
		{
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


/**
 * The DynamicFinfo, if it exists, must intercept operations directed
 * toward the original ValueFinfo. In this case it is able to
 * deal with message requests.
 * \todo: Later must also handle shared message requests. This
 * could get quite messy, because it would lead to possible 
 * interference between requests. Still haven't worked out what is
 * the least surprising outcome.
 */
bool DynamicFinfo::respondToAdd(
					Element* e, Element* src, const Ftype *srcType,
					FuncList& srcFl, FuncList& returnFl,
					unsigned int& destIndex, unsigned int& numDest
) const
{
	assert ( src != 0 && e != 0 );
	assert ( returnFl.size() == 0 );

	// Handle assignment message inputs when ftype is the the same
	// as the original Finfo
	if ( srcType->isSameType( ftype() ) && srcFl.size() == 0 ) {
		returnFl.push_back( recvFunc_ );
		destIndex = destIndex_;
		numDest = 1;
		return 1;
	}

	// Handle trigger message inputs. The trigger function was
	// passed in at creation time if the originating object was
	// a ValueFinfo. Otherwise it should be a dummyFunc. The
	// function will check for a dummyFunc and complain if it is
	// found.
	if ( Ftype0::isA( srcType ) && srcFl.size() == 0 )
	{
		if ( trigFunc_ == dummyFunc )
			return 0;
		returnFl.push_back( trigFunc_ );
		destIndex = destIndex_;
		numDest = 1;
		return 1;
	}

	// Handle SharedFinfo requests. The srcFl should have one
	// RecvFunc designed to handle the returned value. The
	// src Ftype is a SharedFtype that we will have to match.
	if ( srcFl.size() == 1 )
	{
		if ( trigFunc_ == dummyFunc )
			return 0;

		TypeFuncPair respondArray[2] = {
			TypeFuncPair( ftype(), 0 ),
			TypeFuncPair( Ftype0::global(), trigFunc_ )
		};
		SharedFtype sf( respondArray, 2 );
		if ( sf.isSameType( srcType ) ) {
			returnFl.push_back( trigFunc_ );
			// We have to put this end of the message into the
			// MsgSrc array because there is a MsgSrc for the value.
			destIndex = srcIndex_;
			// I don't think numDest is used.
			numDest = 0;
			return 1;
		}
	}
	return 0;
}
			
unsigned int DynamicFinfo::srcList(
	const Element* e, vector< Conn >& list ) const
{
	return 0;
}

unsigned int DynamicFinfo::destList(
	const Element* e, vector< Conn >& list ) const
{
	return 0;
}


/**
* The OrigFinfo knows how to do this conversion.
*/
bool DynamicFinfo::strSet( Element* e, const std::string &s ) const
{
	return 0;
}
			
// The Ftype handles this conversion.
bool DynamicFinfo::strGet( const Element* e, std::string &s ) const {
	return 0;
}


/**
 * The DynamicFinfo is able to handle both MsgSrc and MsgDest,
 * so it books a place in both. 
 * The ConnIndex is used to identify the DynamicFinfo from
 * incoming RecvFuncs. This must, of course, be kept updated
 * in the event of a change in the conn_ vector.
 * \todo: Update ConnIndex_ if there is a change in conn_
 */
///\todo this needs to be defined according to the funcs.
void DynamicFinfo::countMessages( 
	unsigned int& srcIndex, unsigned int& destIndex )
{
	srcIndex_ = srcIndex++;
	destIndex_ = destIndex++;
}

/**
 * Returns self if the specified Conn is managed by this Finfo.
 * The DynamicFinfo::match is likely to be used a lot because
 * any Finfo that doesn't have a compiled-in MsgSrc or MsgDest
 * uses DynamicFinfos, which have to be looked up using this
 * function. The problem is that the isConnOnSrc function is
 * a little tedious. 
 * \todo Need to do benchmarking to see if this needs optimising
 */
const Finfo* DynamicFinfo::match( 
				const Element* e, unsigned int connIndex ) const
{
	if ( e->isConnOnDest( destIndex_, connIndex ) )
			return this;
	if ( e->isConnOnSrc( srcIndex_, connIndex ) )
			return this;
	return 0;
}	
	

void* DynamicFinfo::traverseIndirection( void* data ) const
{
	if ( indirect_.size() == 0 )
			return 0;
	vector< IndirectType >::const_iterator i;
	for ( i = indirect_.begin(); i != indirect_.end(); i++ )
			data =  i->first( data, i->second );
	return data;
}

const DynamicFinfo* getDF( const Conn& c )
{
	// The MAXUINT index is used to show that this conn is a dummy
	// one and must not be used for finding DynamicFinfos.
	assert( c.targetIndex() != MAXUINT );
	Element* e = c.targetElement();
	const Finfo* temp = e->findFinfo( c.targetIndex() );
	const DynamicFinfo* f = dynamic_cast< const DynamicFinfo* >( temp );
	assert( f != 0 );
	return f;
}
