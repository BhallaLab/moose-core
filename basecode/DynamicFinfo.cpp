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
#include "ProcInfo.h"
#include "DerivedFtype.h"
#include "SharedFtype.h"
#include "LookupFinfo.h"
// #include "LookupFtype.h"


DynamicFinfo::~DynamicFinfo()
{
	if ( generalIndex_ != 0 ) {
		// Assume that the ftype knows what to do here.
		ftype()->destroyIndex( generalIndex_ );
	}
}

DynamicFinfo* DynamicFinfo::setupDynamicFinfo(
	Element* e, const string& name, const Finfo* origFinfo,
	RecvFunc setFunc, GetFunc getFunc,
	RecvFunc recvFunc, RecvFunc trigFunc, void* index )
{
	assert( e != 0 );


	// Note that we create this with a null index. This is because
	// the DynamicFinfo is a temporary and we don't want to lose
	// the index when we destroy it.
	DynamicFinfo* ret = new DynamicFinfo(
		name, origFinfo, setFunc, getFunc, recvFunc, trigFunc, 0
	);

	// Here we check if there is a vacant Dynamic Finfo to use
	vector< Finfo* > flist;
	vector< Finfo* >::iterator i;
	e->listLocalFinfos( flist );
	for ( i = flist.begin(); i != flist.end(); i++ ) {
		DynamicFinfo* df = dynamic_cast< DynamicFinfo* >( *i );
		if ( df ) {
			// If this DynamicFinfo is already handling the origFinfo, 
			// just reuse it, but check that the index is the same too.
			if ( df->origFinfo_ == origFinfo && 
				df->generalIndex_ == index ) {
				delete ret;
				return df;
			}
			if ( df->numIncoming( e ) == 0 &&
							df->numOutgoing( e ) == 0 ) {
				ret->srcIndex_ = df->srcIndex_;
				ret->destIndex_ = df->destIndex_;
				if ( df->generalIndex_ != 0 ) {
					df->ftype()->destroyIndex( df->generalIndex_ );
				}

				*df = *ret;
				df->generalIndex_ = index;
				delete ret;
				return df;
			}
		}
	}

	// Nope, we have to use the new DynamicFinfo.
	ret->generalIndex_ = index;
	e->addFinfo( ret );
	return ret;
}

/**
 * Here the DynamicFinfo intercepts add requests that would have
 * initiated from the original ValueFinfo or whatever it is 
 * substituting for.
 * Returns true on success.
 * It handles two cases:
 * - Just sending out a value to a target
 * - A shared message where it receives a trigger and also sends a
 *   value out to the target.
 */
bool DynamicFinfo::add( 
		Element* e, Element* destElm, const Finfo* destFinfo) const
{
		FuncList srcFl;
		FuncList destFl;
		unsigned int destIndex;
		unsigned int numDest;

		// How do we know what the target expects: a simple message
		// or a shared one? Here we use the respondToAdd to query it.
		
		if ( destFinfo->respondToAdd( destElm, e, ftype(),
								srcFl, destFl,
								destIndex, numDest ) )
		{
			assert( srcFl.size() == 0 );
			assert( destFl.size() == 1 );
			assert( numDest == 1 );
			// First we handle the case where this just sends out its
			// value to the target.
			unsigned int srcConn =
				e->insertConnOnSrc( srcIndex_, destFl, 0, 0);
			unsigned int destConn =
				destElm->insertConnOnDest( destIndex, 1 );
			e->connect( srcConn, destElm, destConn );
			return 1;
		} else {
			// Here we make a SharedFtype on the fly for passing in the
			// respondToAdd.
			pair< const Ftype*, RecvFunc >
					p1( Ftype0::global(), trigFunc_ );
			pair< const Ftype*, RecvFunc >
					p2( ftype(), 0 );
			TypeFuncPair tfp[2] = {
					p1, p2
			};
			SharedFtype sf( tfp, 2 );
			assert ( srcFl.size() == 0 );
			srcFl.push_back( trigFunc_ );
			if ( destFinfo->respondToAdd( destElm, e, &sf,
								srcFl, destFl,
								destIndex, numDest ) ) {
				// This is the SharedFinfo case where the incoming
				// message should be a Trigger to request the return.
				unsigned int originatingConn =
					e->insertConnOnSrc( srcIndex_, destFl, 0, 0);
				// Here we know that the target is source to the
				// trigger message. So its Conn must be on
				// its MsgSrc vector.
				unsigned int targetConn =
					destElm->insertConnOnSrc( destIndex, srcFl, 0, 0 );
				e->connect( originatingConn, destElm, targetConn );
				return 1;
			}
		}
		return 0;
}


/**
 * The DynamicFinfo, if it exists, must intercept operations directed
 * toward the original ValueFinfo. In this case it is able to
 * deal with message requests.
 * This Finfo must handle three kinds of requests:
 * - To assign a value: a set
 * - A request to extract a value: a trigger to send back values to
 *   the destinations from this DynamicFinfo.
 * - A sharedFinfo request: Set up both the trigger and the return.
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

/**
 * Disconnects all messages into and out of this DynamicFinfo.
 * This includes incoming as well as outgoing messages.
 */
void DynamicFinfo::dropAll( Element* e ) const
{
	vector< Conn >::const_iterator i;
	unsigned int begin;
	unsigned int end;
	if ( destIndex_ > 0 ) {
		begin = e->connDestBegin( destIndex_ )->sourceIndex( e );
		end = e->connDestEnd( destIndex_ )->sourceIndex( e );
		for ( unsigned int j = end; j > begin; j-- )
			e->disconnect( j - 1 );
	}
	if ( srcIndex_ > 0 ) {
		begin = e->connSrcBegin( srcIndex_ )->sourceIndex( e );
		end = e->connSrcVeryEnd( srcIndex_ )->sourceIndex( e );
		for ( unsigned int j = end; j > begin; j-- )
			e->disconnect( j - 1 );
	}
}

/**
 * drop has somewhat tricky semantics for the DynamicFinfo, because
 * we do not know ahead of time whether it has incoming, outgoing or
 * both kinds of messages. From the principle of least surprise, we
 * will assume that it eliminates the specified indexed message, whether
 * incoming or outgoing. If there are both kinds of messages and there
 * exists one on each with the same index then we complain.
 */
bool DynamicFinfo::drop( Element* e, unsigned int i ) const
{
	if ( destIndex_ == 0 && srcIndex_ == 0 ) {
		cout << "DynamicFinfo::drop: No messages found\n";
		return 0;
	}
	if ( destIndex_ > 0 && srcIndex_ > 0 ) {
		cout << "DynamicFinfo::drop: Ambiguous because both source and dest messages present\n";
		return 0;
	}

	unsigned int begin;
	unsigned int end;
	if ( destIndex_ > 0 ) {
		begin = e->connDestBegin( destIndex_ )->sourceIndex( e );
		end = e->connDestEnd( destIndex_ )->sourceIndex( e );
		i += begin;
		if ( i < end ) {
			e->disconnect( i );
			return 1;
		}
	}
	if ( srcIndex_ > 0 ) {
		begin = e->connSrcBegin( srcIndex_ )->sourceIndex( e );
		end = e->connSrcVeryEnd( srcIndex_ )->sourceIndex( e );
		i += begin;
		if ( i < end ) {
			e->disconnect( i );
			return 1;
		}
	}
	return 0;
}

Finfo* DynamicFinfo::copy() const
{
	void* ge = 0;
	if ( generalIndex_ != 0 ) {
		ge = ftype()->copyIndex( generalIndex_ );
	}
	DynamicFinfo* ret = new DynamicFinfo( *this );
	ret->generalIndex_ = ge;
	return ret;
}

unsigned int DynamicFinfo::numIncoming( const Element* e ) const
{
	if ( destIndex_ != 0 ) {
		return ( e->connDestEnd( destIndex_ ) -
						e->connDestBegin( destIndex_ ) );
	}
	return 0;
}

unsigned int DynamicFinfo::numOutgoing( const Element* e ) const
{
	if ( srcIndex_ != 0 ) {
		return ( e->connSrcVeryEnd( srcIndex_ ) -
						e->connSrcBegin( srcIndex_ ) );
	}
	return 0;
}
			
unsigned int DynamicFinfo::incomingConns(
	const Element* e, vector< Conn >& list ) const
{
	if ( destIndex_ != 0 ) {
		list.insert( list.end(), e->connDestBegin( destIndex_ ),
					e->connDestEnd( destIndex_ ) );
	}
	return list.size();
}

unsigned int DynamicFinfo::outgoingConns(
	const Element* e, vector< Conn >& list ) const
{
	if ( srcIndex_ != 0 ) {
		list.insert( list.end(), e->connSrcBegin( srcIndex_ ),
					e->connSrcVeryEnd( srcIndex_ ) );
	}
	return list.size();
}


/**
* The Ftype of the OrigFinfo knows how to do this conversion.
*/
bool DynamicFinfo::strSet( Element* e, const std::string &s ) const
{
	return ftype()->strSet( e, this, s );
}
			
// The Ftype handles this conversion.
bool DynamicFinfo::strGet( const Element* e, std::string &s ) const {
	return ftype()->strGet( e, this, s );
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

const Finfo* DynamicFinfo:: match( Element* e, const string& n ) const 
{
	if ( n == name() )
		return this;
	return 0;
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
	

/*
void* DynamicFinfo::traverseIndirection( void* data ) const
{
	if ( indirect_.size() == 0 )
			return 0;
	vector< IndirectType >::const_iterator i;
	for ( i = indirect_.begin(); i != indirect_.end(); i++ )
			data =  i->first( data, i->second );
	return data;
}
*/

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

bool DynamicFinfo::getSlotIndex( const string& name, 
					unsigned int& ret ) const
{
	if ( name != this->name() ) return 0;
	if ( destIndex_ != 0 )
		ret = destIndex_;
	else 
		ret = srcIndex_;
	return 1;
}
