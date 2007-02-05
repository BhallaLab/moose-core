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
#include "SharedFtype.h"
#include "SharedFinfo.h"


SharedFinfo::SharedFinfo( const string& name, 
				 pair< const Ftype*, RecvFunc >* types,
				 unsigned int nTypes )
	: Finfo( name, new SharedFtype( types, nTypes ) )
{
	numSrc_ = 0;
	for ( unsigned int i = 0; i < nTypes; i++ ) {
		if ( types[i].second == 0 ) {	// It is a MsgSrc
				numSrc_++;
		} else {
				rfuncs_.push_back( types[i].second );
		}
	}
}


/**
 * This adds a message from a shared message src to its matching
 * msg dest. All of 
 * the grunge work is all done by the insertConnOnSrc function.
 * The only subtlety is where we put the messages: on MsgSrc or MsgDest,
 * at either end.
 * The rule is that it goes on the MsgDest if there are no srcs
 * whatsoever, otherwise we put it on MsgSrc. Note that this is
 * independent of the originator of the message add call. In most
 * cases we have MsgSrcs on both sides.
 *
 */
bool SharedFinfo::add(
	Element* e, Element* destElm, const Finfo* destFinfo
) const
{
	FuncList srcFl = rfuncs_;
	FuncList destFl;
	unsigned int destIndex;
	unsigned int numDest;

	if ( destFinfo->respondToAdd( destElm, e, ftype(),
							srcFl, destFl,
							destIndex, numDest ) )
	{
		assert ( destFl.size() == numSrc_ );
		assert ( numSrc_ + srcFl.size() > 0 );
		unsigned int originatingConn;
		unsigned int targetConn;

		// First we decide where to put the originating Conn.
		if ( numSrc_ == 0 )  // Put it on MsgDest.
			originatingConn = e->insertConnOnDest( msgIndex_, 1);
		else // The usual case: put it on MsgSrc.
			originatingConn = 
					e->insertConnOnSrc( msgIndex_, destFl, 0, 0 );

		// Now the target Conn
		if ( srcFl.size() == 0 ) { // Target has only dests.
			targetConn = destElm->insertConnOnDest( destIndex, 1 );
		} else { // Here we need to put it on target MsgSrc.
			targetConn = 
					destElm->insertConnOnSrc( destIndex, srcFl, 0, 0 );
		}

		// Finally, we're ready to do the connection.
		e->connect( originatingConn, destElm, targetConn );
		return 1;
	}
	return 0;
}


/**
 * This responds to a message request from a shared message src.
 * msg dest. The key issue here is that it has to validate message
 * types, including the guarantee that MsgSrc and MsgDest map up,
 * for the entire set.
 * Either end of a SharedFinfo can initiate the message 'add' request,
 * so either end must be prepared to do the response.
 *
 */
bool SharedFinfo::respondToAdd(
					Element* e, Element* src, const Ftype *srcType,
					FuncList& srcFl, FuncList& returnFl,
					unsigned int& destIndex, unsigned int& numDest
) const
{
	assert ( srcType != 0 );
	assert ( src != 0 && e != 0 );
	assert ( returnFl.size() == 0 );

	// The type comparison uses SharedFtypes, which are a composite
	// of each of the individual types in the message.
	if ( ftype()->isSameType( srcType ) && srcFl.size() == numSrc_ ) {
		returnFl = rfuncs_;
		destIndex = msgIndex_;
		numDest = 1;
		return 1;
	}
	return 0;
}
			
unsigned int SharedFinfo::srcList(
					const Element* e, vector< Conn >& list ) const
{
	return list.size();
}

unsigned int SharedFinfo::destList(
					const Element* e, vector< Conn >& list ) const
{
	return list.size();
}

/**
 * Directly call the recvFunc on the element with the string argument
 * typecast appropriately.
 */
bool SharedFinfo::strSet( Element* e, const std::string &s ) const
{
	/**
	 * \todo Here we will ask the Ftype to do the string conversion
	 * and call the properly typecast rfunc.
	 */
	return 0;
}

void SharedFinfo::countMessages( 
				unsigned int& srcNum, unsigned int& destNum )
{
	if ( numSrc_ > 0 ) {
		msgIndex_ = srcNum;
		srcNum += numSrc_;
	} else {
		msgIndex_ = destNum++;
	}
}

const Finfo* SharedFinfo::match( 
	const Element* e, unsigned int connIndex ) const
{
	if ( numSrc_ > 0 )
		return ( e->isConnOnSrc( msgIndex_, connIndex ) ? this : 0 );
	else
		return ( e->isConnOnDest( msgIndex_, connIndex ) ? this : 0 );
}

////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include <map>
#include "DynamicFinfo.h"
#include "ValueFinfo.h"
#include "DestFinfo.h"
#include "MsgSrc.h"
#include "MsgDest.h"
#include "SimpleElement.h"
#include "DerivedFtype.h"
#include "send.h"
#include "ValueFtype.h"
#include "Cinfo.h"
#include "Ftype2.h"
#include "setget.h"

// Set up two SharedFinfos to test things.
// One of them is designed to talk to a ValueFinfo to trigger a
// return value. The return value * 10 is added to the local dval.
//
// The other is designed to talk to itself. One part sends a double
// and has a dest to recieve it. The other part has a Ftype0 to
// trigger the send of the double, and a dest to recieve that.
// This message ping-pongs doubles back and forth, every time the
// trigger is set off. The recieving dval is set to 2x the value.

class SharedTest
{
	public:
		SharedTest()
				: dval_( 1.0 )
		{;}
		static void tenXdval( const Conn& c, double val ) {
			SharedTest* st = 
				static_cast< SharedTest* >(c.targetElement()->data() );
			st->dval_ += 10.0 * val;
		}

		static void twoXdval( const Conn& c, double val ) {
			SharedTest* st = 
				static_cast< SharedTest* >(c.targetElement()->data() );
			st->dval_ += 2.0 * val;
		}

		static void setDval( const Conn& c, double val ) {
			SharedTest* st = 
				static_cast< SharedTest* >(c.targetElement()->data() );
			st->dval_ = val;
		}

		static double getDval( const Element* e ) {
				return static_cast< SharedTest* >( e->data() )->dval_;
		}

		static void trigRead( const Conn& c ) {
			Element* e = c.targetElement();
			// 0 is the readVal trig MsgSrc.
			send0( e, 0 );
		}

		static void pingPong( const Conn& c ) {
			Element* e = c.targetElement();
			SharedTest* st = 
				static_cast< SharedTest* >( e->data() );
			// 1 is the pingPong dval MsgSrc.
			send1< double >( e, 1, st->dval_ );
		}

		static void trigPing( const Conn& c ) {
			Element* e = c.targetElement();
			// 2 is the pingPong trig MsgSrc.
			send0( e, 2 );
		}

		private:
				double dval_;
};

// We should get the following alignment of MsgSrcs:
// 0: readVal: trigger
// 1: pingPong: sending dval
// 2: pingPong: trigger.

void sharedFinfoTest()
{
	static TypeFuncPair readValTypes[] = 
	{ 	// Receive the double, and trigger its sending.
			TypeFuncPair( Ftype1< double >::global(), 
							RFCAST( &SharedTest::tenXdval ) ),
			TypeFuncPair( Ftype0::global(), 0 )
	};

	static TypeFuncPair pingPongTypes[] = 
	{ 	// Send and receive the double, send and receive the trigger.
			TypeFuncPair( Ftype1< double >::global(), 
							RFCAST( &SharedTest::twoXdval ) ),
			TypeFuncPair( Ftype1< double >::global(), 0 ),
			TypeFuncPair( Ftype0::global(), &SharedTest::pingPong ),
			TypeFuncPair( Ftype0::global(), 0 ),
	};

	static Finfo* testFinfos[] = 
	{
		new ValueFinfo( "dval", ValueFtype1< double >::global(), 
			SharedTest::getDval, RFCAST( &SharedTest::setDval ) ),
		new SharedFinfo( "readVal", readValTypes, 2 ),
		new SharedFinfo( "pingPong", pingPongTypes, 4 ),
		new DestFinfo( "trigRead", Ftype0::global(), 
						&SharedTest::trigRead ),
		new DestFinfo( "trigPing", Ftype0::global(), 
						&SharedTest::trigPing ),
	};

	Cinfo sfc( "sharedFinfoTestClass", "Upi", "Tests shared Finfos",
					"", testFinfos, 
					sizeof( testFinfos ) / sizeof( Finfo*),
					ValueFtype1< SharedTest >::global() );

	Element* e1 = sfc.create( "e1" );
	Element* e2 = sfc.create( "e2" );

	cout << "\nTesting SharedFinfo";

	const Finfo* readVal = e1->findFinfo( "readVal" );
	const Finfo* pingPong = e1->findFinfo( "pingPong" );
	ASSERT( readVal->add( e1, e2, e2->findFinfo( "dval" ) ),
					"Adding readVal to dval" );
	ASSERT( pingPong->add( e1, e2, pingPong ),
					"Adding pingPong to pingPong" );

	set< double >( e1, "dval", 1.0 );
	set< double >( e2, "dval", 2.0 );
	double ret = 0;
	get< double >( e1, "dval", ret );
	ASSERT( ret == 1.0, "initial e1 setup" );
	get< double >( e2, "dval", ret );
	ASSERT( ret == 2.0, "initial e2 setup" );

	set( e1, "trigRead" );
	get< double >( e1, "dval", ret );
	ASSERT( ret == 21.0, "after trigRead on shared message" );

	set( e2, "trigPing" );
	get< double >( e2, "dval", ret );
	ASSERT( ret == 44.0, "after trigPing on shared message" );

	set( e1, "trigPing" );
	get< double >( e1, "dval", ret );
	ASSERT( ret == 109.0, "after trigPing on shared message" );
}

#endif
