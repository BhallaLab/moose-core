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
#include "DestFinfo.h"
#include "SrcFinfo.h"


/**
 * This variant of the constructor is deprecated. It only knows the 
 * ftype and the RecvFunc, which is insufficient for many purposes.
 * See below.
 */
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
 * This variant of the constructor uses an array of Finfos instead
 * of the pair version above. For simplicity we just use Finfos to set
 * up the required info for each part of the message. We assume the Finfos
 * are either SrcFinfo or DestFinfo: other clever variants are not allowed
 * at this time.
 */
SharedFinfo::SharedFinfo( const string& name, Finfo** finfos,
				 unsigned int nFinfos )
	: Finfo( name, new SharedFtype( finfos, nFinfos ) )
{
	numSrc_ = 0;
	for ( unsigned int i = 0; i < nFinfos; i++ ) {
		DestFinfo *df = dynamic_cast< DestFinfo* >( finfos[i] );

		if ( df == 0 ) {
			// It is a MsgSrc: paranoia check.
			assert( dynamic_cast< SrcFinfo* >( finfos[i] ) != 0);
			numSrc_++;
			names_.push_back( finfos[i]->name() );
		} else {
			rfuncs_.push_back( df->recvFunc() );
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
	// This test is ugly and needs to be fixed. 
	if ( destElm->className() == "PostMaster" && 
		!( destFinfo->name() == "parTick" || 
		destFinfo->name() == "serial" )
	)
		return addSeparateConns( e, destElm, destFinfo );

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
 * For the PostMaster and possibly other cases, we have to provide a
 * distinct Conn for each sub-message. This function does so. It
 * assumes that the target SharedFinfo also knows how to do so, and
 * we must set a flag to confirm that separate conns are being used.
 *
 * To be more precise: each unique 
 * MsgSrc must have a separate Conn, and likewise each unique MsgDest. 
 * However, the src and dests conns can overap. So if we only had 
 * 1 MsgSrc and 1 MsgDest, we would use 1 conn. If we had 2 MsgSrc
 * and 1 MsgDest, we would use 2 conns and the first one would be
 * shared by a Src and a Dest.
 */
bool SharedFinfo::addSeparateConns(
	Element* e, Element* destElm, const Finfo* destFinfo
) const
{
	FuncList srcFl = rfuncs_;
	FuncList destFl;
	unsigned int destIndex = 0;
	unsigned int numDest;

	if ( destFinfo->respondToAdd( destElm, e, ftype(),
							srcFl, destFl,
							destIndex, numDest ) )
	{
		assert( destFl.size() == numSrc_ );
		assert( numSrc_ + srcFl.size() > 0 );
		assert( destElm != 0 );
		assert( e != 0 );
		assert( destIndex != 0 );

		unsigned int originatingConn;
		unsigned int targetConn;
		unsigned int i;

		// First we decide where to put the originating Conn.
		// Note that all target conns (on the PostMaster) get put on the 
		// msgDest vector.
		if ( numSrc_ == 0 ) {  // Put it on MsgDest.
			cout << "\nSharedFinfo:" << name() << " to " << destElm->name() << " :addSeparateConns:numSrc_==0:Conns =\n";
			for ( i = 0; i < srcFl.size(); i++ ) {
				originatingConn = e->insertConnOnDest( msgIndex_, 1);
				targetConn = destElm->insertConnOnDest( destIndex, 1 );
				e->connect( originatingConn, destElm, targetConn );
				assert( e->lookupConn( originatingConn )->
					targetElement() == destElm );
				assert( destElm->lookupConn( targetConn )->targetElement() == e );
				cout << originatingConn << ", " << targetConn << "\n";
			}
		} else { // The usual case: mixed srcs and dests.
			unsigned int numConns = srcFl.size();
			if ( numConns < numSrc_ )
				numConns = numSrc_;
			for ( i = numSrc_; i != numConns; i++ )
				destFl.push_back( &dummyFunc );

			originatingConn = 
				e->insertSeparateConnOnSrc( msgIndex_, destFl, 0, 0 );
			/*
			 * The stride is the number of connections made
			 * on this MsgSrc. It is necessary to stride by this much
			 * for finding the correct index for all the other 
			 * Conns going out to the postmaster.
			 */
			unsigned int stride = e->connSrcEnd( msgIndex_ ) -
				e->connSrcBegin( msgIndex_ );

			cout << "\nSharedFinfo:" << name() << " to " << destElm->name() << " :addSeparateConns:numSrc_!=0:Conns =\n";
			for ( i = 0; i < numConns; i++ ) {
				targetConn = destElm->insertConnOnDest( destIndex, 1 );
				e->connect( originatingConn + i * stride,
					destElm, targetConn );
				assert( e->lookupConn( originatingConn + i * stride )->
					targetElement() == destElm );
				assert( destElm->lookupConn( targetConn )->
					targetElement() == e );
				cout << originatingConn + i * stride << ", " << targetConn << "\n";
			}
		}
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

/**
 * Deletes all connections for this SharedFinfo by iterating through the
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
 * For the SharedFinfo we have the extra issue that we do not know
 * ahead of time whether the messages are going to be on the MsgSrc
 * vector or the MsgDest vector. 
 */
void SharedFinfo::dropAll( Element* e ) const
{
	vector< Conn >::const_iterator i;
	unsigned int begin;
	unsigned int end;
	// This assumes that the child message is not a shared message.
	if ( msgIndex_ == 0 )
			return;

	if ( numSrc_ == 0 ) { // Messages are on the MsgDest vector.
		begin = e->connDestBegin( msgIndex_ )->sourceIndex( e );
		end = e->connDestEnd( msgIndex_ )->sourceIndex( e );
		for ( unsigned int j = end; j > begin; j-- )
			e->disconnect( j - 1 );
	} else { // Otherwise put on msgSrc Vector
		begin = e->connSrcBegin( msgIndex_ )->sourceIndex( e );
		end = e->connSrcVeryEnd( msgIndex_ )->sourceIndex( e );
		for ( unsigned int j = end; j > begin; j-- )
			e->disconnect( j - 1 );
	}
}

/**
 * Deletes a specific connection into this SharedFinfo. The index is 
 * numbered within this Finfo because the most common use case is to
 * pick a specific index from a vector of Conns coming into this
 * Finfo.
 * For the SharedFinfo we have the extra issue that we do not know
 * ahead of time whether the messages are going to be on the MsgSrc
 * vector or the MsgDest vector. 
 */
bool SharedFinfo::drop( Element* e, unsigned int i ) const
{
	if ( msgIndex_ == 0 ) {
		cout << "SharedFinfo::drop: No messages found\n";
		return 0;
	}

	unsigned int begin;
	unsigned int end;
	if ( numSrc_ == 0 ) { // Messages are on the MsgDest vector
		begin = e->connDestBegin( msgIndex_ )->sourceIndex( e );
		end = e->connDestEnd( msgIndex_ )->sourceIndex( e );
	} else {
		begin = e->connSrcBegin( msgIndex_ )->sourceIndex( e );
		end = e->connSrcVeryEnd( msgIndex_ )->sourceIndex( e );
	}

	i += begin;
	if ( i < end ) {
		e->disconnect( i );
		return 1;
	}
	return 0;
}
			
/**
 * The semantics of this are a bit tricky. Shared messages
 * are usually both incoming and outgoing. So usually we will
 * want to report the messages as both. What to do about the
 * rare unidirectional cases (all incoming or all outgoing) ?
 * The principle of least surprise suggests that we try to keep
 * the behaviour the same in all cases. Furthermore, a user would
 * normally query the appropriate direction case, where the
 * answer would be right anyway. So, the numIncoming does the
 * same as the numOutgoing function and reports the number of
 * connections on the SharedFinfo without worrying about direction.
 */
unsigned int SharedFinfo::numIncoming( const Element* e ) const
{
	if ( msgIndex_ == 0 )
			return 0;

	if ( numSrc_ == 0 ) {
		return ( 
			e->connDestEnd( msgIndex_ ) - e->connDestBegin( msgIndex_ )
		);
	}

	return (
			e->connSrcVeryEnd( msgIndex_ ) - e->connSrcBegin( msgIndex_ )
		);
}

/**
 * Does the same as numIncoming.
 */
unsigned int SharedFinfo::numOutgoing( const Element* e ) const
{
	return numIncoming( e );
}

/**
 * Same issue of semantics. Here we report all connections regardless
 * of direction.
 */
unsigned int SharedFinfo::incomingConns(
				const Element* e, vector< Conn >& list ) const
{
	list.resize( 0 );
	if ( msgIndex_ != 0 ) {
		if ( numSrc_ == 0 ) {
			list.insert( list.end(), e->connDestBegin( msgIndex_ ),
					e->connDestEnd( msgIndex_ ) );
		} else {
			list.insert( list.end(), e->connSrcBegin( msgIndex_ ),
					e->connSrcVeryEnd( msgIndex_ ) );
		}
	}

	return list.size();
}

unsigned int SharedFinfo::outgoingConns(
				const Element* e, vector< Conn >& list ) const
{
	return incomingConns( e, list );
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
		// Hack for testing
		if ( name() == "slave" )
			srcNum += 4;
		else 
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

bool SharedFinfo::inherit( const Finfo* baseFinfo )
{
	const SharedFinfo* other =
			dynamic_cast< const SharedFinfo* >( baseFinfo );
	if ( other && ftype()->isSameType( baseFinfo->ftype() ) ) {
			msgIndex_ = other->msgIndex_;
			numSrc_ = other->numSrc_;
			return 1;
	} 
	return 0;
}

/**
 * In this class, the field name could refer to the SharedFinfo as a 
 * whole, in which case it is just the name.
 * Or it could refer to one of the subfields. In this case it uses the
 * format:         sharedname.subname
 * where the two names are separated by a dot.
 * Note that this refers only to Src fields.
 */
bool SharedFinfo::getSlotIndex( const string& field, 
					unsigned int& ret ) const
{
	string::size_type len = this->name().length();
	if ( field.length() < len )
		return 0;

	if ( field.substr( 0, len ) == this->name() ) { 
		if ( field.length() == len ) { // this is the whole SharedFinfo
			ret = msgIndex_;
			return 1;
		} else { // Look for a string within the original, skipping a dot.
			if ( field[len] != '.' )
				return 0;
			string temp = field.substr( len + 1 );

			vector< string >::const_iterator i = 
				find( names_.begin(), names_.end(), temp );
			if ( i == names_.end() ) {
				return 0;
			} else {
				ret = msgIndex_ + ( i - names_.begin() );
				return 1;
			}
		}
	} else {
		return 0;
	}
}

////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include "moose.h"

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
			// 0 is the readVal trig MsgSrc., but we have to
			// increment it to 1 because of base class.
			send0( e, 1 );
		}

		static void pingPong( const Conn& c ) {
			Element* e = c.targetElement();
			SharedTest* st = 
				static_cast< SharedTest* >( e->data() );
			// 1 is the pingPong dval MsgSrc. We have to increment it to
			// 2 because of the base class
			send1< double >( e, 2, st->dval_ );
		}

		static void trigPing( const Conn& c ) {
			Element* e = c.targetElement();
			// 2 is the pingPong trig MsgSrc. We have to increment it
			// to 3 because of the base class.
			send0( e, 3 );
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
					initNeutralCinfo(),
					testFinfos, 
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
