/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

//#include "header.h"
#include "moose.h"
#include "SharedFtype.h"
#include "SharedFinfo.h"
#include "DestFinfo.h"
#include "SrcFinfo.h"
#include "SimpleConn.h"


/**
 * This variant of the constructor uses an array of Finfos instead
 * of the pair version above. For simplicity we just use Finfos to set
 * up the required info for each part of the message. We assume the Finfos
 * are either SrcFinfo or DestFinfo: other clever variants are not allowed
 * at this time.
 */
SharedFinfo::SharedFinfo( const string& name, Finfo** finfos,
				 unsigned int nFinfos, const string& doc )
	: Finfo( name, new SharedFtype( finfos, nFinfos ), doc )
{
	assert( nFinfos > 0 );
	isDest_ = ( dynamic_cast< DestFinfo* >( finfos[0] ) );
	for ( unsigned int i = 0; i < nFinfos; i++ ) {
		DestFinfo *df = dynamic_cast< DestFinfo* >( finfos[i] );

		if ( df == 0 ) {
			// It is a MsgSrc: paranoia check.
			assert( dynamic_cast< SrcFinfo* >( finfos[i] ) != 0);
			names_.push_back( finfos[i]->name() );
		} else {
			rfuncs_.push_back( df->recvFunc() );
			destTypes_.push_back( df->ftype() );
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
 * There is a special case where we want to send a shared message between
 * two identical objects, using the same Finfo. This would happen if
 * we had a completely symmetrical message and we did not want to bring
 * in artificial source and dests.
 *
 */
bool SharedFinfo::add(
	Eref e, Eref destElm, const Finfo* destFinfo,
	unsigned int connTainerOption 
) const
{
	unsigned int srcFuncId = fv_->id();
	unsigned int destFuncId = 0;
	int destMsg = 0;
	unsigned int destIndex = 0;

	if ( destFinfo->respondToAdd( destElm, e, ftype(),
							srcFuncId, destFuncId,
							destMsg, destIndex ) )
	{
		assert ( FuncVec::getFuncVec( destFuncId )->size() == 
			names_.size() );
		assert ( names_.size() > 0 );

		unsigned int srcIndex = e.e->numTargets( msg_, e.i );

		if ( srcFuncId == destFuncId ) { // special case of matching msgs.
			return Msg::add(
				e, destElm, msg_, destMsg,
				srcIndex, destIndex, 
				srcFuncId, destFuncId,
				connTainerOption );
		}

		if ( isDest_ ) {
			if ( FuncVec::getFuncVec( destFuncId )->isDest() ) {
				cout << "Error: SharedFinfo::add: dest at both ends: " <<
				e.e->name() << "." << name() << " to " << 
				destElm.e->name() << "." << destFinfo->name() << endl;
				return 0;
			}

			return Msg::add( 
				destElm, e, destMsg, msg_,
				destIndex, srcIndex, 
				destFuncId, srcFuncId,
				connTainerOption );
		} else {
			if ( !FuncVec::getFuncVec( destFuncId )->isDest() ) {
				cerr << "Error: SharedFinfo::add: src at both ends: " <<
				e.e->name() << "." << name() << " to " << 
				destElm.e->name() << "." << destFinfo->name() << endl;
				return 0;
			}
			return Msg::add(
				e, destElm, msg_, destMsg,
				srcIndex, destIndex, 
				srcFuncId, destFuncId,
				connTainerOption );
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
					Eref e, Eref src, const Ftype *srcType,
					unsigned int& srcFuncId, unsigned int& returnFuncId,
					int& destMsg, unsigned int& destIndex
) const
{
	assert ( srcType != 0 );
	assert ( src.e != 0 && e.e != 0 );
	assert ( returnFuncId == 0 );

	// The type comparison uses SharedFtypes, which are a composite
	// of each of the individual types in the message.
	if ( ftype()->isSameType( srcType ) && 
		FuncVec::getFuncVec( srcFuncId )->size() == names_.size() ) {
		returnFuncId = fv_->id();
		destMsg = msg_;
		destIndex = e.e->numTargets( msg_ );
		return 1;
	}
	return 0;
}

/**
 * Directly call the recvFunc on the element with the string argument
 * typecast appropriately.
 */
bool SharedFinfo::strSet( Eref e, const std::string &s ) const
{
	/**
	 * \todo Here we will ask the Ftype to do the string conversion
	 * and call the properly typecast rfunc.
	 */
	return 0;
}

void SharedFinfo::countMessages( unsigned int& num )
{
	if ( isDestOnly() ) {
		msg_ = -num;
	} else {
		msg_ = num;
	}
	num++;
}

const Finfo* SharedFinfo::match( 
	const Element* e, const ConnTainer* c ) const
{
	if ( isDest_ ) {
		if ( c->e2() == e && c->msg2() == msg_ )
			return this;
	} else {
		if ( c->e1() == e && c->msg1() == msg_ )
			return this;
	}
	return 0;
}

bool SharedFinfo::inherit( const Finfo* baseFinfo )
{
	const SharedFinfo* other =
			dynamic_cast< const SharedFinfo* >( baseFinfo );
	if ( other && ftype()->isSameType( baseFinfo->ftype() ) ) {
			msg_ = other->msg_;

			// Seems like this should be an assertion.
			isDest_ = other->isDest_;
			// numSrc_ = other->numSrc_;
			// Don't know quite what to do here.
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
bool SharedFinfo::getSlot( const string& field, Slot& ret ) const
{
	string::size_type len = this->name().length();
	if ( field.length() < len )
		return 0;

	if ( field.substr( 0, len ) == this->name() ) { 
		if ( field.length() == len ) { // this is the whole SharedFinfo
			ret = Slot( msg_, 0 );
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
				ret = Slot( msg_, i - names_.begin() );
				return 1;
			}
		}
	} else {
		return 0;
	}
}

///\todo: Still to implement most of these operations.
void SharedFinfo::addFuncVec( const string& cname )
{
	fv_ = new FuncVec( cname, name() );
	if ( rfuncs_.size() > 0 ) {
		// vector< const Ftype* > destTypes_ = ftype()->destTypes();
		assert ( rfuncs_.size() == destTypes_.size() );
		for ( unsigned int i = 0; i != rfuncs_.size(); i++ )
			fv_->addFunc( rfuncs_[i], destTypes_[i] );
	}
	if ( isDest_ )
		fv_->setDest();
}

bool SharedFinfo::isDestOnly() const
{
	return ( isDest_ && names_.size() == 0 );
}

////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
//#include "moose.h" - causes compilation failure, known bug in gcc - http://gcc.gnu.org/ml/gcc-bugs/2003-10/msg01703.html

// Set up two SharedFinfos to test things.
// One of them is designed to talk to a ValueFinfo to trigger a
// return value. The return value * 10 is added to the local dval.
//
// The other is designed to talk to itself. One part sends a double
// and has a dest to recieve it. The other part has a Ftype0 to
// trigger the send of the double, and a dest to recieve that.
// This message ping-pongs doubles back and forth, every time the
// trigger is set off. The recieving dval is incremented by 2x the value.

class SharedTest
{
	public:
		SharedTest()
				: dval_( 1.0 )
		{;}
		static void tenXdval( const Conn* c, double val ) {
			SharedTest* st = 
				static_cast< SharedTest* >(c->data() );
			st->dval_ += 10.0 * val;
		}

		static void twoXdval( const Conn* c, double val ) {
			SharedTest* st = 
				static_cast< SharedTest* >(c->data() );
			st->dval_ += 2.0 * val;
		}

		static void setDval( const Conn* c, double val ) {
			SharedTest* st = 
				static_cast< SharedTest* >(c->data() );
			st->dval_ = val;
		}

		static double getDval( Eref e ) {
				return static_cast< SharedTest* >( e.data() )->dval_;
		}

		static void trigRead( const Conn* c );
		static void pingPong( const Conn* c );
		static void trigPing( const Conn* c );

		private:
				double dval_;
};

/*
 * A bit of a hack to get the Slots set up as globals.
 */
static Slot tenXdvalSrcSlot;
static Slot pingPongTrigSlot;
static Slot pingPongDataSlot;

void sharedFinfoTest()
{
	static Finfo* readValShared[] =
	{
			new SrcFinfo( "tenXdvalSrc", Ftype0::global() ),
			new DestFinfo( "tenXdval", Ftype1< double >::global(), 
							RFCAST( &SharedTest::tenXdval ) ),
	};

	static Finfo* pingPongSrcShared[] = 
	{ 	
			new SrcFinfo( "trig", Ftype0::global(),
			"Send trigger, receive double" ),
			new DestFinfo( "recv", Ftype1< double >::global(), 
							RFCAST( &SharedTest::twoXdval ) ),
	};

	static Finfo* pingPongDestShared[] = 
	{ 	
			new DestFinfo( "trig", 
							Ftype0::global(), &SharedTest::pingPong,
							"Receive trigger, send double." ),
			new SrcFinfo( "send", Ftype1< double >::global() ),
	};

	static Finfo* testFinfos[] = 
	{
		new ValueFinfo( "dval", ValueFtype1< double >::global(), 
			SharedTest::getDval, RFCAST( &SharedTest::setDval ) ),
		new SharedFinfo( "readVal", readValShared, 
			sizeof( readValShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "pingPongSrc", pingPongSrcShared,
			sizeof( pingPongSrcShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "pingPong", pingPongDestShared,
			sizeof( pingPongDestShared ) / sizeof( Finfo* ) ),
		// new SharedFinfo( "readVal", readValTypes, 2 ),
		// new SharedFinfo( "pingPong", pingPongTypes, 4 ),
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

	FuncVec::sortFuncVec();

	tenXdvalSrcSlot = sfc.getSlot( "readVal.tenXdvalSrc" );
	pingPongTrigSlot = sfc.getSlot( "pingPongSrc.trig" );
	pingPongDataSlot = sfc.getSlot( "pingPong.send" );

	Element* e1 = sfc.create( Id::scratchId(), "e1" );
	Element* e2 = sfc.create( Id::scratchId(), "e2" );

	cout << "\nTesting SharedFinfo";


	bool bret = Eref( e1 ).add( "readVal", e2, "dval", ConnTainer::Default);
	ASSERT( bret, "Adding readVal to dval" );
	bret = Eref( e1 ).add( "pingPongSrc", e2, "pingPong", ConnTainer::Default );
	ASSERT( bret, "Adding pingPongSrc to pingPong" );
	// Note that here we test adding a Shared message backward.
	bret = Eref( e1 ).add( "pingPong", e2, "pingPongSrc", ConnTainer::Default );
	ASSERT( bret, "reverse Adding pingPongSrc to pingPong" );

/*

	const Finfo* readVal = e1->findFinfo( "readVal" );
	const Finfo* pingPongSrc = e1->findFinfo( "pingPongSrc" );
	const Finfo* pingPong = e1->findFinfo( "pingPong" );

	ASSERT( readVal->add( e1, e2, e2->findFinfo( "dval" ) ),
					"Adding readVal to dval" );
	ASSERT( pingPongSrc->add( e1, e2, pingPong ),
					"Adding pingPongSrc to pingPong" );
	
	// Note that here we test adding a Shared message backward.
	ASSERT( pingPong->add( e1, e2, pingPongSrc ),
					"reverse Adding pingPongSrc to pingPong" );

*/

	set< double >( e1, "dval", 1.0 );
	set< double >( e2, "dval", 2.0 );
	double ret = 0;
	get< double >( e1, "dval", ret );
	ASSERT( ret == 1.0, "initial e1 setup" );
	get< double >( e2, "dval", ret );
	ASSERT( ret == 2.0, "initial e2 setup" );

	// e1.readVal->e2.dval: This is a Shared finfo asking e2 for its dval
	//     and adding 10x the result to e1.dval.
	// trigRead on e1 calls a value trigger on e2, which responds by
	// sending its dval back. This is multiplied by 10 and added to e2.dval
	set( e1, "trigRead" );
	get< double >( e1, "dval", ret );
	ASSERT( ret == 21.0, "e1.dval after trigRead on shared message" );
	get< double >( e2, "dval", ret );
	ASSERT( ret == 2.0, "e2.dval after trigRead on shared message" );

	// e1.pingPong->e2.pingPong: This is a SharedFinfo too.
	// trigPing sets off the call to the pingPong function on 
	// the target, e2. This just returns its current dval.
	// The return function in e1 is twoXdval, which doubles dval and
	// adds it to the current dval of e1. e2 is unchanged.
	set( e1, "trigPing" );
	get< double >( e1, "dval", ret );
	ASSERT( ret == 25.0, "after trigPing on shared message" );
	get< double >( e2, "dval", ret );
	ASSERT( ret == 2.0, "e2.dval after trigPing on shared message" );

	// Same as above, but starting at e2 instead.
	set( e2, "trigPing" );
	get< double >( e1, "dval", ret );
	ASSERT( ret == 25.0, "after trigPing on shared message" );
	get< double >( e2, "dval", ret );
	ASSERT( ret == 52.0, "e2.dval after trigPing on shared message" );
}

void SharedTest::trigRead( const Conn* c ) {
			// 0 is the readVal trig MsgSrc., but we have to
			// increment it to 1 because of base class.
			// Slot tenXdvalSrcSlot = sfc.getSlot( "readVal.tenXdvalSrc" );
			// send0( e, 0, Slot( 6, 0 ) );
			send0( c->target(), tenXdvalSrcSlot );
}

void SharedTest::pingPong( const Conn* c ) {
			SharedTest* st = 
				static_cast< SharedTest* >( c->data() );
			// 7 is the pingPong dval MsgSrc. We have to increment it to
			// Slot sendSlot = sfc.getSlot( "pingPong.send" );
			// send1< double >( e, 0, Slot( 7, 0 ), st->dval_ );
			send1< double >( c->target(), pingPongDataSlot, st->dval_ );
}

void SharedTest::trigPing( const Conn* c ) {
			// 7 is the pingPong trig MsgSrc. We have to increment it
			// Slot trigSrcSlot = sfc.getSlot( "pingPong.trigSrc" );
			// send0( e, 0, Slot( 7, 1 ) );
			send0( c->target(), pingPongTrigSlot );
}

#endif
