/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifdef DO_UNIT_TESTS

#include "moose.h"

/*
#include "header.h"
#include <iostream>
#include <algorithm>
#include <map>
#include "Cinfo.h"
#include "MsgSrc.h"
#include "MsgDest.h"
#include "SimpleElement.h"
#include "send.h"
#include "DynamicFinfo.h"
#include "ValueFinfo.h"
#include "DerivedFtype.h"
#include "ValueFtype.h"
#include "Ftype2.h"
#include "setget.h"
*/


/**
 * Unit tests for the Conn and SimpleElement.
 */
void connTest()
{
	// SimpleElement( name, srcSize, destSize );
	cout << "Testing conn basic stuff";
	SimpleElement e1("e1", 2, 2);
	SimpleElement e2("e2", 2, 2);
	Conn c1( &e1, 1234 );
	Conn c2( &e2, 3241 );
	ASSERT( c1.targetElement() == &e1, "targetElement access" );
	ASSERT( c1.targetIndex() == 1234, "targetIndex access" );
	
	unsigned int ic1 = e1.insertConn( 0, 1, 0, 0 );
	unsigned int ic2 = e2.insertConn( 0, 0, 0, 1 );
	e1.connect( ic1, &e2, ic2 );
	ASSERT( e1.lookupConn( ic1 )->targetElement() == &e2,
					"connected targetElement" );
	ASSERT( e1.lookupConn( ic1 )->targetIndex() == ic2,
					"connected targetIndex" );

	/**
	 * Here we set up a conn at range 1, to see if it correctly handles
	 * updating conn indices when we subsequently add conns at range 0.
	 */
	ic1 = e1.insertConn( 1, 1, 0, 0 );
	ic2 = e2.insertConn( 0, 0, 1, 1 );
	e1.connect( ic1, &e2, ic2 );
	ASSERT( e1.lookupConn( ic1 )->targetIndex() == ic2,
					"Testing conn indices following insert: 1" );
	ASSERT( e1.lookupConn( ic2 )->targetIndex() == ic1,
					"Testing conn indices following insert: 2" );
	ASSERT( e1.lookupConn( ic1 )->targetIndex() == 1,
					"Testing conn indices following insert: 3" );
	ASSERT( e2.lookupConn( ic2 )->targetIndex() == 1,
					"Testing conn indices following insert: 4" );

	ASSERT( e1.lookupConn( ic1 )->sourceElement() == &e1,
					"sourceElement()" );
	ASSERT( e1.lookupConn( ic1 )->sourceIndex( &e1 ) == ic1,
					"sourceIndex()" );

	ASSERT( e2.lookupConn( ic2 )->sourceElement() == &e2,
					"sourceElement()" );
	ASSERT( e2.lookupConn( ic2 )->sourceIndex( &e2 ) == ic2,
					"sourceIndex()" );

	SimpleElement e3( "e3", 2, 2 );
	unsigned int ic3 = e3.insertConn( 0, 1, 0, 0 );
	ic2 = e2.insertConn( 0, 0, 0, 1 );
	e3.connect( ic3, &e2, ic2 );
	ASSERT( ic3 == 0 && ic2 == 1,
					"Testing conn indices following insert: 5" );
	ASSERT( e3.lookupConn( ic3 )->targetIndex() == 1,
					"Testing conn indices following insert: 6" );

	ASSERT( e1.lookupConn( ic1 )->targetIndex() == 2,
		"Testing conn indices following insert: This is the key one" );


	ASSERT( e1.lookupConn( ic1 )->sourceElement() == &e1,
					"sourceElement()" );
	ASSERT( e1.lookupConn( ic1 )->sourceIndex( &e1 ) == ic1,
					"sourceIndex()" );

	ASSERT( e2.lookupConn( ic2 )->sourceElement() == &e2,
					"sourceElement()" );
	ASSERT( e2.lookupConn( ic2 )->sourceIndex( &e2 ) == ic2,
					"sourceIndex()" );

	ASSERT( e3.lookupConn( ic3 )->sourceElement() == &e3,
					"sourceElement()" );
	ASSERT( e3.lookupConn( ic3 )->sourceIndex( &e3 ) == ic3,
					"sourceIndex()" );

	/*
	 * The current connections are:
	 * e1_r0_c0 -> e2_r0_c0
	 * e3_r0_c0 -> e2_r0_c1
	 * e1_r1_c1 -> e2_r1_c2
	 */
	
	ASSERT( e2.lookupConn( 0 )->targetIndex() == 0, "e2 target" );
	ASSERT( e2.lookupConn( 0 )->targetElement() == &e1, "e2 target" );
	ASSERT( e2.lookupConn( 1 )->targetIndex() == 0, "Check e2 target 2" );
	ASSERT( e2.lookupConn( 1 )->targetElement() == &e3, "Check e2 target 2" );
	ASSERT( e2.lookupConn( 2 )->targetIndex() == 1, "Check e2 target 3" );
	ASSERT( e2.lookupConn( 2 )->targetElement() == &e1, "Check e2 target 3" );
	
	/**
	 * Here we delete a conn and check that the ranges and indices
	 * are properly fixed up.
	 */
	e1.disconnect( 0 );
	/*
	 * This should leave us with:
	 * e3_r0_c0 -> e2_r0_c0
	 * e1_r1_c0 -> e2_r1_c1
	 */
	ASSERT( e1.connSize() == 1, "Check conn size after delete" );
	ASSERT( e2.connSize() == 2, "Check conn size after delete" );
	ASSERT( e3.connSize() == 1, "Check conn size after delete" );

	ASSERT( e1.lookupConn( 0 )->targetIndex() == 1, "Check e1 target" );
	ASSERT( e1.lookupConn( 0 )->targetElement() == &e2, "Check e1 target" );
	ASSERT( e2.lookupConn( 0 )->targetIndex() == 0, "Check e2 target" );
	ASSERT( e2.lookupConn( 0 )->targetElement() == &e3, "Check e2 target" );
	ASSERT( e2.lookupConn( 1 )->targetIndex() == 0, "Check e2 target 2" );
	ASSERT( e2.lookupConn( 1 )->targetElement() == &e1, "Check e2 target 2" );
	ASSERT( e3.lookupConn( 0 )->targetIndex() == 0, "Check e3 target" );
	ASSERT( e3.lookupConn( 0 )->targetElement() == &e2, "Check e3 target" );


	ASSERT( e1.lookupConn( 0 )->sourceElement() == &e1,
					"sourceElement()" );
	ASSERT( e1.lookupConn( 0 )->sourceIndex( &e1 ) == 0,
					"sourceIndex()" );

	ASSERT( e2.lookupConn( 0 )->sourceElement() == &e2,
					"sourceElement()" );
	ASSERT( e2.lookupConn( 0 )->sourceIndex( &e2 ) == 0,
					"sourceIndex()" );
	ASSERT( e2.lookupConn( 1 )->sourceElement() == &e2,
					"sourceElement()" );
	ASSERT( e2.lookupConn( 1 )->sourceIndex( &e2 ) == 1,
					"sourceIndex()" );

	ASSERT( e3.lookupConn( 0 )->sourceElement() == &e3,
					"sourceElement()" );
	ASSERT( e3.lookupConn( 0 )->sourceIndex( &e3 ) == 0,
					"sourceIndex()" );
}

/**
 * Unit tests for the MsgSrc
 */

// Nasty ugly globals here for the test funcs to work on.

vector < unsigned int > targetIndex;
vector < unsigned int > funcNum;
unsigned int funcCounter = 0;
const char* sourceName = "";
const char* targetName = "";

void commonTestFunc( const Conn& c, unsigned int f )
{
	ASSERT( c.sourceElement()->name() == sourceName, "commonTestFunc: sourceElement" );
	ASSERT( c.targetElement()->name() == targetName, "commonTestFunc: targetElement" );
	ASSERT( c.targetIndex() == targetIndex[ funcCounter ], "commonTestFunc: targetIndex" );
	ASSERT( f == funcNum[ funcCounter++ ], "commonTestFunc: funcNum" );
}

void msgSrcTestFunc1( const Conn& c ) {
	commonTestFunc( c, 1 );
}

void msgSrcTestFunc2( const Conn& c ) {
	commonTestFunc( c, 2 );
}

void msgSrcTestFunc3( const Conn& c ) {
	commonTestFunc( c, 3 );
}

void msgSrcTestFunc4( const Conn& c ) {
	commonTestFunc( c, 4 );
}

void msgSrcTestFuncDbl( const Conn& c, double v ) {
	*static_cast< double* >( c.targetElement()->data() ) = v;
	commonTestFunc( c, 5 );
}

/*
 * Here we set up a message structure where the first src on e1 has
 * a single dest, and the second src is shared with 3 dests.
 */
void msgSrcTest()
{
	FuncList fl;
	targetIndex.resize( 0 );
	fl.push_back( &msgSrcTestFunc1 );
	fl.push_back( &msgSrcTestFunc2 );
	fl.push_back( &msgSrcTestFunc3 );

	// SimpleElement( name, srcSize, destSize );

	SimpleElement e1("e1", 4, 2 );
	SimpleElement e2("e2", 4, 2 );

	// unsigned int insertConnOnSrc( src, FuncList& rf, dest, nDest );
	unsigned int ic1 = e1.insertConnOnSrc( 1, fl, 0, 1 );
	unsigned int ic2 = e2.insertConnOnDest( 0, 1 );

	cout << "\nTesting Conn creation on shared src and dest";

	ASSERT( e1.src_[ 1 ].begin() == 0, "Conn creation on src" );
	ASSERT( e1.src_[ 1 ].end() == 1, "Conn creation on src" );
	ASSERT( e1.src_[ 2 ].begin() == 0, "Conn creation on src" );
	ASSERT( e1.src_[ 2 ].end() == 1, "Conn creation on src" );
	ASSERT( e1.src_[ 3 ].begin() == 0, "Conn creation on src" );
	ASSERT( e1.src_[ 3 ].end() == 1, "Conn creation on src" );

	ASSERT( e1.dest_[ 0 ].begin() == 0, "Shared dest creation" );
	ASSERT( e1.dest_[ 0 ].end() == 1, "Shared dest creation" );


	e1.connect( ic1, &e2, ic2 );

	ASSERT( e1.lookupConn( ic1 )->targetElement() == &e2,
					"connected targetElement" );
	ASSERT( e1.lookupConn( ic1 )->targetIndex() == ic2,
					"connected targetIndex" );

	targetIndex.push_back( 0 );
	funcNum.push_back( 1 );
	funcCounter = 0;
	sourceName = "e1";
	targetName = "e2";
	send0( &e1, 1 );

	// Something funny here about msgdest ordering wrt msgsrc
	cout << "\nTesting insertion of conns and increment of later srcs";
	SimpleElement e3("e3", 4, 2 );
	FuncList fl2;
	fl2.push_back( &msgSrcTestFunc2 );
	unsigned int ic1a = e1.insertConnOnSrc( 0, fl2, 0, 0 );
	unsigned int ic3 = e3.insertConnOnDest( 0, 1 );
	
	e1.connect( ic1a, &e3, ic3 );


	ASSERT( e1.src_[ 0 ].begin() == 0, "Conn increment" );
	ASSERT( e1.src_[ 0 ].end() == 1, "Conn increment" );
	ASSERT( e1.src_[ 1 ].begin() == 1, "Conn increment on src" );
	ASSERT( e1.src_[ 1 ].end() == 2, "Conn increment on src" );
	ASSERT( e1.src_[ 2 ].begin() == 1, "Conn increment on src" );
	ASSERT( e1.src_[ 2 ].end() == 2, "Conn increment on src" );
	ASSERT( e1.src_[ 3 ].begin() == 1, "Conn increment on src" );
	ASSERT( e1.src_[ 3 ].end() == 2, "Conn increment on src" );
	
	ASSERT( e1.dest_[ 0 ].begin() == 1, "Shared dest creation" );
	ASSERT( e1.dest_[ 0 ].end() == 2, "Shared dest creation" );

	targetIndex[0] = 0;
	funcNum[0] = 2;
	funcCounter = 0;
	sourceName = "e1";
	targetName = "e3";
	send0( &e1, 0 );

	targetIndex[0] = 0;
	funcNum[0] = 1;
	funcCounter = 0;
	sourceName = "e1";
	targetName = "e2";
	send0( &e1, 1 );

	// Here try out the formation of link lists on the src.
	// We make a unique combination of src funcs and add a new conn
	// with it.
	cout << "\nTesting MsgSrc link list formation for shared msgs";
	fl[2] = msgSrcTestFunc4;
	unsigned int ic1b = e1.insertConnOnSrc( 1, fl, 0, 1 );
	unsigned int ic2b = e2.insertConnOnDest( 0, 1 );
	e1.connect( ic1b, &e2, ic2b );

	targetIndex[0] = 0;
	targetIndex.push_back( 1 );
	funcNum[0] = 1;
	funcNum.push_back( 1 );
	funcCounter = 0;
	sourceName = "e1";
	targetName = "e2";
	send0( &e1, 1 );

	targetIndex[0] = 0;
	targetIndex[1] = 1;
	funcNum[0] = 2;
	funcNum[1] = 2;
	funcCounter = 0;
	sourceName = "e1";
	targetName = "e2";
	send0( &e1, 2 );

	targetIndex[0] = 0;
	targetIndex[1] = 1;
	funcNum[0] = 3;
	funcNum[1] = 4;
	funcCounter = 0;
	sourceName = "e1";
	targetName = "e2";
	send0( &e1, 3 );

	ASSERT( e1.src_.size() == 7, "Check formation of link list");
	ASSERT( e1.src_[ 0 ].begin() == 0, "Conn increment" );
	ASSERT( e1.src_[ 0 ].end() == 1, "Conn increment" );
	ASSERT( e1.src_[ 1 ].begin() == 1, "Conn increment on src" );
	ASSERT( e1.src_[ 1 ].end() == 2, "Conn increment on src" );
	ASSERT( e1.src_[ 2 ].begin() == 1, "Conn increment on src" );
	ASSERT( e1.src_[ 2 ].end() == 2, "Conn increment on src" );
	ASSERT( e1.src_[ 3 ].begin() == 1, "Conn increment on src" );
	ASSERT( e1.src_[ 3 ].end() == 2, "Conn increment on src" );


	ASSERT( e1.src_[ 4 ].begin() == 2, "Conn increment on src" );
	ASSERT( e1.src_[ 4 ].end() == 3, "Conn increment on src" );
	ASSERT( e1.src_[ 5 ].begin() == 2, "Conn increment on src" );
	ASSERT( e1.src_[ 5 ].end() == 3, "Conn increment on src" );
	ASSERT( e1.src_[ 6 ].begin() == 2, "Conn increment on src" );
	ASSERT( e1.src_[ 6 ].end() == 3, "Conn increment on src" );

	ASSERT( e1.dest_[ 0 ].begin() == 1, "Shared dest creation" );
	ASSERT( e1.dest_[ 0 ].end() == 3, "Shared dest creation" );


	cout << "\nTesting conn deletion and updates of src and dests";
	e1.disconnect( ic1a );

	ASSERT( e1.src_[ 0 ].begin() == 0, "Conn delete" );
	ASSERT( e1.src_[ 0 ].end() == 0, "Conn delete" );
	ASSERT( e1.src_[ 1 ].begin() == 0, "Conn delete on src" );
	ASSERT( e1.src_[ 1 ].end() == 1, "Conn delete on src" );
	ASSERT( e1.src_[ 2 ].begin() == 0, "Conn delete on src" );
	ASSERT( e1.src_[ 2 ].end() == 1, "Conn delete on src" );
	ASSERT( e1.src_[ 3 ].begin() == 0, "Conn delete on src" );
	ASSERT( e1.src_[ 3 ].end() == 1, "Conn delete on src" );


	ASSERT( e1.src_[ 4 ].begin() == 1, "Conn delete on src" );
	ASSERT( e1.src_[ 4 ].end() == 2, "Conn delete on src" );
	ASSERT( e1.src_[ 5 ].begin() == 1, "Conn delete on src" );
	ASSERT( e1.src_[ 5 ].end() == 2, "Conn delete on src" );
	ASSERT( e1.src_[ 6 ].begin() == 1, "Conn delete on src" );
	ASSERT( e1.src_[ 6 ].end() == 2, "Conn delete on src" );

	ASSERT( e1.dest_[ 0 ].begin() == 0, "Shared dest update on delete" );
	ASSERT( e1.dest_[ 0 ].end() == 2, "Shared dest update on delete" );

	cout << "\nCompleted msgSrcTest()\n";
}


#include "DerivedFtype.h"
#include "SrcFinfo.h"
#include "DestFinfo.h"
/*
 * Here we set up a message structure where the first src on e1 has
 * a single dest, and the second src is shared with 3 dests.
 */
void msgFinfoTest()
{
	cout << "Testing use of Finfos in making messages";
	// SimpleElement( name, srcSize, destSize );
	
	double e2data = 1.5;

	SimpleElement e1("e1", 4, 2 );
	SimpleElement e2("e2", 4, 2, &e2data );
	Ftype0 zft;
	ValueFtype1< double > dft;

	SrcFinfo sf1( "sf1", &zft, 0 );
	SrcFinfo sf2( "sf2", &dft, 1 );

	DestFinfo df1( "df1", &zft, msgSrcTestFunc1, 0 );
	DestFinfo df2( "df2", &dft, 
		reinterpret_cast< RecvFunc >( msgSrcTestFuncDbl ), 0 );

	ASSERT (sf1.add( &e1, &e2, &df1 ) == 1,
					"zero to zero ftype message" );
	ASSERT (sf1.add( &e1, &e2, &df2 ) == 0,
					"Zero to dbl message" );
	ASSERT (sf2.add( &e1, &e2, &df1 ) == 0,
					"dbl to zero ftype message" );
	ASSERT (sf2.add( &e1, &e2, &df2 ) == 1,
					"dbl to dbl message" );

	ASSERT( e1.src_[ 0 ].begin() == 0, "Finfo Msg" );
	ASSERT( e1.src_[ 0 ].end() == 1, "Finfo Msg" );
	ASSERT( e1.src_[ 1 ].begin() == 1, "Finfo Msg" );
	ASSERT( e1.src_[ 1 ].end() == 2, "Finfo Msg" );


	targetIndex[0] = 1;
	targetIndex[1] = 0;
	funcNum[0] = 5;
	funcNum[1] = 0;
	funcCounter = 0;
	sourceName = "e1";
	targetName = "e2";

	ASSERT( e2data == 1.5, "Testing before double message passing" );

	send1< double >( &e1, 1, 1234.5678 );

	ASSERT( e2data == 1234.5678, "Testing after double message passing" );
	cout << "\nCompleted msgFinfoTest()\n";
}

#define DATA(e) reinterpret_cast< double* >( e->data() )
static void sum( Conn& c, double val ) {
	*static_cast< double* >( c.targetElement()->data() ) += val;
}

static void sub( Conn& c, double val ) {
	*static_cast< double* >( c.targetElement()->data() ) -= val;
}

void print( const Conn& c )
{
	// Now this is done in the unit tests, so user doesn't need to see
	/*
	double ret = 
			*static_cast< double* >( c.targetElement()->data() );
	// cout << c.targetElement()->name() << ": " << ret << endl;
	*/
}

void proc( const Conn& c )
{
	double ret = 
			*static_cast< double* >( c.targetElement()->data() );
	// It is a bad idea to use absolute indices here, because
	// these depend on what Finfos in a base class might do.
	// For now I'll set them to the correct value, but you should
	// suspect these if there are any further problems with the
	// unit tests.
	send1< double >( c.targetElement(), 1, ret );
	send1< double >( c.targetElement(), 2, ret );
}

#include <map>
#include "Cinfo.h"

void cinfoTest()
{
	/**
	 * This test class contains a double and 
	 * 4 messages going in and 4 going out.
	 * The incoming messages are sum, sub, print and proc.
	 * The outgoing messages are sumout, subout, printout and procout.
	 * The sumout and subout are triggered by the proc message.
	 */

	cout << "Testing class formation and messaging";

	const Ftype* f1d = ValueFtype1<double>::global();
	const Ftype* f0 = Ftype0::global();
	static Finfo* testFinfos[] = 
	{
		new DestFinfo( "sum", f1d, RFCAST( sum ) ),
		new DestFinfo( "sub", f1d, RFCAST( sub ) ),
		new DestFinfo( "print", f0, print ),
		new DestFinfo( "proc", f0, proc ),
		new SrcFinfo( "sumout", f1d ),
		new SrcFinfo( "subout", f1d ),
		new SrcFinfo( "printout", f0 ),
		new SrcFinfo( "procout", f0 ),
	};
	
	ASSERT (sizeof( testFinfos ) / sizeof( Finfo* ) == 8, 
					"Finfo array creation" );

	Cinfo testclass( "testclass", "Upi", "Test class", 
					initNeutralCinfo(),
					testFinfos, 
					sizeof( testFinfos ) / sizeof( Finfo*),
					f1d );

	// I want to check the indexing of srcIndex_ and destIndex_ here
	// but the fields are private. So I've made this function their
	// friend.
	
	unsigned int startIndex = testFinfos[0]->getSlotIndex();

	unsigned int sumSlotIndex = testclass.getSlotIndex( "sum" );

	ASSERT( startIndex == sumSlotIndex, 
					"getFinfoIndex during cinfo initialization" );
	ASSERT( dynamic_cast< DestFinfo* >( testFinfos[0] )->
					destIndex_ == 0 + startIndex ,
					"msg counting during cinfo inititialization" );
	ASSERT( dynamic_cast< DestFinfo* >( testFinfos[1] )->
					destIndex_ == 1 + startIndex ,
					"msg counting during cinfo inititialization" );
	ASSERT( dynamic_cast< DestFinfo* >( testFinfos[2] )->
					destIndex_ == 2 + startIndex ,
					"msg counting during cinfo inititialization" );
	ASSERT( dynamic_cast< DestFinfo* >( testFinfos[3] )->
					destIndex_ == 3 + startIndex ,
					"msg counting during cinfo inititialization" );

	startIndex = testFinfos[4]->getSlotIndex();
	unsigned int sumOutSlotIndex = testclass.getSlotIndex( "sumout" );

	ASSERT( startIndex == sumOutSlotIndex, 
					"sumOutIndex during cinfo initialization" );

	ASSERT( dynamic_cast< SrcFinfo* >( testFinfos[4] )->
					srcIndex_ == 0 + startIndex ,
					"msg counting during cinfo inititialization" );
	ASSERT( dynamic_cast< SrcFinfo* >( testFinfos[5] )->
					srcIndex_ == 1 + startIndex ,
					"msg counting during cinfo inititialization" );
	ASSERT( dynamic_cast< SrcFinfo* >( testFinfos[6] )->
					srcIndex_ == 2 + startIndex ,
					"msg counting during cinfo inititialization" );
	ASSERT( dynamic_cast< SrcFinfo* >( testFinfos[7] )->
					srcIndex_ == 3 + startIndex ,
					"msg counting during cinfo inititialization" );

	Element* clock = testclass.create( "clock" );
	Element* e1 = testclass.create( "e1" );
	Element* e2 = testclass.create( "e2" );
	Element* e3 = testclass.create( "e3" );

	cout << "\nUsing above class to make and connect objects";

	// Here we set up a small network:
	// e1 -sum-> e2
	// e1 -sum-> e3
	// e2 -sum-> e3
	// e3 -sub-> e1
	// clock -print-> e1
	// clock -print-> e2
	// clock -print-> e3
	// clock -proc-> e1
	// clock -proc-> e2
	// clock -proc-> e3
	//
	// At each tick we'll trigger proc 1, 2, and 3 in sequence.
	// Start all at 1.
	// e1	e2	e3
	//	1	1	1
	//	-3	2	4
	//	-3	-1	0
	//	4	-4	-7

	SimpleElement* se1 = static_cast< SimpleElement* >( e1 );

	testFinfos[4]->add( e1, e2, testFinfos[0] );
	ASSERT( se1->src_.size() == 4 + startIndex, "" );
	ASSERT( se1->src_[sumOutSlotIndex].begin() == 0 , "" );
	ASSERT( se1->src_[sumOutSlotIndex].end() == 1 , "" );
	ASSERT( se1->conn_.size() == 1, "" );
	ASSERT( se1->conn_[0].targetElement()->name() == "e2", "" );
	ASSERT( se1->conn_[0].targetIndex() == 0, "" );

	testFinfos[4]->add( e1, e3, testFinfos[0] );
	ASSERT( se1->src_.size() == 4 + startIndex, "" );
	ASSERT( se1->src_[sumOutSlotIndex].begin() == 0 , "" );
	ASSERT( se1->src_[sumOutSlotIndex].end() == 2 , "" );
	ASSERT( se1->conn_.size() == 2, "" );
	ASSERT( se1->conn_[1].targetElement()->name() == "e3", "" );
	ASSERT( se1->conn_[1].targetIndex() == 0, "" );

	testFinfos[4]->add( e2, e3, testFinfos[0] );
	testFinfos[5]->add( e3, e1, testFinfos[1] );

	testFinfos[6]->add( clock, e1, testFinfos[2] );
	testFinfos[6]->add( clock, e2, testFinfos[2] );
	testFinfos[6]->add( clock, e3, testFinfos[2] );
	testFinfos[7]->add( clock, e1, testFinfos[3] );
	testFinfos[7]->add( clock, e2, testFinfos[3] );
	testFinfos[7]->add( clock, e3, testFinfos[3] );

	unsigned int subOutSlotIndex = testclass.getSlotIndex( "subout" );
	unsigned int printOutSlotIndex = testclass.getSlotIndex( "printout" );
	unsigned int procOutSlotIndex = testclass.getSlotIndex( "procout" );

	ASSERT( se1->src_.size() == 4 + startIndex, "" );
	ASSERT( se1->conn_.size() == 5, "" );
	ASSERT( se1->src_[sumOutSlotIndex].begin() == 0 , "" );
	ASSERT( se1->src_[sumOutSlotIndex].end() == 2 , "" );
	ASSERT( se1->src_[subOutSlotIndex].begin() == 2 , "" );
	ASSERT( se1->src_[subOutSlotIndex].end() == 2 , "" );
	ASSERT( se1->src_[printOutSlotIndex].begin() == 2 , "" );
	ASSERT( se1->src_[printOutSlotIndex].end() == 2 , "" );
	ASSERT( se1->src_[procOutSlotIndex].begin() == 2 , "" );
	ASSERT( se1->src_[procOutSlotIndex].end() == 2 , "" );

	
	unsigned int subSlotIndex = testclass.getSlotIndex( "sub" );
	unsigned int printSlotIndex = testclass.getSlotIndex( "print" );
	unsigned int procSlotIndex = testclass.getSlotIndex( "proc" );
	ASSERT( se1->dest_.size() == 4 + sumSlotIndex, "" );
	ASSERT( se1->dest_[sumSlotIndex].begin() == 2 , "" );
	ASSERT( se1->dest_[sumSlotIndex].end() == 2 , "" );
	ASSERT( se1->dest_[subSlotIndex].begin() == 2 , "" );
	ASSERT( se1->dest_[subSlotIndex].end() == 3 , "" );
	ASSERT( se1->dest_[printSlotIndex].begin() == 3 , "" );
	ASSERT( se1->dest_[printSlotIndex].end() == 4 , "" );
	ASSERT( se1->dest_[procSlotIndex].begin() == 4 , "" );
	ASSERT( se1->dest_[procSlotIndex].end() == 5 , "" );
	ASSERT( se1->conn_.size() == 5, "" );

	// e2->conn[0] goes to e3 as it is a msgsrc
	ASSERT( se1->conn_[0].targetElement()->name() == "e2", "" );
	ASSERT( se1->conn_[0].targetIndex() == 1, "" );

	// e3->conn[0] goes to e1 as it is a msgsrc, so this comes after.
	ASSERT( se1->conn_[1].targetElement()->name() == "e3", "" );
	ASSERT( se1->conn_[1].targetIndex() == 1, "" );

	// Now we are into the msgdests on e1.
	// e3->conn[0] goes to e1 as it is a msgsrc.
	ASSERT( se1->conn_[2].targetElement()->name() == "e3", "" );
	ASSERT( se1->conn_[2].targetIndex() == 0, "" );
	ASSERT( se1->conn_[3].targetElement()->name() == "clock", "" );
	ASSERT( se1->conn_[3].targetIndex() == 0, "" );
	ASSERT( se1->conn_[4].targetElement()->name() == "clock", "" );
	ASSERT( se1->conn_[4].targetIndex() == 3, "" );

	*DATA( e1 ) = 1;
	*DATA( e2 ) = 1;
	*DATA( e3 ) = 1;

	send0( clock, printOutSlotIndex ); // print
	ASSERT( *reinterpret_cast< double* >( e1->data() ) == 1, "msg1" );
	ASSERT( *reinterpret_cast< double* >( e2->data() ) == 1, "msg2" );
	ASSERT( *reinterpret_cast< double* >( e3->data() ) == 1, "msg3" );

	send0( clock, procOutSlotIndex ); // process
	send0( clock, printOutSlotIndex ); // print
	ASSERT( *reinterpret_cast< double* >( e1->data() ) == -3, "msg1" );
	ASSERT( *reinterpret_cast< double* >( e2->data() ) == 2, "msg2" );
	ASSERT( *reinterpret_cast< double* >( e3->data() ) == 4, "msg3" );

	send0( clock, procOutSlotIndex ); // process
	send0( clock, printOutSlotIndex ); // print
	ASSERT( *reinterpret_cast< double* >( e1->data() ) == -3, "msg1" );
	ASSERT( *reinterpret_cast< double* >( e2->data() ) == -1, "msg2" );
	ASSERT( *reinterpret_cast< double* >( e3->data() ) == 0, "msg3" );

	send0( clock, procOutSlotIndex ); // process
	send0( clock, printOutSlotIndex ); // print
	ASSERT( *reinterpret_cast< double* >( e1->data() ) == 4, "msg1" );
	ASSERT( *reinterpret_cast< double* >( e2->data() ) == -4, "msg2" );
	ASSERT( *reinterpret_cast< double* >( e3->data() ) == -7, "msg3" );

	cout << "\nCompleted cinfoTest() including some messaging\n";
}

/**
 * Here we use pretty much the same class as above, but test the
 * correct lookup of finfos from their names rather than by remembering
 * indices.
 */
void finfoLookupTest()
{
	/**
	 * This test class contains a double and 
	 * 4 messages going in and 4 going out.
	 * The incoming messages are sum, sub, print and proc.
	 * The outgoing messages are sumout, subout, printout and procout.
	 * The sumout and subout are triggered by the proc message.
	 */

	cout << "Testing field lookup by name";

	const Ftype* f1d = ValueFtype1< double >::global();
	const Ftype* f0 = Ftype0::global();
	static Finfo* testFinfos[] = 
	{
		new DestFinfo( "sum", f1d, RFCAST( sum ) ),
		new DestFinfo( "sub", f1d, RFCAST( sub ) ),
		new DestFinfo( "print", f0, print ),
		new DestFinfo( "proc", f0, proc ),
		new SrcFinfo( "sumout", f1d ),
		new SrcFinfo( "subout", f1d ),
		new SrcFinfo( "printout", f0 ),
		new SrcFinfo( "procout", f0 ),
	};
	
	ASSERT (sizeof( testFinfos ) / sizeof( Finfo* ) == 8, 
					"Finfo array creation" );

	Cinfo testclass( "testclass", "Upi", "Test class",
					initNeutralCinfo(),
					testFinfos, 
					sizeof( testFinfos ) / sizeof( Finfo*),
					ValueFtype1< double >::global() );



	Element* clock = testclass.create( "clock" );
	Element* e1 = testclass.create( "e1" );
	Element* e2 = testclass.create( "e2" );
	Element* e3 = testclass.create( "e3" );

	ASSERT( clock->findFinfo( "sum" ) == testFinfos[0], "finfoLookup");
	ASSERT( clock->findFinfo( "sub" ) == testFinfos[1], "finfoLookup");
	ASSERT( clock->findFinfo( "print" ) == testFinfos[2], "finfoLookup");
	ASSERT( clock->findFinfo( "proc" ) == testFinfos[3], "finfoLookup");
	ASSERT( clock->findFinfo( "sumout" ) == testFinfos[4], "finfoLookup");
	ASSERT( clock->findFinfo( "subout" ) == testFinfos[5], "finfoLookup");
	ASSERT( clock->findFinfo( "printout" ) == testFinfos[6], "finfoLookup");
	ASSERT( clock->findFinfo( "procout" ) == testFinfos[7], "finfoLookup");

	// Here we set up a small network as before:
	// e1 -sum-> e2
	// e1 -sum-> e3
	// e2 -sum-> e3
	// e3 -sub-> e1
	// clock -print-> e1
	// clock -print-> e2
	// clock -print-> e3
	// clock -proc-> e1
	// clock -proc-> e2
	// clock -proc-> e3
	//
	// At each tick we'll trigger proc 1, 2, and 3 in sequence.
	// Start all at 1.
	// e1	e2	e3
	//	1	1	1
	//	-3	2	4
	//	-3	-1	0
	//	4	-4	-7
	
	SimpleElement* se1 = static_cast< SimpleElement* >( e1 );

	e1->findFinfo( "sumout" )->add( e1, e2, e2->findFinfo( "sum" ) );
	// testFinfos[4]->add( e1, e2, testFinfos[0] );
	
	unsigned int sumOutSlotIndex = testclass.getSlotIndex( "sumout" );

	ASSERT( se1->src_.size() == 4 + sumOutSlotIndex, "" );
	ASSERT( se1->src_[sumOutSlotIndex].begin() == 0 , "" );
	ASSERT( se1->src_[sumOutSlotIndex].end() == 1 , "" );
	ASSERT( se1->conn_.size() == 1, "" );
	ASSERT( se1->conn_[0].targetElement()->name() == "e2", "" );
	ASSERT( se1->conn_[0].targetIndex() == 0, "" );

	e1->findFinfo( "sumout" )->add( e1, e3, e3->findFinfo( "sum" ) );
	// testFinfos[4]->add( e1, e3, testFinfos[0] );
	ASSERT( se1->src_.size() == 4 + sumOutSlotIndex, "" );
	ASSERT( se1->src_[sumOutSlotIndex].begin() == 0 , "" );
	ASSERT( se1->src_[sumOutSlotIndex].end() == 2 , "" );
	ASSERT( se1->conn_.size() == 2, "" );
	ASSERT( se1->conn_[1].targetElement()->name() == "e3", "" );
	ASSERT( se1->conn_[1].targetIndex() == 0, "" );

	e2->findFinfo( "sumout" )->add( e2, e3, e3->findFinfo( "sum" ) );
	e3->findFinfo( "subout" )->add( e3, e1, e1->findFinfo( "sub" ) );
	// testFinfos[4]->add( e2, e3, testFinfos[0] );
	// testFinfos[5]->add( e3, e1, testFinfos[1] );

	clock->findFinfo( "printout" )->add( clock, e1, e1->findFinfo( "print" ) );
	clock->findFinfo( "printout" )->add( clock, e2, e2->findFinfo( "print" ) );
	clock->findFinfo( "printout" )->add( clock, e3, e3->findFinfo( "print" ) );
	clock->findFinfo( "procout" )->add( clock, e1, e1->findFinfo( "proc" ) );
	clock->findFinfo( "procout" )->add( clock, e2, e2->findFinfo( "proc" ) );
	clock->findFinfo( "procout" )->add( clock, e3, e3->findFinfo( "proc" ) );
	// testFinfos[6]->add( clock, e1, testFinfos[2] );
	// testFinfos[6]->add( clock, e2, testFinfos[2] );
	// testFinfos[6]->add( clock, e3, testFinfos[2] );
	// testFinfos[7]->add( clock, e1, testFinfos[3] );
	// testFinfos[7]->add( clock, e2, testFinfos[3] );
	// testFinfos[7]->add( clock, e3, testFinfos[3] );
	//
	unsigned int subOutSlotIndex = testclass.getSlotIndex( "subout" );
	unsigned int printOutSlotIndex = testclass.getSlotIndex( "printout" );
	unsigned int procOutSlotIndex = testclass.getSlotIndex( "procout" );

	ASSERT( se1->src_.size() == 4 + sumOutSlotIndex, "" );
	ASSERT( se1->conn_.size() == 5, "" );
	ASSERT( se1->src_[sumOutSlotIndex].begin() == 0 , "" );
	ASSERT( se1->src_[sumOutSlotIndex].end() == 2 , "" );
	ASSERT( se1->src_[subOutSlotIndex].begin() == 2 , "" );
	ASSERT( se1->src_[subOutSlotIndex].end() == 2 , "" );
	ASSERT( se1->src_[printOutSlotIndex].begin() == 2 , "" );
	ASSERT( se1->src_[printOutSlotIndex].end() == 2 , "" );
	ASSERT( se1->src_[procOutSlotIndex].begin() == 2 , "" );
	ASSERT( se1->src_[procOutSlotIndex].end() == 2 , "" );

	unsigned int sumSlotIndex = testclass.getSlotIndex( "sum" );
	unsigned int subSlotIndex = testclass.getSlotIndex( "sub" );
	unsigned int printSlotIndex = testclass.getSlotIndex( "print" );
	unsigned int procSlotIndex = testclass.getSlotIndex( "proc" );

	ASSERT( se1->dest_.size() == 4 + sumSlotIndex, "" );
	ASSERT( se1->dest_[sumSlotIndex].begin() == 2 , "" );
	ASSERT( se1->dest_[sumSlotIndex].end() == 2 , "" );
	ASSERT( se1->dest_[subSlotIndex].begin() == 2 , "" );
	ASSERT( se1->dest_[subSlotIndex].end() == 3 , "" );
	ASSERT( se1->dest_[printSlotIndex].begin() == 3 , "" );
	ASSERT( se1->dest_[printSlotIndex].end() == 4 , "" );
	ASSERT( se1->dest_[procSlotIndex].begin() == 4 , "" );
	ASSERT( se1->dest_[procSlotIndex].end() == 5 , "" );
	ASSERT( se1->conn_.size() == 5, "" );

	// e2->conn[0] goes to e3 as it is a msgsrc
	ASSERT( se1->conn_[0].targetElement()->name() == "e2", "" );
	ASSERT( se1->conn_[0].targetIndex() == 1, "" );

	// e3->conn[0] goes to e1 as it is a msgsrc, so this comes after.
	ASSERT( se1->conn_[1].targetElement()->name() == "e3", "" );
	ASSERT( se1->conn_[1].targetIndex() == 1, "" );

	// Now we are into the msgdests on e1.
	// e3->conn[0] goes to e1 as it is a msgsrc.
	ASSERT( se1->conn_[2].targetElement()->name() == "e3", "" );
	ASSERT( se1->conn_[2].targetIndex() == 0, "" );
	ASSERT( se1->conn_[3].targetElement()->name() == "clock", "" );
	ASSERT( se1->conn_[3].targetIndex() == 0, "" );
	ASSERT( se1->conn_[4].targetElement()->name() == "clock", "" );
	ASSERT( se1->conn_[4].targetIndex() == 3, "" );

	*DATA( e1 ) = 1;
	*DATA( e2 ) = 1;
	*DATA( e3 ) = 1;

	send0( clock, printOutSlotIndex ); // print
	ASSERT( *reinterpret_cast< double* >( e1->data() ) == 1, "msg1" );
	ASSERT( *reinterpret_cast< double* >( e2->data() ) == 1, "msg2" );
	ASSERT( *reinterpret_cast< double* >( e3->data() ) == 1, "msg3" );

	send0( clock, procOutSlotIndex ); // process
	send0( clock, printOutSlotIndex ); // print
	ASSERT( *reinterpret_cast< double* >( e1->data() ) == -3, "msg1" );
	ASSERT( *reinterpret_cast< double* >( e2->data() ) == 2, "msg2" );
	ASSERT( *reinterpret_cast< double* >( e3->data() ) == 4, "msg3" );

	send0( clock, procOutSlotIndex ); // process
	send0( clock, printOutSlotIndex ); // print
	ASSERT( *reinterpret_cast< double* >( e1->data() ) == -3, "msg1" );
	ASSERT( *reinterpret_cast< double* >( e2->data() ) == -1, "msg2" );
	ASSERT( *reinterpret_cast< double* >( e3->data() ) == 0, "msg3" );

	send0( clock, procOutSlotIndex ); // process
	send0( clock, printOutSlotIndex ); // print
	ASSERT( *reinterpret_cast< double* >( e1->data() ) == 4, "msg1" );
	ASSERT( *reinterpret_cast< double* >( e2->data() ) == -4, "msg2" );
	ASSERT( *reinterpret_cast< double* >( e3->data() ) == -7, "msg3" );

	cout << "\nCompleted finfoLookupTest() including some messaging\n";
}

/**
 * This test class contains a double and an int that we
 * try to access in various ways.
 */
class TestClass
{
		public:
			TestClass()
					: dval( 1234.5 ), ival( 56789 )
			{;}
			static double getDval( const Element* e ) {
				return static_cast< TestClass* >( e->data() )->dval;
			}
			static void setDval( const Conn& c, double val ) {
				static_cast< TestClass* >( 
					c.targetElement()->data() )->dval = val;
			}

			static int getIval( const Element* e ) {
				return static_cast< TestClass* >( e->data() )->ival;
			}
			static void setIval( const Conn& c, int val ) {
				static_cast< TestClass* >( 
					c.targetElement()->data() )->ival = val;
			}

			// A proper message, adds incoming val to dval.
			static void dsum( const Conn& c, double val ) {
				static_cast< TestClass* >( 
					c.targetElement()->data() )->dval += val;
			}

			// Another proper message, just assignes incomving ival.
			static void iset( const Conn& c, int val ) {
				static_cast< TestClass* >( 
					c.targetElement()->data() )->ival = val;
			}

			// another proper message. Triggers a local operation,
			// triggers sending of dval, and triggers a trigger out.
			static void proc( const Conn& c ) {
				Element* e = c.targetElement();
				TestClass* tc = static_cast< TestClass* >( e->data() );
					tc->dval *= tc->ival;

					// This sends the double value out to a target
					// dsumout == 0, but base class changes it
					send1< double >( e, 1, tc->dval );

					// This sends the int value out to a target
					// isetout == 1, but base class changes it
					send1< int >( e, 2, tc->ival );

					// This just sends a trigger to the remote object.
					// procout == 2, but base class changes it.
					// Either it will trigger dproc itself, or it
					// could trigger a getfunc.
					// 
					// Again, it is a bad idea to use a literal index
					// here because the actual index depends on
					// base classes.
					send0( e, 3 );
			}

		private:
			double dval;
			int ival;

};

#include "DynamicFinfo.h"
#include "ValueFinfo.h"
void valueFinfoTest()
{

	cout << "Testing valueFinfos";

	const Ftype* f1d = ValueFtype1< double >::global();
	const Ftype* f1i = ValueFtype1< int >::global();
	const Ftype* f0 = Ftype0::global();
	static Finfo* testFinfos[] = 
	{
		new ValueFinfo( "dval", f1d,
				TestClass::getDval,
				reinterpret_cast< RecvFunc >( &TestClass::setDval ) ),
		new ValueFinfo( "ival", f1i,
				reinterpret_cast< GetFunc >( &TestClass::getIval ), 
				reinterpret_cast< RecvFunc >( &TestClass::setIval ) ),
		new SrcFinfo( "dsumout", f1d ),
		new SrcFinfo( "isetout", f1i ),
		new SrcFinfo( "procout", f0 ),
		new DestFinfo( "dsum", f1d, RFCAST( &TestClass::dsum ) ),
		new DestFinfo( "iset", f1i, RFCAST( &TestClass::iset ) ),
		new DestFinfo( "proc", f0, &TestClass::proc ),
	};

	Cinfo testclass( "testclass2", "Upi", "Test class 2",
					initNeutralCinfo(),
					testFinfos, 
					sizeof( testFinfos ) / sizeof( Finfo*),
					ValueFtype1< TestClass >::global() );



	Element* clock = testclass.create( "clock" );
	string sret = "";
	double dret = 0;
	int iret;

	get< double >( clock, clock->findFinfo( "dval" ), dret );
	ASSERT( dret == 1234.5, "test get1");

	ASSERT( clock->findFinfo( "dval" )->strGet( clock, sret ), 
					"testing strGet" );
	ASSERT( sret == "1234.5", "testing strGet" );

	set< double >( clock, clock->findFinfo( "dval" ), 54321 );
	dret = 0;
	get< double >( clock, clock->findFinfo( "dval" ), dret );
	ASSERT( dret == 54321, "test set1");

	ASSERT( clock->findFinfo( "dval" )->strGet( clock, sret ), 
					"testing strGet" );
	ASSERT( sret == "54321", "testing strGet" );

	get< int >( clock, clock->findFinfo( "ival" ), iret );
	ASSERT( iret == 56789, "test iget");

	ASSERT( clock->findFinfo( "ival" )->strGet( clock, sret ), 
					"testing strGet" );
	ASSERT( sret == "56789", "testing strGet" );

	set< int >( clock, clock->findFinfo( "ival" ), 77777 );
	iret = 0;

	ASSERT( clock->findFinfo( "ival" )->strGet( clock, sret ), 
					"testing strGet" );
	ASSERT( sret == "77777", "testing strGet" );

	get< int >( clock, clock->findFinfo( "ival" ), iret );
	ASSERT( iret == 77777, "test iset");

	/////////////////////////////////////////////////////////////
	//  Testing strSet
	/////////////////////////////////////////////////////////////
	cout << "\nTesting valueFinfo strSet/Get";
	
	ASSERT( clock->findFinfo( "ival" )->strSet( clock, "121314" ), 
					"testing strSet" );
	get< int >( clock, clock->findFinfo( "ival" ), iret );
	ASSERT( iret == 121314, "test strset");
	ASSERT( clock->findFinfo( "ival" )->strGet( clock, sret ), 
					"testing strSet" );
	ASSERT( sret == "121314", "testing strGet" );
	
	ASSERT( clock->findFinfo( "dval" )->strSet( clock, "6.25" ), 
					"testing strSet" );
	get< double >( clock, clock->findFinfo( "dval" ), dret );
	ASSERT( dret == 6.25, "test strset");
	ASSERT( clock->findFinfo( "dval" )->strGet( clock, sret ), 
					"testing strSet" );
	ASSERT( sret == "6.25", "testing strGet" );
	

	/////////////////////////////////////////////////////////////
	// Now let's try sending messages to and from values
	// Many cases to check. We still don't have shared messages,
	// that should be yet another case.
	// --proc--> e1/dsumout --> e2/dsum
	//              This should give interesting changes in clock
	//              locally, and changes in e1 too.
	// 				Message --> regular message dest.
	// 				Just to check it all works.
	// --proc--> e1/isetout --> e3/ival
	// 				This should send a value from e1 to the integer
	// 				Message --> Value field
	// --proc--> e1/procout --> e0/dval --> e4/dsum
	// 				This should send the dval from e0 to dsum of e4.
	// 				trigger-> Value field -> regular message
	// 				Here we set up the trigger first and then the
	// 				dsum message.
	// --proc--> e1/procout --> e5/dval --> e6/dsum
	// 				This should send the dval from e5 to dsum of e6.
	// 				trigger-> Value field -> regular message
	// 				Here we set up dsum message first and then the
	// 				trigger.
	// --proc--> e1/procout --> e7/dval --> e8/dval
	// 				This should send the dval from e7 to dval of e8
	// 				trigger-> Value field -> Value field
	// 				Here we set up trigger first, then the dval msg.
	// --proc--> e1/procout --> e9/dval --> e10/dval
	// 				This should send the dval from e9 to dval of e10
	// 				trigger-> Value field -> Value field
	// 				Here we set up dval first, then the trigger msg.
	/////////////////////////////////////////////////////////////
	
	unsigned int procOutSlotIndex = testclass.getSlotIndex( "procout" );

	Element* e1 = testclass.create( "e1" );
	Element* e2 = testclass.create( "e2" );
	set< double >( e1, e1->findFinfo( "dval" ), 1 );
	set< int >( e1, e1->findFinfo( "ival" ), -1 );

	get< int >( e1, e1->findFinfo( "ival" ), iret );

	set< double >( e2, e2->findFinfo( "dval" ), 2 );
	set< int >( e2, e2->findFinfo( "ival" ), -2 );
	clock->findFinfo( "procout" )->add( clock, e1, e1->findFinfo( "proc" ) );
	e1->findFinfo( "dsumout" )->add( e1, e2, e2->findFinfo( "dsum" ) );
	send0( clock, procOutSlotIndex ); // procout
	get< double >( e1, e1->findFinfo( "dval" ), dret );
	ASSERT( dret == -1.0, "proc--> e1/dsumout --> e2/dsum" );
	get< int >( e1, e1->findFinfo( "ival" ), iret );
	ASSERT( iret == -1, "proc--> e1/dsumout --> e2/dsum" );
	get< double >( e2, e2->findFinfo( "dval" ), dret );
	ASSERT( dret == 1.0, "proc--> e1/dsumout --> e2/dsum" );
	get< int >( e2, e2->findFinfo( "ival" ), iret );
	ASSERT( iret == -2, "proc--> e1/dsumout --> e2/dsum" );

	/////////////////////////////////////////////////////////////////// 
	cout << "\nTesting --proc--> e1/isetout --> e3/ival";
	// --proc--> e1/isetout --> e3/ival
	Element* e3 = testclass.create( "e3" );
	set< double >( e3, e3->findFinfo( "dval" ), 3 );
	set< int >( e3, e3->findFinfo( "ival" ), -3 );

	get< double >( e3, e3->findFinfo( "dval" ), dret );
	ASSERT( dret == 3.0, "proc--> e1/isetout --> e3/ival" );
	get< int >( e3, e3->findFinfo( "ival" ), iret );
	ASSERT( iret == -3, "proc--> e1/isetout --> e3/ival" );

	e1->findFinfo( "isetout" )->add( e1, e3, e3->findFinfo( "ival" ) );
	send0( clock, procOutSlotIndex ); // procout

	get< double >( e1, e1->findFinfo( "dval" ), dret );
	ASSERT( dret == 1.0, "proc--> e1/isetout --> e3/ival" );
	get< int >( e1, e1->findFinfo( "ival" ), iret );
	ASSERT( iret == -1, "proc--> e1/isetout --> e3/ival" );
	get< double >( e3, e3->findFinfo( "dval" ), dret );
	ASSERT( dret == 3.0, "proc--> e1/isetout --> e3/ival" );
	get< int >( e3, e3->findFinfo( "ival" ), iret );
	ASSERT( iret == -1, "proc--> e1/isetout --> e3/ival" );

	
	/////////////////////////////////////////////////////////////////// 
	cout << "\nTesting trig then dval: --proc--> e1/procout --> e0/dval --> e4/dsum";
	// --proc--> e1/procout --> e0/dval --> e4/dsum
	Element* e0 = testclass.create( "e0" );
	Element* e4 = testclass.create( "e4" );
	set< double >( e0, e0->findFinfo( "dval" ), 2 );
	set< int >( e0, e0->findFinfo( "ival" ), -2 );
	set< double >( e4, e4->findFinfo( "dval" ), 4 );
	set< int >( e4, e4->findFinfo( "ival" ), -4 );

	dret = 123455.0;
	get< double >( e0, e0->findFinfo( "dval" ), dret );
	ASSERT( dret == 2.0,"--proc--> e1/procout --> e0/dval --> e4/dsum");
	dret = 23455.0;
	get< double >( e4, e4->findFinfo( "dval" ), dret );
	ASSERT( dret == 4.0,"--proc--> e1/procout --> e0/dval --> e4/dsum");

	ASSERT( 
		e1->findFinfo( "procout" )->
			add( e1, e0, e0->findFinfo( "dval" ) ),
			"Adding procout to dval"
		);
	ASSERT( e0->findFinfo( "dval" )->
			add( e0, e4, e4->findFinfo( "dsum" ) ),
			"adding dval to dsum"
			);

	dret = 123456.0;
	get< double >( e0, e0->findFinfo( "dval" ), dret );
	ASSERT( dret == 2.0,"--proc--> e1/procout --> e0/dval --> e4/dsum");

	send0( clock, procOutSlotIndex ); // procout


	get< double >( e1, e1->findFinfo( "dval" ), dret );
	ASSERT(dret == -1.0,"--proc--> e1/procout --> e0/dval --> e4/dsum");
	get< int >( e1, e1->findFinfo( "ival" ), iret );
	ASSERT( iret == -1, "--proc--> e1/procout --> e0/dval --> e4/dsum");

	get< double >( e0, e0->findFinfo( "dval" ), dret );
	ASSERT( dret == 2.0,"--proc--> e1/procout --> e0/dval --> e4/dsum");
	get< int >( e0, e0->findFinfo( "ival" ), iret );
	ASSERT( iret == -2, "--proc--> e1/procout --> e0/dval --> e4/dsum");

	get< double >( e4, e4->findFinfo( "dval" ), dret );
	ASSERT( dret == 6.0,"--proc--> e1/procout --> e0/dval --> e4/dsum");
	get< int >( e4, e4->findFinfo( "ival" ), iret );
	ASSERT( iret == -4, "--proc--> e1/procout --> e0/dval --> e4/dsum");
 
	/////////////////////////////////////////////////////////////////// 
	cout << "\nTesting dval then trig: --proc--> e1/procout --> e5/dval --> e6/dsum";
	// Here we first add the message from d5 to e6, then the trigger.
	Element* e5 = testclass.create( "e5" );
	Element* e6 = testclass.create( "e6" );
	set< double >( e5, e5->findFinfo( "dval" ), 5 );
	set< int >( e5, e5->findFinfo( "ival" ), -5 );
	set< double >( e6, e6->findFinfo( "dval" ), 6 );
	set< int >( e6, e6->findFinfo( "ival" ), -6 );

	dret = 123455.0;
	get< double >( e5, e5->findFinfo( "dval" ), dret );
	ASSERT( dret == 5.0,"--proc--> e1/procout --> e5/dval --> e6/dsum");
	dret = 23455.0;
	get< double >( e6, e6->findFinfo( "dval" ), dret );
	ASSERT( dret == 6.0,"--proc--> e1/procout --> e5/dval --> e6/dsum");

	// Note the order of the operations here.
	ASSERT( e5->findFinfo( "dval" )->
			add( e5, e6, e6->findFinfo( "dsum" ) ),
			"adding dval to dsum"
			);
	ASSERT( 
		e1->findFinfo( "procout" )->
			add( e1, e5, e5->findFinfo( "dval" ) ),
			"Adding procout to dval"
		);

	dret = 123456.0;
	get< double >( e5, e5->findFinfo( "dval" ), dret );
	ASSERT( dret == 5.0,"--proc--> e1/procout --> e5/dval --> e6/dsum");

	send0( clock, procOutSlotIndex ); // procout


	get< double >( e1, e1->findFinfo( "dval" ), dret );
	ASSERT(dret == 1.0,"--proc--> e1/procout --> e5/dval --> e6/dsum");
	get< int >( e1, e1->findFinfo( "ival" ), iret );
	ASSERT( iret == -1, "--proc--> e1/procout --> e5/dval --> e6/dsum");

	get< double >( e5, e5->findFinfo( "dval" ), dret );
	ASSERT( dret == 5.0,"--proc--> e1/procout --> e5/dval --> e6/dsum");
	get< int >( e5, e5->findFinfo( "ival" ), iret );
	ASSERT( iret == -5, "--proc--> e1/procout --> e5/dval --> e6/dsum");

	get< double >( e6, e6->findFinfo( "dval" ), dret );
	ASSERT( dret == 11.0,"--proc--> e1/procout --> e5/dval --> e6/dsum");
	get< int >( e6, e6->findFinfo( "ival" ), iret );
	ASSERT( iret == -6, "--proc--> e1/procout --> e5/dval --> e6/dsum");
 
	/////////////////////////////////////////////////////////////////// 
	cout << "\nTesting trig then dval: --proc--> e1/procout --> e7/dval --> e8/dval";
	// First set up trigger,  then we send message from value to value.
	Element* e7 = testclass.create( "e7" );
	Element* e8 = testclass.create( "e8" );
	set< double >( e7, e7->findFinfo( "dval" ), 7 );
	set< int >( e7, e7->findFinfo( "ival" ), -7 );
	set< double >( e8, e8->findFinfo( "dval" ), 8 );
	set< int >( e8, e8->findFinfo( "ival" ), -8 );

	dret = 11111.0;
	get< double >( e7, e7->findFinfo( "dval" ), dret );
	ASSERT( dret == 7.0,"--proc--> e1/procout --> e7/dval --> e8/dsum");
	dret = 22222.0;
	get< double >( e8, e8->findFinfo( "dval" ), dret );
	ASSERT( dret == 8.0,"--proc--> e1/procout --> e7/dval --> e8/dsum");

	// Note the order of the operations here.
	ASSERT( 
		e1->findFinfo( "procout" )->
			add( e1, e7, e7->findFinfo( "dval" ) ),
			"Adding procout to dval"
		);
	ASSERT( e7->findFinfo( "dval" )->
			add( e7, e8, e8->findFinfo( "dval" ) ),
			"adding dval to dval"
			);

	dret = 333333.0;
	get< double >( e7, e7->findFinfo( "dval" ), dret );
	ASSERT( dret == 7.0,"--proc--> e1/procout --> e7/dval --> e8/dsum");

	send0( clock, procOutSlotIndex ); // procout


	get< double >( e1, e1->findFinfo( "dval" ), dret );
	ASSERT(dret == -1.0,"--proc--> e1/procout --> e7/dval --> e8/dsum");
	get< int >( e1, e1->findFinfo( "ival" ), iret );
	ASSERT( iret == -1, "--proc--> e1/procout --> e7/dval --> e8/dsum");

	get< double >( e7, e7->findFinfo( "dval" ), dret );
	ASSERT( dret == 7.0,"--proc--> e1/procout --> e7/dval --> e8/dsum");
	get< int >( e7, e7->findFinfo( "ival" ), iret );
	ASSERT( iret == -7, "--proc--> e1/procout --> e7/dval --> e8/dsum");

	get< double >( e8, e8->findFinfo( "dval" ), dret );
	ASSERT( dret == 7.0,"--proc--> e1/procout --> e7/dval --> e8/dsum");
	get< int >( e8, e8->findFinfo( "ival" ), iret );
	ASSERT( iret == -8, "--proc--> e1/procout --> e7/dval --> e8/dsum");

	/////////////////////////////////////////////////////////////////// 
	cout << "\nTesting dval then trig: --proc--> e1/procout --> e9/dval --> e10/dval";
	// First set up message from value to value, then trigger.
	Element* e9 = testclass.create( "e9" );
	Element* e10 = testclass.create( "e10" );
	set< double >( e9, e9->findFinfo( "dval" ), 9 );
	set< int >( e9, e9->findFinfo( "ival" ), -9 );
	set< double >( e10, e10->findFinfo( "dval" ), 10 );
	set< int >( e10, e10->findFinfo( "ival" ), -10 );

	dret = 10101.0;
	get< double >( e9, e9->findFinfo( "dval" ), dret );
	ASSERT( dret == 9.0,"--proc--> e1/procout --> e9/dval --> e10/dsum");
	dret = 20202.0;
	get< double >( e10, e10->findFinfo( "dval" ), dret );
	ASSERT( dret == 10.0,"--proc--> e1/procout --> e9/dval --> e10/dsum");

	// Note the order of the operations here.
	ASSERT( e9->findFinfo( "dval" )->
			add( e9, e10, e10->findFinfo( "dval" ) ),
			"adding dval to dval"
			);
	ASSERT( 
		e1->findFinfo( "procout" )->
			add( e1, e9, e9->findFinfo( "dval" ) ),
			"Adding procout to dval"
		);

	dret = 303030.0;
	get< double >( e9, e9->findFinfo( "dval" ), dret );
	ASSERT( dret == 9.0,"--proc--> e1/procout --> e9/dval --> e10/dsum");

	send0( clock, procOutSlotIndex ); // procout


	get< double >( e1, e1->findFinfo( "dval" ), dret );
	ASSERT(dret == 1.0,"--proc--> e1/procout --> e9/dval --> e10/dsum");
	get< int >( e1, e1->findFinfo( "ival" ), iret );
	ASSERT( iret == -1, "--proc--> e1/procout --> e9/dval --> e10/dsum");

	get< double >( e9, e9->findFinfo( "dval" ), dret );
	ASSERT( dret == 9.0,"--proc--> e1/procout --> e9/dval --> e10/dsum");
	get< int >( e9, e9->findFinfo( "ival" ), iret );
	ASSERT( iret == -9, "--proc--> e1/procout --> e9/dval --> e10/dsum");

	get< double >( e10, e10->findFinfo( "dval" ), dret );
	ASSERT( dret == 9.0,"--proc--> e1/procout --> e9/dval --> e10/dsum");
	get< int >( e10, e10->findFinfo( "ival" ), iret );
	ASSERT( iret == -10, "--proc--> e1/procout --> e9/dval --> e10/dsum");
}

/**
 * This test class contains a vector of doubles, a regular double,
 * and an evaluated int with the size of the vector.
 */
class ArrayTestClass
{
		public:
			ArrayTestClass()
					: dvec( 4, 0.32 ), dval( 2.2202 )
			{
				dvec[0] = 0.1;
				dvec[1] = 0.2;
				dvec[2] = 0.3;
				dvec[3] = 0.4;
			}

			static double getDvec( const Element* e, unsigned int i ) {
				ArrayTestClass* atc = 
						static_cast< ArrayTestClass* >( e->data() );
				if ( i < atc->dvec.size() )
					return atc->dvec[i];

				cout << "Error: ArrayTestClass::getDvec: index out of range\n";
				return 0.0;
			}

			static void setDvec( 
						const Conn& c, double val, unsigned int i ) {
				ArrayTestClass* atc = 
					static_cast< ArrayTestClass* >(
									c.targetElement()->data() );
				if ( i < atc->dvec.size() )
					atc->dvec[i] = val ;
				else
					cout << "Error: ArrayTestClass::setDvec: index out of range\n";
			}

			static double getDval( const Element* e ) {
				return static_cast< ArrayTestClass* >( e->data() )->dval;
			}
			static void setDval( const Conn& c, double val ) {
				static_cast< ArrayTestClass* >( 
					c.targetElement()->data() )->dval = val;
			}

			static int getIval( const Element* e ) {
				return static_cast< ArrayTestClass* >( e->data() )->
						dvec.size();
			}
			static void setIval( const Conn& c, int val ) {
				static_cast< ArrayTestClass* >( 
					c.targetElement()->data() )->dvec.resize( val );
			}

			// A proper message, adds incoming val to dval.
			static void dsum( const Conn& c, double val ) {
				static_cast< ArrayTestClass* >( 
					c.targetElement()->data() )->dval += val;
			}

			// another proper message. Triggers a local operation,
			// triggers sending of dval, and triggers a trigger out.
			static void proc( const Conn& c ) {
				Element* e = c.targetElement();
				ArrayTestClass* tc = static_cast< ArrayTestClass* >( e->data() );
					tc->dval = 0.0;
					for ( unsigned int i = 0; i < tc->dvec.size(); i++ )
						tc->dval += tc->dvec[i];

					// This sends the double value out to a target
					// dsumout == 0, but base class shifts it.
					send1< double >( e, 1, tc->dval );

					// This just sends a trigger to the remote object.
					// procout == 1, but base class shifts it.
					// Either it will trigger dproc itself, or it
					// could trigger a getfunc.
					send0( e, 2 );
			}

		private:
			vector< double > dvec;
			double dval;
};

/*
#include "ArrayFinfo.h"
#include "ArrayFtype.h"

void arrayFinfoTest()
{

	cout << "\nTesting arrayFinfo set and get";

	const Ftype* f1a = ArrayFtype1< double >::global();
	const Ftype* f1d = ValueFtype1< double >::global();
	const Ftype* f1i = ValueFtype1< int >::global();
	const Ftype* f0 = Ftype0::global();
	static Finfo* testFinfos[] = 
	{
		new ArrayFinfo( "dvec", f1a,
				reinterpret_cast< GetFunc >( &ArrayTestClass::getDvec ),
				reinterpret_cast< RecvFunc >( &ArrayTestClass::setDvec ) ),
		new ValueFinfo( "dval", f1d,
				ArrayTestClass::getDval,
				reinterpret_cast< RecvFunc >( &ArrayTestClass::setDval ) ),
		new ValueFinfo( "ival", f1i,
				reinterpret_cast< GetFunc >( &ArrayTestClass::getIval ), 
				reinterpret_cast< RecvFunc >( &ArrayTestClass::setIval ) ),
		new SrcFinfo( "dsumout", f1d ),
		new SrcFinfo( "procout", f0 ),
		new DestFinfo( "dsum", f1d, RFCAST( &ArrayTestClass::dsum ) ),
		new DestFinfo( "proc", f0, &ArrayTestClass::proc ),
	};

	Cinfo arraytestclass( "arraytestclass", "Upi",
					"Array Test class",
					initNeutralCinfo(),
					testFinfos, 
					sizeof( testFinfos ) / sizeof( Finfo*),
					ValueFtype1< ArrayTestClass >::global() );

	Element* a1 = arraytestclass.create( "a1" );
	double dret = 0;
	int iret;

	get< double >( a1, a1->findFinfo( "dval" ), dret );
	ASSERT( dret == 2.2202, "test get1");
	set< double >( a1, a1->findFinfo( "dval" ), 555.5 );
	dret = 0;
	get< double >( a1, a1->findFinfo( "dval" ), dret );
	ASSERT( dret == 555.5, "test set1");

	get< int >( a1, a1->findFinfo( "ival" ), iret );
	ASSERT( iret == 4, "test get1");
	set< int >( a1, a1->findFinfo( "ival" ), 5 );
	iret = 0;
	get< int >( a1, a1->findFinfo( "ival" ), iret );
	ASSERT( iret == 5, "test set1");

	vector< const Finfo* > flist;
	a1->listFinfos( flist );
	size_t s = flist.size();

	get< double >( a1, a1->findFinfo( "dvec[0]" ), dret );
	ASSERT( dret == 0.1, "test get0");
	set< double >( a1, a1->findFinfo( "dvec[0]" ), 1.1 );
	dret = 0;
	get< double >( a1, a1->findFinfo( "dvec[0]" ), dret );
	ASSERT( dret == 1.1, "test set0");

	// Check that there is only one DynamicFinfo set up when looking at
	// the same array index.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 1, "Checking formation of DynamicFinfos" );


	get< double >( a1, a1->findFinfo( "dvec[1]" ), dret );
	ASSERT( dret == 0.2, "test get1");
	set< double >( a1, a1->findFinfo( "dvec[1]" ), 2.2 );
	dret = 0;
	get< double >( a1, a1->findFinfo( "dvec[1]" ), dret );
	ASSERT( dret == 2.2, "test set1");

	// Check that there is only one DynamicFinfo set up when looking at
	// the same array index.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 2, "Checking formation of DynamicFinfos" );

	get< double >( a1, a1->findFinfo( "dvec[4]" ), dret );
	ASSERT( dret == 0.0, "test get4");
	set< double >( a1, a1->findFinfo( "dvec[4]" ), 4.4 );
	dret = 1234.567;
	get< double >( a1, a1->findFinfo( "dvec[4]" ), dret );
	ASSERT( dret == 4.4, "test set4");

	// Check that there is only one DynamicFinfo set up when looking at
	// the same array index.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 3, "Checking formation of DynamicFinfos" );

	////////////////////////////////////////////////////////////////
	// Now we start testing messages between ArrayFinfo fields.
	////////////////////////////////////////////////////////////////
	
	cout << "\nTesting arrayFinfo messaging";
	Element* a2 = arraytestclass.create( "a2" );

	// We will follow a1 messages to call proc on a2. Check a2->dval.
	//
	// proc on a2 will send this value of dval to a1->dvec[0]. Check it.
	//
	// a1 trigger message will call send on a2->dvec[1]. This goes
	//   to a1->dval. Check it.
	//
	// a1 trigger message will call send on a2->dvec[2]. This goes
	//   to a1->dvec[2]. The trigger is created first. Check it.
	//
	// a1 trigger message will call send on a2->dvec[3]. This goes
	//   to a1->dvec[3]. The send is created first. Check it.

	// 1. We will follow a1 messages to call proc on a2. Check a2->dval.
	ASSERT( a1->findFinfo( "procout" )->
			add( a1, a2, a2->findFinfo( "proc" ) ),
			"adding procout to proc"
			);

	// 2. proc on a2 will send this value of dval to a1->dvec[0].
	ASSERT( 
		a2->findFinfo( "dsumout" )->
			add( a2, a1, a1->findFinfo( "dvec[0]" ) ),
			"Adding dsumout to dval"
		);
	// We have already made a finfo for a1->dvec[0]. Check that this
	// is the one that is used for the messaging.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 3, "reuse of DynamicFinfos" );

	// 3. a1 trigger message will call send on a2->dvec[1]. This goes
	//   to a1->dval.
	ASSERT( 
		a1->findFinfo( "procout" )->
			add( a1, a2, a2->findFinfo( "dvec[1]" ) ),
			"Adding procout to dvec[1]"
		);
	ASSERT( 
		a2->findFinfo( "dvec[1]" )->
			add( a2, a1, a1->findFinfo( "dval" ) ),
			"Adding dvec[1] to dval"
		);
	// Here we made a new DynamicFinfo for the regular ValueFinfo.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 4, "No new DynamicFinfos." );

	// 4. a1 trigger message will call send on a2->dvec[2]. This goes
	//   to a1->dvec[2]. The trigger is created first. Check it.
	ASSERT( 
		a1->findFinfo( "procout" )->
			add( a1, a2, a2->findFinfo( "dvec[2]" ) ),
			"Adding procout to dvec[2]"
		);
	ASSERT( 
		a2->findFinfo( "dvec[2]" )->
			add( a2, a1, a1->findFinfo( "dvec[2]" ) ),
			"Adding dvec[2] to dvec[2] after trigger"
		);
	// We have not made a finfo for a1->dvec[2]. Check that this
	// new one is used for the messaging.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 5, "New DynamicFinfo for dvec[2]" );

	// 5. a1 trigger message will call send on a2->dvec[3]. This goes
	//   to a1->dvec[3]. The send is created first. Check it.
	ASSERT( 
		a2->findFinfo( "dvec[3]" )->
			add( a2, a1, a1->findFinfo( "dvec[3]" ) ),
			"Adding dvec[3] to dvec[3] before trigger"
		);
	ASSERT( 
		a1->findFinfo( "procout" )->
			add( a1, a2, a2->findFinfo( "dvec[3]" ) ),
			"Adding procout to dvec[3]"
		);
	// We have not made a finfo for a1->dvec[3]. Check that this
	// new one is used for the messaging.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 6, "New DynamicFinfo for dvec[3]" );

	unsigned int procOutSlotIndex = arraytestclass.getSlotIndex( "procout" );

	send0( a1, procOutSlotIndex ); // procout
	// Here a2->dval should simply become the sum of its array entries.
	// As this has just been initialized, the sum should be 1.0.
	// Bad Upi: should never test for equality of doubles.
	get< double >( a2, a2->findFinfo( "dval" ), dret );
	ASSERT( dret == 1.0, "test msg1");

	// proc on a2 will send this value of dval to a1->dvec[0]. Check it.
	dret = 0.0;
	get< double >( a1, a1->findFinfo( "dvec[0]" ), dret );
	ASSERT( dret == 1.0, "test msg2");

	// a1 trigger message will call send on a2->dvec[1]. This goes
	//   to a1->dval. Check it.
	dret = 0.0;
	get< double >( a1, a1->findFinfo( "dval" ), dret );
	ASSERT( dret == 0.2, "test msg3");

	// a1 trigger message will call send on a2->dvec[2]. This goes
	//   to a1->dvec[2]. The trigger is created first. Check it.
	dret = 0.0;
	get< double >( a1, a1->findFinfo( "dvec[2]" ), dret );
	ASSERT( dret == 0.3, "test msg4");

	// a1 trigger message will call send on a2->dvec[3]. This goes
	//   to a1->dvec[3]. The send is created first. Check it.
	dret = 0.0;
	get< double >( a1, a1->findFinfo( "dvec[3]" ), dret );
	ASSERT( dret == 0.4, "test msg5");

	// Check that there are no strange things happening with the
	// Finfos when the messaging is actually used.
	flist.resize( 0 );
	a1->listFinfos( flist );
	ASSERT ( flist.size() - s == 6, "Same DynamicFinfo for dvec[3]" );
}
*/

class TestTransientFinfo: public DynamicFinfo
{
	public:
			static int numInstances;

			TestTransientFinfo( Finfo* temp )
					: DynamicFinfo( "foo", temp, 0, 0, 0, 0, 0)
			{
					numInstances++;
			}

			~TestTransientFinfo() {
					numInstances--;
			}
};

int TestTransientFinfo::numInstances = 0;

void transientFinfoDeletionTest()
{
	SimpleElement *e1 = new SimpleElement( "e1" );
	Finfo* temp = new DestFinfo( "temp", Ftype1< double >::global(), 
					0, 0 );


	cout << "\nTesting transientFinfo and element deletion";

	ASSERT( TestTransientFinfo::numInstances == 0, "num = 0");
	e1->addFinfo( new TestTransientFinfo( temp ) );
	ASSERT( TestTransientFinfo::numInstances == 1, "num = 1");
	e1->addFinfo( new TestTransientFinfo( temp ) );
	ASSERT( TestTransientFinfo::numInstances == 2, "num = 2");

	delete e1;
	ASSERT( TestTransientFinfo::numInstances == 0, "num = 0");
}

#endif
