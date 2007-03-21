/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifdef DO_UNIT_TESTS

#include "moose.h"
#include "../element/Neutral.h"

/**
 * Unit tests for the scheduling system.
 */

class TickTest {
	public:
		TickTest()
				{;}

		static void process( const Conn& c, ProcInfo p );
		static void reinit( const Conn& c, ProcInfo p );
		static unsigned int count_;
		static unsigned int countReinit_;
	private:
};

unsigned int TickTest::count_ = 0;
unsigned int TickTest::countReinit_ = 0;

void TickTest::process( const Conn& c, ProcInfo p )
{
	static const string schedResponse[] = {
			"t1a  t = 0, dt = 1",
			"t1b  t = 0, dt = 1",
			"t2   t = 0, dt = 2",
			"t5   t = 0, dt = 5",
			"t1a  t = 1, dt = 1",
			"t1b  t = 1, dt = 1",
			"t1a  t = 2, dt = 1",
			"t1b  t = 2, dt = 1",
			"t2   t = 2, dt = 2",
			"t1a  t = 3, dt = 1",
			"t1b  t = 3, dt = 1",
			"t1a  t = 4, dt = 1",
			"t1b  t = 4, dt = 1",
			"t2   t = 4, dt = 2",
			"t1a  t = 5, dt = 1",
			"t1b  t = 5, dt = 1",
			"t5   t = 5, dt = 5",
			"t1a  t = 6, dt = 1",
			"t1b  t = 6, dt = 1",
			"t2   t = 6, dt = 2",
			"t1a  t = 7, dt = 1",
			"t1b  t = 7, dt = 1",
			"t1a  t = 8, dt = 1",
			"t1b  t = 8, dt = 1",
			"t2   t = 8, dt = 2",
			"t1a  t = 9, dt = 1",
			"t1b  t = 9, dt = 1",
			"t1a  t = 10, dt = 1",
			"t1b  t = 10, dt = 1",
			"t2   t = 10, dt = 2",
			"t5   t = 10, dt = 5",
	};
	char line[200];
	sprintf( line, "%-4s t = %g, dt = %g",
		c.sourceElement()->name().c_str(), p->currTime_, p->dt_ );
	if ( count_ < sizeof( schedResponse )/sizeof( string ) )
		ASSERT( schedResponse[ count_++ ] == string( line ), line );
	countReinit_ = 0;
}

void TickTest::reinit( const Conn& c, ProcInfo p )
{
	static const string schedResponse[] = {
			"t1a  t = 0, dt = 1, reinit",
			"t1b  t = 0, dt = 1, reinit",
			"t2   t = 0, dt = 2, reinit",
			"t5   t = 0, dt = 5, reinit",
	};
	char line[200];
	sprintf( line, "%-4s t = %g, dt = %g, reinit",
		c.sourceElement()->name().c_str(), p->currTime_, p->dt_ );
	if ( countReinit_ < sizeof( schedResponse )/sizeof( string ) )
		ASSERT( schedResponse[ countReinit_++ ] == string( line ),
						line );
	count_ = 0;
}

void testSched()
{
	static TypeFuncPair processTypes[] =
	{
			// The process func call
		TypeFuncPair( Ftype1< ProcInfo >::global(), 
				RFCAST( TickTest::process ) ),
		TypeFuncPair( Ftype1< ProcInfo >::global(), 
				RFCAST( TickTest::reinit ) ),
	};

	static Finfo* tickTestFinfos[] =
	{
		new SharedFinfo( "process", processTypes, 2 ),
	};

	static Cinfo tickTestCinfo(
		"TickTest",
		"Upinder S. Bhalla, Mar 2007, NCBS",
		"TickTest: Checks scheduling",
		initNeutralCinfo(),
		tickTestFinfos,
		sizeof(tickTestFinfos)/sizeof(Finfo *),
		ValueFtype1< TickTest >::global()
	);

	cout << "\nTesting sched basic stuff";

	Element* sched =
			Neutral::create( "Neutral", "sched", Element::root() );
	
	Element* cj = Neutral::create( "ClockJob", "cj", sched );
	Element* t1a = Neutral::create( "Tick", "t1a", cj );
	Element* t1b = Neutral::create( "Tick", "t1b", cj );
	Element* t2 = Neutral::create( "Tick", "t2", cj );
	Element* t5 = Neutral::create( "Tick", "t5", cj );

	Element* tt =
			Neutral::create( "TickTest", "tt", Element::root() );

	set< double >( t1a, "dt", 1.0 );
	set< double >( t1b, "dt", 1.0 );
	set< double >( t2, "dt", 2.0 );
	set< double >( t5, "dt", 5.0 );
	set< int >( t1a, "stage", 0 );
	set< int >( t1b, "stage", 1 );

	ASSERT( t1a->findFinfo( "process" )->add( t1a, tt,
				tt->findFinfo( "process" ) ), "connect process" );
	ASSERT( t1b->findFinfo( "process" )->add( t1b, tt,
				tt->findFinfo( "process" ) ), "connect process" );
	ASSERT( t2->findFinfo( "process" )->add( t2, tt,
				tt->findFinfo( "process" ) ), "connect process" );
	ASSERT( t5->findFinfo( "process" )->add( t5, tt,
				tt->findFinfo( "process" ) ), "connect process" );

	ASSERT( set( cj, "resched" ), "resched" );
	ASSERT( set( cj, "reinit" ), "reinit" );
	ASSERT( TickTest::countReinit_ == 4, "reinit cascading" );

	cout << "\nTesting sched with 4 ticks";
	ASSERT( set< double >( cj, "start", 10.0 ), "start" );
	ASSERT( TickTest::count_ == 31, "# of individual tick counts" );
	ASSERT( set( sched, "destroy" ), "cleanup" );
	ASSERT( set( tt, "destroy" ), "cleanup" );
}

#endif
