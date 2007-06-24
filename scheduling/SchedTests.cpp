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
#include "../shell/Shell.h"

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

//////////////////////////////////////////////////////////////////////
// Here we check how the system handles process calls to multiple
// target types.
//////////////////////////////////////////////////////////////////////

static string procSeq1[] = {
	"Process0: s0_0 at 0",
	"Process0: s0_1 at 0",
	"Process1: s1 at 0",
	"Process2: s2 at 0",
	"Process0: s0_0 at 1",
	"Process0: s0_1 at 1",
	"Process1: s1 at 1",
	"Process2: s2 at 1",
	"Process0: s0_0 at 2",
	"Process0: s0_1 at 2",
	"Process1: s1 at 2",
	"Process2: s2 at 2",
	"Process0: s0_0 at 3",
	"Process0: s0_1 at 3",
	"Process1: s1 at 3",
	"Process2: s2 at 3",
};

static string reinitSeq[] = {
	"Reinit: s0_0",
	"Reinit: s0_1",
	"Reinit: s1",
	"Reinit: s2",
};

static string* seqStr;
static unsigned int seqCount;

void processCall0( const Conn& c, ProcInfo p )
{
	char line[40];
	sprintf( line, "Process0: %s at %g",
			c.targetElement()->name().c_str(), p->currTime_ );
	ASSERT( seqStr[ seqCount++ ] == line , "process0" );
}

void processCall1( const Conn& c, ProcInfo p )
{
	char line[40];
	sprintf( line, "Process1: %s at %g",
			c.targetElement()->name().c_str(), p->currTime_ );
	ASSERT( seqStr[ seqCount++ ] == line , "process1" );
}

void processCall2( const Conn& c, ProcInfo p )
{
	char line[40];
	sprintf( line, "Process2: %s at %g",
			c.targetElement()->name().c_str(), p->currTime_ );
	ASSERT( seqStr[ seqCount++ ] == line , "process2" );
}

void reinitCall( const Conn& c, ProcInfo p )
{
	char line[40];
	sprintf( line, "Reinit: %s", c.targetElement()->name().c_str() );
	ASSERT( seqStr[ seqCount++ ] == line , "reinit" );
}

void testSchedProcess()
{
	static TypeFuncPair processTypes0[] =
	{
			// The process func call
		TypeFuncPair( Ftype1< ProcInfo >::global(), 
				RFCAST( processCall0 ) ),
		TypeFuncPair( Ftype1< ProcInfo >::global(), 
				RFCAST( reinitCall ) ),
	};

	static Finfo* sched0Finfos[] =
	{
		new SharedFinfo( "process", processTypes0, 2 ),
	};

	static Cinfo sched0Cinfo(
		"Sched0",
		"Upinder S. Bhalla, Mar 2007, NCBS",
		"Sched0: Checks scheduling of multiple target types",
		initNeutralCinfo(),
		sched0Finfos,
		sizeof(sched0Finfos)/sizeof(Finfo *),
		ValueFtype1< double >::global()
	);


	static TypeFuncPair processTypes1[] =
	{
			// The process func call
		TypeFuncPair( Ftype1< ProcInfo >::global(), 
				RFCAST( processCall1 ) ),
		TypeFuncPair( Ftype1< ProcInfo >::global(), 
				RFCAST( reinitCall ) ),
	};

	static Finfo* sched1Finfos[] =
	{
		new SharedFinfo( "process", processTypes1, 2 ),
	};

	static Cinfo sched1Cinfo(
		"Sched1",
		"Upinder S. Bhalla, Mar 2007, NCBS",
		"Sched1: Checks scheduling of multiple target types",
		initNeutralCinfo(),
		sched1Finfos,
		sizeof(sched1Finfos)/sizeof(Finfo *),
		ValueFtype1< double >::global()
	);
	static TypeFuncPair processTypes2[] =
	{
			// The process func call
		TypeFuncPair( Ftype1< ProcInfo >::global(), 
				RFCAST( processCall2 ) ),
		TypeFuncPair( Ftype1< ProcInfo >::global(), 
				RFCAST( reinitCall ) ),
	};

	static Finfo* sched2Finfos[] =
	{
		new SharedFinfo( "process", processTypes2, 2 ),
	};

	static Cinfo sched2Cinfo(
		"Sched2",
		"Upinder S. Bhalla, Mar 2007, NCBS",
		"Sched2: Checks scheduling of multiple target types",
		initNeutralCinfo(),
		sched2Finfos,
		sizeof(sched2Finfos)/sizeof(Finfo *),
		ValueFtype1< double >::global()
	);

	cout << "\nTesting sched process sequencing";
	Element* n = Neutral::create( "Neutral", "n", Element::root() );
	Element* cj = Neutral::create( "ClockJob", "cj", n );
	Element* t0 = Neutral::create( "Tick", "t0", cj );
	Element* s0_0 = Neutral::create( "Sched0", "s0_0", n );
	Element* s0_1 = Neutral::create( "Sched0", "s0_1", n );
	Element* s1 = Neutral::create( "Sched1", "s1", n );
	Element* s2 = Neutral::create( "Sched2", "s2", n );

	const Finfo* proc = t0->findFinfo( "process" );
	proc->add( t0, s0_0, s0_0->findFinfo( "process" ) );
	proc->add( t0, s0_1, s0_1->findFinfo( "process" ) );
	proc->add( t0, s1, s1->findFinfo( "process" ) );
	proc->add( t0, s2, s2->findFinfo( "process" ) );

	seqStr = procSeq1;
	seqCount = 0;
	set( cj, "resched" );
	set< double >( cj, "start", 3.0 );
	ASSERT( seqCount == 16, "sequencing" );

	ASSERT( set( n, "destroy" ), "cleanup" );

	cout << "\nTesting Shell-based scheduling commands";

	n = Neutral::create( "Neutral", "n", Element::root() );
	s0_0 = Neutral::create( "Sched0", "s0_0", n );
	s0_1 = Neutral::create( "Sched0", "s0_1", n );
	s1 = Neutral::create( "Sched1", "s1", n );
	s2 = Neutral::create( "Sched2", "s2", n );

	Element* shell = Neutral::create( "Shell", "shell",
					Element::root() );
	ASSERT( shell != 0 , "shell creation");
	Conn c( shell, 0 );
	vector< Id > path;
	path.push_back( s0_0->id() );
	path.push_back( s0_1->id() );
	path.push_back( s1->id() );
	path.push_back( s2->id() );
			
	Id schedId = Neutral::getChildByName( Element::root(), "sched" );
	ASSERT( !schedId.zero() && !schedId.bad(), "find sched" );
	Id cjId = Neutral::getChildByName( schedId(), "cj" );
	ASSERT( !cjId.zero() != 0 && !cjId.bad(), "find cjId" );
	Id t0Id = Neutral::getChildByName( cjId(), "t0" );
	ASSERT( !t0Id.zero() && !t0Id.bad(), "find t0Id" );

	Shell::useClock( c, t0Id, path, string( "process" ) );
	Shell::resched( c );
	seqStr = reinitSeq;
	seqCount = 0;
	Shell::reinit( c );
	ASSERT( seqCount == 4, "sequencing" );

	seqStr = procSeq1;
	seqCount = 0;
	Shell::step( c, 3.0 );

	ASSERT( seqCount == 16, "sequencing" );
	ASSERT( set( n, "destroy" ), "cleanup" );
	ASSERT( set( shell, "destroy" ), "cleanup" );
}

#endif
