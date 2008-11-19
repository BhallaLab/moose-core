/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include <math.h>
#include "moose.h"
#include "Average.h"

const Cinfo* initAverageCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &Average::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &Average::reinitFunc ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ) );

	static Finfo* averageFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "total",
			ValueFtype1< double >::global(),
			GFCAST( &Average::getTotal ),
			RFCAST( &Average::setTotal )
		),
		new ValueFinfo( "baseline",
			ValueFtype1< double >::global(),
			GFCAST( &Average::getBaseline ),
			RFCAST( &Average::setBaseline )
		),
		new ValueFinfo( "n",
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Average::getN ),
			RFCAST( &Average::setN )
		),
		new ValueFinfo( "mean",
			ValueFtype1< double >::global(),
			GFCAST( &Average::getMean ),
			&dummyFunc
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		new SrcFinfo( "output", Ftype1< double >::global() ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "input",
			Ftype1< double >::global(),
			RFCAST( &Average::input )
		),
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		process,
	};

	// Schedule molecules for the slower clock, stage 0.
	static SchedInfo schedInfo[] = { { process, 0, 0 } };

	static string doc[] =
	{
		"Name", "Average",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "Average: Example MOOSE class.",
	};

	static Cinfo averageCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		averageFinfos,
		sizeof( averageFinfos )/sizeof(Finfo *),
		ValueFtype1< Average >::global(),
			schedInfo, 1
	);

	return &averageCinfo;
}

static const Cinfo* averageCinfo = initAverageCinfo();

static const Slot outputSlot = initAverageCinfo()->getSlot( "output" );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

Average::Average()
	:
	total_( 0.0 ),
	baseline_( 0.0 ),
	n_( 0 )
{
		;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void Average::setTotal( const Conn* c, double value )
{
	static_cast< Average* >( c->data() )->total_ = value;
}

double Average::getTotal( Eref e )
{
	return static_cast< Average* >( e.data() )->total_;
}

void Average::setBaseline( const Conn* c, double value )
{
	static_cast< Average* >( c->data() )->baseline_ = value;
}

double Average::getBaseline( Eref e )
{
	return static_cast< Average* >( e.data() )->baseline_;
}

void Average::setN( const Conn* c, unsigned int value )
{
	static_cast< Average* >( c->data() )->n_ = value;
}

unsigned int Average::getN( Eref e )
{
	return static_cast< Average* >( e.data() )->n_;
}

double Average::getMean( Eref e )
{
	return static_cast< Average* >( e.data() )->mean();
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void Average::inputLocal( double value )
{
	total_ += value;
	++n_;
}

void Average::input( const Conn* c, double value )
{
	static_cast< Average* >( c->data() )->inputLocal( value );
}


void Average::reinitFunc( const Conn* c, ProcInfo info )
{
	static_cast< Average* >( c->data() )->reinitFuncLocal( );
}
void Average::reinitFuncLocal( )
{
	total_ = 0.0;
	n_ = 0;
}

void Average::processFunc( const Conn* c, ProcInfo info )
{
	static_cast< Average* >( c->data() )->processFuncLocal( c->target(), info );
}

double Average::mean() const
{
	return ( n_ > 0 ) ? baseline_ + total_ / n_ : baseline_;
}

void Average::processFuncLocal( Eref e, ProcInfo info )
{
	send1< double >( e, outputSlot, mean() );
}



#ifdef DO_UNIT_TESTS
#include "../element/Neutral.h"

void testAverage()
{
	static const double EPSILON = 0.001;
	static double check[] = { 0.1, 6.1, 8.3875, 9.5736, 10.3754, 10.9807,
		11.4666, 11.8724, 12.2207,12.5257, 12.797};

	cout << "\nTesting Average" << flush;

	Eref n = Neutral::create( "Neutral", "n", Element::root()->id(),
		Id::scratchId() );
	Element* m0 = Neutral::create( "Average", "m0", n->id(),
		Id::scratchId() );
	ASSERT( m0 != 0, "creating average" );
	Element* m1 = Neutral::create( "Average", "m1", n->id(),
		Id::scratchId() );
	ASSERT( m1 != 0, "creating average" );
	Element* m2 = Neutral::create( "Average", "m2", n->id(),
		Id::scratchId() );
	ASSERT( m2 != 0, "creating average" );

	bool ret;

	ProcInfoBase p;
	SetConn cm0( m0, 0 );
	SetConn cm1( m1, 0 );
	SetConn cm2( m2, 0 );
	p.dt_ = 1.0;
	set< double >( m0, "baseline", 1.0 );
	set< double >( m1, "baseline", 10.0 );
	set< double >( m2, "baseline", 0.1 );

	//ret = m0->findFinfo( "output" )->add( m0, m1, m1->findFinfo( "input" ) );
	ret = Eref(m0).add("output", m1, "input");
	ASSERT( ret, "adding msg 0" );
	//ret = m0->findFinfo( "output" )->add( m0, m2, m2->findFinfo( "input" ) );
	ret = Eref(m0).add("output", m2, "input");
	ASSERT( ret, "adding msg 1" );
	//ret = m1->findFinfo( "output" )->add( m1, m2, m2->findFinfo( "input" ) );
	ret = Eref(m1).add("output", m2, "input");
	ASSERT( ret, "adding msg 2" );

	//ret = m2->findFinfo( "output" )->add( m2, m0, m0->findFinfo( "input" ) );
	ret = Eref(m2).add("output", m0, "input");
	ASSERT( ret, "adding msg 3" );

	Average::reinitFunc( &cm0, &p );
	Average::reinitFunc( &cm1, &p );
	Average::reinitFunc( &cm2, &p );

	unsigned int i = 0;
	for ( p.currTime_ = 0.0; p.currTime_ < 10.0; p.currTime_ += p.dt_ )
	{
//		double n0 = Average::getMean( m0 );
//		double n1 = Average::getMean( m1 );
		double n2 = Average::getMean( m2 );
		Average::processFunc( &cm0, &p );
		Average::processFunc( &cm1, &p );
		Average::processFunc( &cm2, &p );
//		cout << p.currTime_ << "	" << n0 << "	" << n1 << "	" << n2 << "	" << check[i] << endl;
		ASSERT( fabs ( n2 - check[ i ] ) < EPSILON, "testing example/average values" );
		i++;
	}
	// Get rid of all the compartments.
	set( n, "destroy" );
}
#endif
