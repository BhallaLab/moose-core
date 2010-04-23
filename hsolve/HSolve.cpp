/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "element/Neutral.h"
#include "HSolveStruct.h"
#include "HinesMatrix.h"
#include "HSolvePassive.h"
#include "RateLookup.h"
#include "HSolveActive.h"
#include "HSolve.h"

const Cinfo* initHSolveCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &HSolve::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			dummyFunc ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ) );
	
	static Finfo* cellShared[] =
	{
		new DestFinfo( "integSetup",
			Ftype2< Id, double >::global(),
			RFCAST( &HSolve::setupFunc ) ),
		new SrcFinfo( "comptList",
			Ftype1< const vector< Id >* >::global() ),
	};
	
	static Finfo* hsolveFinfos[] = 
	{
	//////////////////////////////////////////////////////////////////
	// Field definitions
	//////////////////////////////////////////////////////////////////
		new ValueFinfo( "path", ValueFtype1< string >::global(),
			GFCAST( &HSolve::getPath ),
			dummyFunc
		),
		new ValueFinfo( "CaAdvance", ValueFtype1< int >::global(),
			GFCAST( &HSolve::getCaAdvance ),
			RFCAST( &HSolve::setCaAdvance )
		),		
		new ValueFinfo( "VDiv", ValueFtype1< int >::global(),
			GFCAST( &HSolve::getVDiv ),
			RFCAST( &HSolve::setVDiv )
		),
		new ValueFinfo( "VMin", ValueFtype1< double >::global(),
			GFCAST( &HSolve::getVMin ),
			RFCAST( &HSolve::setVMin )
		),
		new ValueFinfo( "VMax", ValueFtype1< double >::global(),
			GFCAST( &HSolve::getVMax ),
			RFCAST( &HSolve::setVMax )
		),
		new ValueFinfo( "CaDiv", ValueFtype1< int >::global(),
			GFCAST( &HSolve::getCaDiv ),
			RFCAST( &HSolve::setCaDiv )
		),
		new ValueFinfo( "CaMin", ValueFtype1< double >::global(),
			GFCAST( &HSolve::getCaMin ),
			RFCAST( &HSolve::setCaMin )
		),
		new ValueFinfo( "CaMax", ValueFtype1< double >::global(),
			GFCAST( &HSolve::getCaMax ),
			RFCAST( &HSolve::setCaMax )
		),
	//////////////////////////////////////////////////////////////////
	// MsgSrc definitions
	//////////////////////////////////////////////////////////////////
		new SrcFinfo( "integ-hub",
			Ftype1< HSolveActive* >::global() ),
	
	//////////////////////////////////////////////////////////////////
	// MsgDest definitions
	//////////////////////////////////////////////////////////////////
	
	//////////////////////////////////////////////////////////////////
	// Shared definitions
	//////////////////////////////////////////////////////////////////
		new SharedFinfo( "cell-integ", cellShared,
			sizeof( cellShared ) / sizeof( Finfo* ),
			"Shared message from Cell" ),
		process,
	};
	
	static SchedInfo schedInfo[] = { { process, 0, 2 } };
	
	static string doc[] =
	{
		"Name", "HSolve",
		"Author", "Niraj Dudani, 2007, NCBS",
		"Description", "HSolve: Hines solver, for solving branching neuron models.",
	};
	static Cinfo hsolveCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		hsolveFinfos,
		sizeof( hsolveFinfos ) / sizeof( Finfo* ),
		ValueFtype1< HSolve >::global(),
		schedInfo, 1
	);
	
	return &hsolveCinfo;
}

static const Cinfo* hsolveCinfo = initHSolveCinfo();

static const Slot comptListSlot =
	initHSolveCinfo()->getSlot( "cell-integ.comptList" );
static const Slot hubSlot =
	initHSolveCinfo()->getSlot( "integ-hub" );

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void HSolve::processFunc( const Conn*c, ProcInfo p )
{
	static_cast< HSolve* >( c->data() )->solve( p );
}

void HSolve::setupFunc( const Conn* c, Id seed, double dt )
{
	static_cast< HSolve* >( c->data() )->
		setup( c->target(), seed, dt );
}

void HSolve::setup( Eref integ, Id seed, double dt )
{
	// Set internal field to point to seed's path.
	path_ = seed.path();
	
	// Setup solver.
	this->HSolveActive::setup( seed, dt );
	
	// Setup hub.
	setupHub( integ );
}

void HSolve::setupHub( Eref integ )
{
	// Create Hub, and link self with it.
	Id solve = Neutral::getParent( integ );
	Element* hub = Neutral::create( "HSolveHub", "hub", solve, Id::scratchId() );
	assert( hub != 0 );
	bool ret = Eref( integ ).add( "integ-hub", hub, "integ-hub" );
	assert( ret );
	
	send1< HSolveActive* >(
		integ, hubSlot, dynamic_cast< HSolveActive* >( this ) );
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

string HSolve::getPath( Eref e )
{
	return static_cast< const HSolve* >( e.data() )->path_;
}

void HSolve::setCaAdvance( const Conn* c, int value )
{
	if ( value != 0 && value != 1 ) {
		cout << "Error: HSolve: caAdvance should be either 0 or 1.\n";
		return;
	}
	
	static_cast< HSolve* >( c->data() )->caAdvance_ = value;
}

int HSolve::getCaAdvance( Eref e )
{
	return static_cast< const HSolve* >( e.data() )->caAdvance_;
}

void HSolve::setVDiv( const Conn* c, int vDiv )
{
	static_cast< HSolve* >( c->data() )->vDiv_ = vDiv;
}

int HSolve::getVDiv( Eref e )
{
	return static_cast< const HSolve* >( e.data() )->vDiv_;
}

void HSolve::setVMin( const Conn* c, double vMin )
{
	static_cast< HSolve* >( c->data() )->vMin_ = vMin;
}

double HSolve::getVMin( Eref e )
{
	return static_cast< const HSolve* >( e.data() )->vMin_;
}

void HSolve::setVMax( const Conn* c, double vMax )
{
	static_cast< HSolve* >( c->data() )->vMax_ = vMax;
}

double HSolve::getVMax( Eref e )
{
	return static_cast< const HSolve* >( e.data() )->vMax_;
}

void HSolve::setCaDiv( const Conn* c, int caDiv )
{
	static_cast< HSolve* >( c->data() )->caDiv_ = caDiv;
}

int HSolve::getCaDiv( Eref e )
{
	return static_cast< const HSolve* >( e.data() )->caDiv_;
}

void HSolve::setCaMin( const Conn* c, double caMin )
{
	static_cast< HSolve* >( c->data() )->caMin_ = caMin;
}

double HSolve::getCaMin( Eref e )
{
	return static_cast< const HSolve* >( e.data() )->caMin_;
}

void HSolve::setCaMax( const Conn* c, double caMax )
{
	static_cast< HSolve* >( c->data() )->caMax_ = caMax;
}

double HSolve::getCaMax( Eref e )
{
	return static_cast< const HSolve* >( e.data() )->caMax_;
}
