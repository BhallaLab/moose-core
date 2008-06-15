/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Messaging Object Oriented Simulation Environment.
 **           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser General Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

#include "moose.h"
#include "../element/Neutral.h"
#include <queue>
#include "SynInfo.h"
#include "RateLookup.h"
#include "HSolveStruct.h"
#include "NeuroHub.h"
#include "NeuroScanBase.h"
#include "NeuroScan.h"
#include "HSolveBase.h"
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
		new DestFinfo( "solveInit",
			Ftype2< Id, double >::global(),
			RFCAST( &HSolve::initFunc ) ),
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
		new SrcFinfo( "readModel",
			Ftype2< Id, double >::global() ),
	//////////////////////////////////////////////////////////////////
	// MsgDest definitions
	//////////////////////////////////////////////////////////////////
		new DestFinfo( "scanCreate", Ftype0::global(),
			&HSolve::scanCreateFunc ),
	//////////////////////////////////////////////////////////////////
	// Shared definitions
	//////////////////////////////////////////////////////////////////
		new SharedFinfo( "cell-solve", cellShared,
			sizeof( cellShared ) / sizeof( Finfo* ) ),
		process,
	};
	
	static SchedInfo schedInfo[] = { { process, 0, 0 } };
	
        static Cinfo hsolveCinfo(
		"HSolve",
		"Niraj Dudani, 2007, NCBS",
		"HSolve: Hines solver, for solving branching neuron models.",
		initNeutralCinfo(),
		hsolveFinfos,
		sizeof( hsolveFinfos ) / sizeof( Finfo* ),
		ValueFtype1< HSolve >::global(),
		schedInfo, 1
	);
	
	return &hsolveCinfo;
}

static const Cinfo* hsolveCinfo = initHSolveCinfo();

static const Slot readModelSlot =
	initHSolveCinfo()->getSlot( "readModel" );
static const Slot comptListSlot =
	initHSolveCinfo()->getSlot( "cell-solve.comptList" );

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

string HSolve::getPath( Eref e )
{
	return static_cast< const HSolve* >( e.data() )->path_;
}

void HSolve::setVDiv( const Conn* c, int vDiv )
{
	HSolve* solve = static_cast< HSolve* >( c->data() );
	set< int >( solve->scanElm_, "vDiv", vDiv );
}

int HSolve::getVDiv( Eref e )
{
	int vDiv;
	HSolve* solve = static_cast< HSolve* >( e.data() );
	get< int >( solve->scanElm_, "vDiv", vDiv );
	return vDiv;
}

void HSolve::setVMin( const Conn* c, double vMin )
{
	HSolve* solve = static_cast< HSolve* >( c->data() );
	set< double >( solve->scanElm_, "vMin", vMin );
}

double HSolve::getVMin( Eref e )
{
	double vMin;
	HSolve* solve = static_cast< HSolve* >( e.data() );
	get< double >( solve->scanElm_, "vMin", vMin );
	return vMin;
}

void HSolve::setVMax( const Conn* c, double vMax )
{
	HSolve* solve = static_cast< HSolve* >( c->data() );
	set< double >( solve->scanElm_, "vMax", vMax );
}

double HSolve::getVMax( Eref e )
{
	double vMax;
	HSolve* solve = static_cast< HSolve* >( e.data() );
	get< double >( solve->scanElm_, "vMax", vMax );
	return vMax;
}

int HSolve::getCaDiv( Eref e )
{
	int caDiv;
	HSolve* solve = static_cast< HSolve* >( e.data() );
	get< int >( solve->scanElm_, "vDiv", caDiv );
	return caDiv;
}

void HSolve::setCaDiv( const Conn* c, int caDiv )
{
	HSolve* solve = static_cast< HSolve* >( c->data() );
	set< int >( solve->scanElm_, "vDiv", caDiv );
}

void HSolve::setCaMin( const Conn* c, double caMin )
{
	HSolve* solve = static_cast< HSolve* >( c->data() );
	set< double >( solve->scanElm_, "vMin", caMin );
}

double HSolve::getCaMin( Eref e )
{
	double caMin;
	HSolve* solve = static_cast< HSolve* >( e.data() );
	get< double >( solve->scanElm_, "vMin", caMin );
	return caMin;
}

void HSolve::setCaMax( const Conn* c, double caMax )
{
	HSolve* solve = static_cast< HSolve* >( c->data() );
	set< double >( solve->scanElm_, "vMax", caMax );
}

double HSolve::getCaMax( Eref e )
{
	double caMax;
	HSolve* solve = static_cast< HSolve* >( e.data() );
	get< double >( solve->scanElm_, "vMax", caMax );
	return caMax;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void HSolve::processFunc( const Conn*c, ProcInfo p )
{
	static_cast< HSolve* >( c->data() )->
		step( p );
}

void HSolve::initFunc( const Conn* c, Id seed, double dt )
{
	static_cast< HSolve* >( c->data() )->
		innerInitFunc( c->target(), seed, dt );
}

void HSolve::innerInitFunc( Eref solve, Id seed, double dt )
{
	path_ = seed.path();
	send2< Id, double >( solve, readModelSlot, seed, dt );
}

void HSolve::scanCreateFunc( const Conn* c )
{
	static_cast< HSolve* >( c->data() )->
		innerScanCreateFunc( c->target() );
}

void HSolve::innerScanCreateFunc( Eref integ )
{
	Id solve = Neutral::getParent( integ );
	// Scan element's data field is owned by its parent HSolve
	// structure, so we set it's noDelFlag to 1.
	scanElm_ = initNeuroScanCinfo()->create( 
		Id::scratchId(), "scan",
		static_cast< void* >( &scanData_ ), 1 );
	
	Eref( solve() ).add( "childSrc", scanElm_, "child" );
	Eref( integ ).add( "readModel", scanElm_, "readModel" );
}
