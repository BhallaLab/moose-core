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
			Ftype2< const Element*, double >::global(),
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
		new ValueFinfo( "NDiv", ValueFtype1< int >::global(),
			GFCAST( &HSolve::getNDiv ),
			RFCAST( &HSolve::setNDiv )
		),
		new ValueFinfo( "VLo", ValueFtype1< double >::global(),
			GFCAST( &HSolve::getVLo ),
			RFCAST( &HSolve::setVLo )
		),
		new ValueFinfo( "VHi", ValueFtype1< double >::global(),
			GFCAST( &HSolve::getVHi ),
			RFCAST( &HSolve::setVHi )
		),
	//////////////////////////////////////////////////////////////////
	// MsgSrc definitions
	//////////////////////////////////////////////////////////////////
		new SrcFinfo( "readModel",
			Ftype2< Element*, double >::global() ),
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

static const unsigned int readModelSlot =
	initHSolveCinfo()->getSlotIndex( "readModel" );
static const unsigned int comptListSlot =
	initHSolveCinfo()->getSlotIndex( "cell-solve.comptList" );

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

string HSolve::getPath( const Element* e )
{
	return static_cast< const HSolve* >( e->data() )->path_;
}

/** Lookup table specifics (NDiv, VLo, VHi) actually are fields on NeuroScan.
 *  Here we provide port-holes to access the same.
 */
void HSolve::setNDiv( const Conn& c, int NDiv )
{
	HSolve* solve = static_cast< HSolve* >( c.data() );
	set< int >( solve->scanElm_, "NDiv", NDiv );
}

int HSolve::getNDiv( const Element* e )
{
	int NDiv;
	HSolve* solve = static_cast< HSolve* >( e->data() );
	get< int >( solve->scanElm_, "NDiv", NDiv );
	return NDiv;
}

/** Lookup table specifics (NDiv, VLo, VHi) actually are fields on NeuroScan.
 *  Here we provide port-holes to access the same.
 */
void HSolve::setVLo( const Conn& c, double VLo )
{
	HSolve* solve = static_cast< HSolve* >( c.data() );
	set< double >( solve->scanElm_, "VLo", VLo );
}

double HSolve::getVLo( const Element* e )
{
	double VLo;
	HSolve* solve = static_cast< HSolve* >( e->data() );
	get< double >( solve->scanElm_, "VLo", VLo );
	return VLo;
}

void HSolve::setVHi( const Conn& c, double VHi )
{
	HSolve* solve = static_cast< HSolve* >( c.data() );
	set< double >( solve->scanElm_, "VHi", VHi );
}

double HSolve::getVHi( const Element* e )
{
	double VHi;
	HSolve* solve = static_cast< HSolve* >( e->data() );
	get< double >( solve->scanElm_, "VHi", VHi );
	return VHi;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void HSolve::processFunc( const Conn&c, ProcInfo p )
{
	static_cast< HSolve* >( c.data() )->
		step( p );
}

void HSolve::initFunc( const Conn& c,
	const Element* seed, double dt )
{
	Element* e = c.targetElement();
	static_cast< HSolve* >( c.data() )->
		innerInitFunc( e, seed, dt );
}

void HSolve::innerInitFunc( Element* solve,
	const Element* seed, double dt )
{
	path_ = seed->id().path();
	send2< const Element*, double >( solve, readModelSlot, seed, dt );
}

void HSolve::scanCreateFunc( const Conn& c )
{
	static_cast< HSolve* >( c.data() )->
		innerScanCreateFunc( c.targetElement() );
}

void HSolve::innerScanCreateFunc( Element* integ )
{
	Id solve = Neutral::getParent( integ );
	// Scan element's data field is owned by its parent HSolve
	// structure, so we set it's noDelFlag to 1.
	scanElm_ = initNeuroScanCinfo()->create( 
		Id::scratchId(), "scan",
		static_cast< void* >( &scanData_ ), 1 );
	bool ret = solve()->findFinfo( "childSrc" )->
		add( solve(), scanElm_, scanElm_->findFinfo( "child" ) );
	assert( ret );
	
	ret = integ->findFinfo( "readModel" )->
		add( integ, scanElm_, scanElm_->findFinfo( "readModel" ) );
	assert( ret );
}
