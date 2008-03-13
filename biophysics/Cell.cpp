/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Messaging Object Oriented Simulation Environment.
 **   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser General Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

/*
 * 24 December 2007
 * Updating Subhasis' Cell class to manage automatic solver setup.
 * Niraj Dudani
 */

/*******************************************************************
 * File:            Cell.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-02 13:38:29
 ********************************************************************/

#include "moose.h"
#include "../element/Neutral.h"
#include "Cell.h"

map< string, MethodInfo > Cell::methodMap_;

const Cinfo* initCellCinfo()
{
	static Finfo* processShared[] =
	{
		/* Cell does not process at simulation time--it only sets up
		   the solver at reset */
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			dummyFunc ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &Cell::reinitFunc ) ),
	};
	
	static Finfo* solveShared[] =
	{
		new SrcFinfo( "solveInit",
			Ftype2< const Element*, double >::global() ),
		// Placeholder for receiving list of compartments from solver.
		new DestFinfo( "comptList",
			Ftype1< const vector< Id >* >::global(),
			RFCAST( &dummyFunc ) ),
	};
	
	static Finfo* process = new SharedFinfo( "process", processShared,
			sizeof( processShared ) / sizeof( Finfo* ) );
	
	static Finfo* cellFinfos[] = 
	{
	//////////////////////////////////////////////////////////////////
	// Field definitions
	//////////////////////////////////////////////////////////////////
		new ValueFinfo( "method", 
			ValueFtype1< string >::global(),
			GFCAST( &Cell::getMethod ), 
			RFCAST( &Cell::setMethod )
		),
		new ValueFinfo( "variableDt", 
			ValueFtype1< bool >::global(),
			GFCAST( &Cell::getVariableDt ), 
			dummyFunc
		),
		new ValueFinfo( "implicit", 
			ValueFtype1< bool >::global(),
			GFCAST( &Cell::getImplicit ), 
			dummyFunc
		),
		new ValueFinfo( "description", 
			ValueFtype1< string >::global(),
			GFCAST( &Cell::getDescription ), 
			dummyFunc
		),
	//////////////////////////////////////////////////////////////////
	// MsgSrc definitions
	//////////////////////////////////////////////////////////////////
	
	//////////////////////////////////////////////////////////////////
	// MsgDest definitions
	//////////////////////////////////////////////////////////////////
	
	//////////////////////////////////////////////////////////////////
	// Shared definitions
	//////////////////////////////////////////////////////////////////
		new SharedFinfo( "cell-solve", solveShared,
			sizeof( solveShared ) / sizeof( Finfo* ) ),
		process,
	};
	
	// Clock 0 will do for reset.
	static SchedInfo schedInfo[] = { { process, 0, 0 } };
	
	static Cinfo cellCinfo(
		"Cell",
		"Subhasis Ray, Niraj Dudani, 2007, NCBS",
		"Cell: Container for a neuron's components. Also manages automatic solver setup.",
		initNeutralCinfo(),
		cellFinfos,
		sizeof( cellFinfos ) / sizeof( Finfo* ),
		ValueFtype1< Cell >::global(),
		schedInfo, 1
	);
	
	Cell::addMethod( "ee", 
		"GENESIS Exponential Euler method.",
		0, 0 );
	Cell::addMethod( "hsolve", 
		"Hines' solver.",
		0, 1 );
	
	return &cellCinfo;
}

static const Cinfo* cellCinfo = initCellCinfo();

static const Slot solveInitSlot =
	initCellCinfo()->getSlot( "cell-solve.solveInit" );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

Cell::Cell()
{
	innerSetMethod( "hsolve" );
}

void Cell::addMethod( 
	const string& name, const string& description,
	bool isVariableDt, bool isImplicit )
{
	MethodInfo mi;
	mi.description = description;
	mi.isVariableDt = isVariableDt;
	mi.isImplicit = isImplicit;
	methodMap_[name] = mi;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void Cell::setMethod( const Conn* c, string method )
{
	static_cast< Cell* >( c->data() )->innerSetMethod( method );
}

void Cell::innerSetMethod( string value )
{
	map< string, MethodInfo >::iterator i = methodMap_.find( value );
	if ( i != methodMap_.end() ) {
		method_ = value;
		variableDt_ = i->second.isVariableDt;
		implicit_ = i->second.isImplicit;
		description_ = i->second.description;
	} else {
		method_ = "hsolve";
		cout << "Warning: method '" << value << "' not known. Using '"
		     <<	method_ << "'\n";
		innerSetMethod( method_ );
	}
}

string Cell::getMethod( const Element* e )
{
	return static_cast< Cell* >( e->data() )->method_;
}

bool Cell::getVariableDt( const Element* e )
{
	return static_cast< Cell* >( e->data() )->variableDt_;
}

bool Cell::getImplicit( const Element* e )
{
	return static_cast< Cell* >( e->data() )->implicit_;
}

string Cell::getDescription( const Element* e )
{
	return static_cast< Cell* >( e->data() )->description_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void Cell::reinitFunc( const Conn* c, ProcInfo p )
{
	static_cast< Cell* >( c->data() )->
		innerReinitFunc( c->targetElement()->id() );
}

void Cell::innerReinitFunc( const Id& cell )
{
	if ( method_ == "ee" ) {
		// Delete existing solver
		Id oldSolve( cell.path() + "/solve" );
		if ( oldSolve.good() )
			set( oldSolve(), "destroy" );
		return;
	}
	
	Id seed = findCompt( cell );
	if ( seed.bad() ) // No compartment found.
		return;
	
	// The solver could be set up to send back a list of solved compartments.
	setupSolver( cell, seed );
	
	// The compartment list could then be used to see if the tree below the
	// cell is built correctly. For instance, one could check if there are
	// compartments dangling outside the cell.
	checkTree( );
}

/**
 * This function performs a depth-first search of the tree under the current
 * cell. First compartment found is returned as the seed.
 */ 
Id Cell::findCompt( const Id& cell )
{
	/* 'curr' is the current element under consideration. 'cstack' is a list
	 * of all elements (and their immediate siblings) found on the path from
	 * the root element (the Cell) to the current element.
	 */
	vector< vector< Id > > cstack;
	Id seed;
	const Cinfo* compartment = Cinfo::find( "Compartment" );
	
	cstack.push_back( Neutral::getChildList( cell() ) );
	while ( !cstack.empty() ) {
		vector< Id >& child = cstack.back();
		
		if ( child.empty() ) {
			cstack.pop_back();
			if ( !cstack.empty() )
				cstack.back().pop_back();
		} else {
			const Id& curr = child.back();
			if ( curr()->cinfo()->isA( compartment ) ) {
				seed = curr;
				break;
			}
			cstack.push_back( Neutral::getChildList( curr() ) );
		}
	}
	
	return seed;
}

void Cell::setupSolver( const Id& cell, const Id& seed ) const
{
	// Destroy any existing child called 'solve'.
	Id oldSolve( cell.path() + "/solve" );
	if ( oldSolve.good() )
		set( oldSolve(), "destroy" );
	
	// Create solve, and its children: scan, hub, integ.
	Element* solve = Neutral::create( "Neutral", "solve",
		cell(), Id::scratchId() );
	
	// integ
	Element* integ = Neutral::create( "HSolve", "integ",
		solve, Id::scratchId() );
	assert( integ != 0 );
	bool ret = cell()->findFinfo( "cell-solve" )->
		add( cell(), integ, integ->findFinfo( "cell-solve" ) );
	assert( ret );
	
	// scan
	ret = set( integ, "scanCreate" );
	assert( ret );
	
	// hub
	Id scan( solve->id().path() + "/scan" );
	assert( scan.good() );
	ret = set( scan(), "hubCreate" );
	assert( ret );
	
	// Solver initialization.
	// Scheduling is currently simple: all solvers attach to t0.
	Id cj( "/sched/cj" );
	Id t0( "/sched/cj/t0" );
	assert( cj.good() );
	assert( t0.good() );
	
	double dt;
	get< double >( t0(), "dt", dt );
	send2< const Element*, double >( 
		cell(), solveInitSlot,
		seed(), dt );
	
	set( cj(), "resched" );
}

void Cell::checkTree( ) const
{
	;
}
