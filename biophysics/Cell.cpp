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

map< string, MethodInfo >& Cell::methodMap()
{
    static map<string, MethodInfo> * methodMap_  = new map<string, MethodInfo>();
    return *methodMap_;
}

const Cinfo* initCellCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			&dummyFunc,
			"Cell does not process at simulation time--it only sets up the solver at reset "),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &Cell::reinitFunc ) ),
	};
	
	static Finfo* integShared[] =
	{
		new SrcFinfo( "integSetup",
			Ftype2< Id, double >::global() ),
		new DestFinfo( "comptList",
			Ftype1< const vector< Id >* >::global(),
			&dummyFunc,
			"Placeholder for receiving list of compartments from solver." ),
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
			&dummyFunc
		),
		new ValueFinfo( "implicit", 
			ValueFtype1< bool >::global(),
			GFCAST( &Cell::getImplicit ), 
			&dummyFunc
		),
		new ValueFinfo( "description", 
			ValueFtype1< string >::global(),
			GFCAST( &Cell::getDescription ), 
			&dummyFunc
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
		new SharedFinfo( "cell-integ", integShared,
			sizeof( integShared ) / sizeof( Finfo* ) ),
		process,
	};
	
	// Clock 0 will do for reset.
	static SchedInfo schedInfo[] = { { process, 0, 0 } };
	
	static string doc[] =
	{
		"Name", "Cell",
		"Author", "Subhasis Ray, Niraj Dudani, 2007, NCBS",
		"Description", "Cell: Container for a neuron's components. Also manages automatic solver setup.",
	};	
	static Cinfo cellCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
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

static const Slot integSetupSlot =
	initCellCinfo()->getSlot( "cell-integ.integSetup" );

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
	methodMap()[name] = mi;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void Cell::reinitFunc( const Conn* c, ProcInfo p )
{
	static_cast< Cell* >( c->data() )->
		innerReinitFunc( c->target()->id(), p );
}

void Cell::innerReinitFunc( Id cell, ProcInfo p )
{
	double dt;
	
	// Delete existing solver
	Id oldSolve = Id::localId( cell.path() + "/solve" );
	if ( oldSolve.good() )
		set( oldSolve(), "destroy" );
	
	if ( method_ == "ee" )
		return;
	
	// Find any compartment that is a (grand)child of this cell
	Id seed = findCompt( cell );
	if ( seed.bad() ) // No compartment found.
		return;
	
	// t0's dt is used to set the solver's dt
	Id t0( "/sched/cj/t0" );
	assert( t0.good() );
	get< double >( t0(), "dt", dt );
	
	// The solver could be set up to send back a list of solved compartments.
	setupSolver( cell, seed, dt );
	
	// The compartment list could then be used to see if the tree below the
	// cell is built correctly. For instance, one could check if there are
	// compartments dangling outside the cell.
	checkTree( );
}

/**
 * This function performs a depth-first search of the tree under the current
 * cell. First compartment found is returned as the seed.
 */ 
Id Cell::findCompt( Id cell )
{
	/* 'curr' is the current element under consideration. 'cstack' is a list
	 * of all elements (and their immediate siblings) found on the path from
	 * the root element (the Cell) to the current element.
	 */
	vector< vector< Id > > cstack;
	Id seed = Id::badId();
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

void Cell::setupSolver( Id cell, Id seed, double dt ) const
{
	// Create solve, and its children: integ and hub.
	Element* solve = Neutral::create( "Neutral", "solve",
		cell, Id::scratchId() );
	
	// integ
	Element* integ = Neutral::create( "HSolve", "integ",
		solve->id(), Id::scratchId() );
	assert( integ != 0 );
	Eref( cell() ).add( "cell-integ", integ, "cell-integ" );
	
	// With this request, the HSolve integrator sets itself up (reads in the
	// cell model). Then it creates the hub as its sibling (i.e., child of
	// 'solve'). The solver gets autoscheduled on t0 during its creation.
	send2< Id, double >( cell(), integSetupSlot, seed, dt );
}

void Cell::checkTree( ) const
{
	;
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
	map< string, MethodInfo >::iterator i = methodMap().find( value );
	if ( i != methodMap().end() ) {
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

string Cell::getMethod( Eref e )
{
	return static_cast< Cell* >( e.data() )->method_;
}

bool Cell::getVariableDt( Eref e )
{
	return static_cast< Cell* >( e.data() )->variableDt_;
}

bool Cell::getImplicit( Eref e )
{
	return static_cast< Cell* >( e.data() )->implicit_;
}

string Cell::getDescription( Eref e )
{
	return static_cast< Cell* >( e.data() )->description_;
}

