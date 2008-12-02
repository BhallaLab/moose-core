/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <map>
#include <algorithm>

#include "moose.h"
#include "InterSolverFlux.h"

const Cinfo* initInterSolverFluxCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process",
			Ftype1< ProcInfo >::global(),
			RFCAST( &InterSolverFlux::processFunc )),
		new DestFinfo( "reinit",
			Ftype1< ProcInfo >::global(),
			RFCAST( &InterSolverFlux::reinitFunc )),
	};

	static Finfo* transferShared[] =
	{
		new DestFinfo( "transfer",
			Ftype1< ProcInfo >::global(),
			RFCAST( &InterSolverFlux::transferFunc )),
		new DestFinfo( "reinit",
			Ftype1< ProcInfo >::global(),
			RFCAST( &InterSolverFlux::reinitFunc )),
	};

	static Finfo* fluxShared[] =
	{
		new SrcFinfo( "fluxSrc",
			Ftype1< vector< double > >::global(),
			"Passes efflux vector to other solvers."
		),
		new DestFinfo( "flux",
			Ftype1< vector< double > >::global(),
			RFCAST( &InterSolverFlux::sumFlux ),
			"Handles incoming flux from other solvers"
		),
	};

	static Finfo* interSolverFluxFinfos[] =
	{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		new ValueFinfo( "method", 
			ValueFtype1< string >::global(),
			GFCAST( &InterSolverFlux::getMethod ), 
			RFCAST( &InterSolverFlux::setMethod ),
			"Specifies the numerical method to use to calculate flux."
			"Currently there is only one option, Euler"
		),
		/*
		///////////////////////////////////////////////////////
		// MsgSrc definitions
		///////////////////////////////////////////////////////
		new SrcFinfo( "fluxSrc",
			Ftype1< vector< double > >::global(),
			"Passes efflux vector to other solvers."
		),
		///////////////////////////////////////////////////////
		// MsgDest definitions
		///////////////////////////////////////////////////////
		new DestFinfo( "flux",
			Ftype1< vector< double > >::global(),
			RFCAST( &InterSolveFlux::sumFlux ),
			"Handles incoming flux from other solvers"
		),
		*/
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		new SharedFinfo( "process", processShared, 
				sizeof( processShared )/ sizeof( Finfo* ) ),
		new SharedFinfo( "transfer", transferShared, 
				sizeof( transferShared )/ sizeof( Finfo* ),
				"Initiates call to pass out the efflux vector."
				"Has to be out of phase with the 'process' call"
				"since we need all data transferred before process."
				),
		new SharedFinfo( "flux", fluxShared, 
				sizeof( fluxShared )/ sizeof( Finfo* ),
				"Handles reciprocal flux exchange between two"
				"solvers."
				),
	};

	static string doc[] =
	{
		"Name", "InterSolverFlux",
		"Author", "Upinder S. Bhalla, Dec 2008, NCBS",
		"Description","InterSolverFlux: Handles flux of molecules"
		"between kinetic solvers. Currently just does simple "
		"forward Euler summation."
		"Designed to act as a stub class for stoich. A given stoich"
		"object may have multiple InterSolveFlux handles. Each one"
		"talks to a single other InterSolveFlux object."
	};	
	static  Cinfo interSolverFluxCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		interSolverFluxFinfos,
		sizeof(interSolverFluxFinfos)/sizeof(Finfo *),
		ValueFtype1< InterSolverFlux >::global()
	);

	return &interSolverFluxCinfo;
}

static const Cinfo* interSolverFluxCinfo = initInterSolverFluxCinfo();

static const Slot fluxSlot =
	initInterSolverFluxCinfo()->getSlot( "fluxSrc" );


///////////////////////////////////////////////////
// Basic class function definitions
///////////////////////////////////////////////////
InterSolverFlux::InterSolverFlux()
	: method_( "Euler" )
{;}

InterSolverFlux::InterSolverFlux( vector< double* > localPtrs,
			vector< double > fluxRate )
	: method_( "Euler" ), localPtrs_( localPtrs ), fluxRate_( fluxRate )
{;}
///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

/*
bool InterSolverFlux::getIsInitialized( Eref e )
{
	return static_cast< const InterSolverFlux* >( e.data() )->isInitialized_;
}
*/

string InterSolverFlux::getMethod( Eref e )
{
	return static_cast< const InterSolverFlux* >( e.data() )->method_;
}
void InterSolverFlux::setMethod( const Conn* c, string method )
{
	static_cast< InterSolverFlux* >( c->data() )->innerSetMethod( method );
}

void InterSolverFlux::innerSetMethod( const string& method )
{
	method_ = method;
	cout << "in void InterSolverFlux::innerSetMethod( string method ) \n";
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void InterSolverFlux::processFunc( const Conn* c, ProcInfo info )
{
	static_cast< InterSolverFlux* >( c->data() )->innerProcessFunc( 
		c->target(), info );
}

void InterSolverFlux::innerProcessFunc( Eref e, ProcInfo info )
{
	vector< double >::iterator f = flux_.begin();
	for (vector< double* >::iterator i = 
		localPtrs_.begin(); i != localPtrs_.end(); ++i )
		**i -= *f++;

	// No need to zero it out, as we have only one incoming msg.
	// flux_.assign( sizeof( flux_ ), 0.0 );
}

void InterSolverFlux::transferFunc( const Conn* c, ProcInfo info )
{
	static_cast< InterSolverFlux* >( c->data() )->innerTransferFunc( 
		c->target(), info );
}
/**
 * This function happens in a different stage from the main tick.
 * All the fluxes must be transferred before the 'process' calculations
 */
void InterSolverFlux::innerTransferFunc( Eref e, ProcInfo info )
{
	vector< double >::iterator k = fluxRate_.begin();
	vector< double >::iterator j = flux_.begin();
	// Set up the outgoing flux in the flux vector. 
	// It is used to send out to target as well as for local calculation
	for (vector< double* >::iterator i = 
		localPtrs_.begin(); i != localPtrs_.end(); ++i )
		*j++ = *k * **i;
	
	send1< vector< double > >( e, fluxSlot, flux_ );
}

// Separate flux transfer phase and process phase. The flux transfer
// happens first, so that we can do a trapezoidal calculation... later.
void InterSolverFlux::sumFlux( const Conn* c, vector< double > n )
{
	static_cast< InterSolverFlux* >( c->data() )->innerSumFlux( n );
}

/**
 * Flux contains the efflux. So n is subtracted from it.
 */
void InterSolverFlux::innerSumFlux( vector< double >& n )
{
	vector< double >::iterator f = flux_.begin();
	for( vector< double >::iterator i = n.begin(); i != n.end(); ++i ) {
		*f++ -= *i;
	}
}


void InterSolverFlux::reinitFunc( const Conn* c, ProcInfo info )
{
	// send0( c->target(), reinitSlot );
}
