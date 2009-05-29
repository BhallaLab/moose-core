/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <math.h>
#include "moose.h"
#include "../builtins/Interpol.h"
#include "HHGate.h"

static const double SINGULARITY = 1.0e-6;

const Cinfo* initHHGateCinfo()
{
	static Finfo* gateShared[] =
	{
		new DestFinfo( "Vm", Ftype1< double >::global(),
						RFCAST( &HHGate::gateFunc ) ),
		new SrcFinfo( "gate", Ftype2< double, double >::global() ),
	};
	
	static Finfo* HHGateFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions.
	///////////////////////////////////////////////////////
		new LookupFinfo( "A",
			LookupFtype< double, double >::global(),
			GFCAST( &HHGate::getAValue ),
			&dummyFunc ),
		new LookupFinfo( "B",
			LookupFtype< double, double >::global(),
			GFCAST( &HHGate::getBValue ),
			&dummyFunc ),
		
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "gate", gateShared, 
			sizeof( gateShared ) / sizeof( Finfo* ),
			"This is a shared message to communicate with the channel.\n"
			"Receives Vm \n"
			"Sends A and B from the respective table lookups based on Vm." ),

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "createInterpols",
			Ftype1< IdGenerator >::global(),
			RFCAST( &HHGate::createInterpols ),
			"Request the gate explicitly to create Interpols, with the given "
			"ids. This is used when the gate is a global object, and so the "
			"interpols need to be globals too. Comes in use in TABCREATE in the "
			"parallel context." ),
		new DestFinfo( "setupAlpha",
			Ftype1< vector< double > >::global(),
			RFCAST( &HHGate::setupAlpha ) ),
		new DestFinfo( "setupTau",
			Ftype1< vector< double > >::global(),
			RFCAST( &HHGate::setupTau ) ),
		new DestFinfo( "tweakAlpha", Ftype0::global(),
			&HHGate::tweakAlpha ),
		new DestFinfo( "tweakTau", Ftype0::global(),
			&HHGate::tweakTau ),
		new DestFinfo( "setupGate",
			Ftype1< vector< double > >::global(),
			RFCAST( &HHGate::setupGate ) ),
	};

	static string doc[] =
	{
		"Name", "HHGate",
		"Author", "Upinder S. Bhalla, 2005, NCBS",
		"Description", "HHGate: Gate for Hodkgin-Huxley type channels, equivalent to the "
				"m and h terms on the Na squid channel and the n term on K. "
				"This takes the voltage and state variable from the channel, "
				"computes the new value of the state variable and a scaling, "
				"depending on gate power, for the conductance. These two "
				"terms are sent right back in a message to the channel.",
	};	
	static Cinfo HHGateCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		HHGateFinfos,
		sizeof(HHGateFinfos)/sizeof(Finfo *),
		ValueFtype1< HHGate >::global()
	);

	return &HHGateCinfo;
}

static const Cinfo* hhGateCinfo = initHHGateCinfo();
static const Slot gateSlot =
	initHHGateCinfo()->getSlot( "gate" );


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

double HHGate::getAValue( Eref e, const double& v )
{
	HHGate* h = static_cast< HHGate* >( e.data() );
	return h->A_.innerLookup( v );
}

double HHGate::getBValue( Eref e, const double& v )
{
	HHGate* h = static_cast< HHGate* >( e.data() );
	return h->B_.innerLookup( v );
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void HHGate::gateFunc( const Conn* c, double v )
{
	// static_cast< HHGate *>( c.data() )->innerGateFunc( c, v );
	// cout << "HHGate func: " << c->data() << " with v= " << v << " on " << c->target().name() << " from " << c->source().name() << endl << flush;
	HHGate *h = static_cast< HHGate *>( c->data() );
	sendBack2< double, double >( c, gateSlot,
		h->A_.innerLookup( v ) , h->B_.innerLookup( v ) );
}

/**
 * Request the gate explicitly to create Interpols, with the given ids. This is
 * used when the gate is a global object, and so the interpols need to be
 * globals too. Comes in use in TABCREATE in the parallel context.
 */
void HHGate::createInterpols( const Conn* c, IdGenerator idGen )
{
	HHGate* h = static_cast< HHGate* >( c->data() );
	Eref e = c->target();
	
	const Cinfo* ic = initInterpolCinfo();
	// Here we must set the noDelFlag to 1 because these data
	// parts belong to the parent HHGate structure.
	Element* A = ic->create( 
		idGen.next(), "A", static_cast< void* >( &h->A_ ), 1 );
	e.add( "childSrc", A, "child" );
	
	Element* B = ic->create( 
		idGen.next(), "B", static_cast< void* >( &h->B_), 1 );
	e.add( "childSrc", B, "child" );
}

// static func
void HHGate::setupAlpha( const Conn* c, vector< double > parms )
{
	if ( parms.size() != 13 ) {
			cout << "HHGate::setupAlpha: Error: parms.size() != 13\n";
			return;
	}
	// static_cast< HHGate *>( c.data() )->innerGateFunc( c, v );
	static_cast< HHGate *>( c->data() )->setupTables( parms, 0 );
}

// static func
void HHGate::setupTau( const Conn* c, vector< double > parms )
{
	if ( parms.size() != 13 ) {
			cout << "HHGate::setupTau: Error: parms.size() != 13\n";
			return;
	}
	// static_cast< HHGate *>( c.data() )->innerGateFunc( c, v );
	static_cast< HHGate *>( c->data() )->setupTables( parms, 1 );
}

// static func
void HHGate::tweakAlpha( const Conn* c )
{
	static_cast< HHGate *>( c->data() )->tweakTables( 0 );
}

// static func
void HHGate::tweakTau( const Conn* c )
{
	static_cast< HHGate *>( c->data() )->tweakTables( 1 );
}

// static func
void HHGate::setupGate( const Conn* c, vector< double > parms )
{
	if ( parms.size() != 9 ) {
			cout << "HHGate::setupGate: Error: parms.size() != 9\n";
			return;
	}
	// static_cast< HHGate *>( c.data() )->innerGateFunc( c, v );
	static_cast< HHGate *>( c->data() )->innerSetupGate( parms );
}

/**
 * Sets up the tables. See code in GENESIS/src/olf/new_interp.c,
 * function setup_tab_values,
 * fine tuned by Erik De Schutter.
 */
void HHGate::setupTables( const vector< double >& parms, bool doTau )
{
	assert( parms.size() == 13 );
	static const int XDIVS = 10;
	static const int XMIN = 11;
	static const int XMAX = 12;
	if ( parms[XDIVS] < 1 ) return;
	unsigned int xdivs = static_cast< unsigned int >( parms[XDIVS] );

	A_.resize( xdivs + 1 );
	B_.resize( xdivs + 1 );
	double xmin = parms[XMIN];
	double xmax = parms[XMAX];
	A_.localSetXmin( xmin );
	B_.localSetXmin( xmin );
	A_.localSetXmax( xmax );
	B_.localSetXmax( xmax );

	double x = xmin;
	double dx = ( xmax - xmin ) / xdivs;
	double prevAentry = 0.0;
	double prevBentry = 0.0;
	double temp; 
	double temp2 = 0.0;
	unsigned int i;

	for( i = 0; i <= xdivs; i++ ) {
		if ( fabs( parms[4] ) < SINGULARITY ) {
			temp = 0.0;
			A_.setTableValue( temp, i );
		} else {
			temp2 = parms[2] + exp( ( x + parms[3] ) / parms[4] );
			if ( fabs( temp2 ) < SINGULARITY ) {
				temp = prevAentry;
				A_.setTableValue( temp, i );
			} else {
				temp = ( parms[0] + parms[1] * x) / temp2;
				A_.setTableValue( temp, i );
			}
		}
		if ( fabs( parms[9] ) < SINGULARITY ) {
			B_.setTableValue( 0.0, i );
		} else {
			temp2 = parms[7] + exp( ( x + parms[8] ) / parms[9] );
			if ( fabs( temp2 ) < SINGULARITY )
				B_.setTableValue( prevBentry, i );
			else
				B_.setTableValue( 
						(parms[5] + parms[6] * x ) / temp2, i );
				// B_.table_[i] = ( parms[5] + parms[6] * x ) / temp2;
		}
		// There are cleaner ways to do this, but this keeps
		// the relation to the GENESIS version clearer.
		// Note the additional SINGULARITY check, to fix a bug
		// in the earlier code.
		if ( doTau == 0 && fabs( temp2 ) > SINGULARITY )
			B_.setTableValue( B_.getTableValue( i ) + temp, i );

		prevAentry = A_.getTableValue( i );
		prevBentry = B_.getTableValue( i );
		x += dx;
	}

	prevAentry = 0.0;
	prevBentry = 0.0;
	if ( doTau ) {
		for( i = 0; i <= xdivs; i++ ) {
			temp = A_.getTableValue( i );
			temp2 = B_.getTableValue( i );
			if ( fabs( temp ) < SINGULARITY ) {
				A_.setTableValue( prevAentry, i );
				B_.setTableValue( prevBentry, i );
			} else {
				A_.setTableValue( temp2 / temp, i );
				B_.setTableValue( 1.0 / temp, i );
				// B_.setTableValue( temp - temp2, i );
			}
			prevAentry = A_.getTableValue( i );
			prevBentry = B_.getTableValue( i );
		}
	}
}

/**
 * Tweaks the A and B entries in the tables from the original
 * alpha/beta or minf/tau values. See code in 
 * GENESIS/src/olf/new_interp.c, function tweak_tab_values
 */
void HHGate::tweakTables( bool doTau )
{
	unsigned int i;
	unsigned int size = A_.size();
	assert( size == B_.size() );
	if ( doTau ) {
		for ( i = 0; i < size; i++ ) {
			double temp = A_.getTableValue( i );
			double temp2 = B_.getTableValue( i );
			if ( fabs( temp ) < SINGULARITY )  {
				if ( temp < 0.0 )
					temp = -SINGULARITY;
				else
					temp = SINGULARITY;
			}
			A_.setTableValue( temp2 / temp, i );
			B_.setTableValue( 1.0 / temp, i );
		}
	} else {
		for ( i = 0; i < size; i++ )
			B_.setTableValue( 
				A_.getTableValue( i ) + B_.getTableValue( i ), i );
	}
}

void HHGate::innerSetupGate( const vector< double >& parms )
{
	// The nine arguments are :
	// A B C D F size min max isbeta
	// If size == 0 then we check that the gate has already been allocated.
	// If isbeta is true then we also have to do the conversion to 
	// HHGate form of alpha, alpha+beta, assuming that the alpha gate 
	// has already been setup. This uses tweakTables.
	// We may need to resize the tables if they don't match here.

	assert( parms.size() == 9 );
	const double& A = parms[0];
	const double& B = parms[1];
	const double& C = parms[2];
	const double& D = parms[3];
	const double& F = parms[4];
	int size = static_cast< int > (parms[5] );
	const double& min = parms[6];
	const double& max = parms[7];
	bool isBeta = static_cast< bool >( parms[8] );

	Interpol& ip = ( isBeta ) ? B_ : A_;
	if ( size <= 0 ) { // Look up size, min, max from the interpol
		size = ip.size() - 1;
		if ( size <= 0 ) {
			cout << "Error: setupGate has zero size\n";
			return;
		}
	} else {
		ip.resize( size + 1 );
	}

	double dx = ( max - min ) / static_cast< double >( size );
	double x = min + dx / 2.0;
	for ( int i = 0; i <= size; i++ ) {
		if ( fabs ( F ) < SINGULARITY ) {
			ip.setTableValue( 0.0, i );
		} else {
			double temp2 = C + exp( ( x + D ) / F );
			if ( fabs( temp2 ) < SINGULARITY )
				ip.setTableValue( ip.getTableValue( i - 1 ), i );
			else
				ip.setTableValue( ( A + B * x ) / temp2 , i );
		}
	}

	if ( isBeta ) {
		assert( A_.size() > 0 );
		// Here we ensure that the tables are the same size
		if ( A_.size() != B_.size() ) {
			if ( A_.size() > B_.size() ) {
				int mode = B_.mode();
				// Note that the innerTabFill expects to allocate the
				// terminating entry, so we put in size - 1.
				B_.innerTabFill( A_.size() - 1, mode );
			} else {
				int mode = A_.mode();
				A_.innerTabFill( B_.size() - 1, mode );
			}
		}
		// Then we do the tweaking to convert to HHChannel form.
		tweakTables( 0 );
	}
}
