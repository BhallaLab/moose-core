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
	/**
	 * This is a shared message to communicate with the channel.
	 * Receives Vm
	 * Sends A and B from the respective table lookups based on Vm.
	 */
	static TypeFuncPair gateTypes[] =
	{
		TypeFuncPair( Ftype1< double >::global(),
						RFCAST( &HHGate::gateFunc ) ),
		TypeFuncPair( Ftype2< double, double >::global(), 0),
	};
	
	static Finfo* HHGateFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions.  Currently empty
	///////////////////////////////////////////////////////
		
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "gate", gateTypes, 2 ),

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "postCreate", Ftype0::global(),
			&HHGate::postCreate ),
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
	};
	
	static Cinfo HHGateCinfo(
		"HHGate",
		"Upinder S. Bhalla, 2005, NCBS",
		"HHGate: Gate for Hodkgin-Huxley type channels, equivalent to the\nm and h terms on the Na squid channel and the n term on K.\nThis takes the voltage and state variable from the channel,\ncomputes the new value of the state variable and a scaling,\ndepending on gate power, for the conductance. These two\nterms are sent right back in a message to the channel.",
		initNeutralCinfo(),
		HHGateFinfos,
		sizeof(HHGateFinfos)/sizeof(Finfo *),
		ValueFtype1< HHGate >::global()
	);

	return &HHGateCinfo;
}

static const Cinfo* hhGateCinfo = initHHGateCinfo();
static const unsigned int gateSlot =
	initHHGateCinfo()->getSlotIndex( "gate" );


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void HHGate::gateFunc(
				const Conn& c, double v )
{
	// static_cast< HHGate *>( c.data() )->innerGateFunc( c, v );
	HHGate *h = static_cast< HHGate *>( c.data() );

	sendTo2< double, double >( c.targetElement(), gateSlot,
		c.targetIndex(),
		h->A_.innerLookup( v ) , h->B_.innerLookup( v ) );
}

/**
 * This creates two nested child objects on the HHGate, to hold the
 * Interpols.
 * \todo: We need to figure out how to handle the deletion correctly,
 * without zapping the data fields of these child objects.
 */

void HHGate::postCreate( const Conn& c )
{
	HHGate* h = static_cast< HHGate *>( c.data() );
	Element* e = c.targetElement();

	// cout << "HHGate::postCreate called\n";
	const Cinfo* ic = initInterpolCinfo();
	// Here we must set the noDelFlag to 1 because these data
	// parts belong to the parent HHGate structure.
	Element* A = ic->create( "A", static_cast< void* >( &h->A_ ), 1 );
	e->findFinfo( "childSrc" )->add( e, A, A->findFinfo( "child" ) );

	Element* B = ic->create( "B", static_cast< void* >( &h->B_), 1 );
	e->findFinfo( "childSrc" )->add( e, B, B->findFinfo( "child" ) );
}

// static func
void HHGate::setupAlpha( const Conn& c, vector< double > parms )
{
	if ( parms.size() != 13 ) {
			cout << "HHGate::setupAlpha: Error: parms.size() != 13\n";
			return;
	}
	// static_cast< HHGate *>( c.data() )->innerGateFunc( c, v );
	static_cast< HHGate *>( c.data() )->setupTables( parms, 0 );
}

// static func
void HHGate::setupTau( const Conn& c, vector< double > parms )
{
	if ( parms.size() != 13 ) {
			cout << "HHGate::setupTau: Error: parms.size() != 13\n";
			return;
	}
	// static_cast< HHGate *>( c.data() )->innerGateFunc( c, v );
	static_cast< HHGate *>( c.data() )->setupTables( parms, 1 );
}

// static func
void HHGate::tweakAlpha( const Conn& c )
{
	static_cast< HHGate *>( c.data() )->tweakTables( 0 );
}

// static func
void HHGate::tweakTau( const Conn& c )
{
	static_cast< HHGate *>( c.data() )->tweakTables( 1 );
}

/**
 * Sets up the tables. See code in GENESIS/src/olf/new_interp.c,
 * function setup_tab_values,
 * fine tuned by Erik De Schutter.
 */
void HHGate::setupTables( const vector< double >& parms, bool doTau )
{
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
	double temp, temp2;
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
