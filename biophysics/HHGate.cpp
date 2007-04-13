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

	A_.table_.resize( xdivs );
	B_.table_.resize( xdivs );

	double x = parms[XMIN];
	double dx = ( parms[XMAX] - x ) / xdivs;
	double prevAentry = 0.0;
	double prevBentry = 0.0;
	double temp, temp2;
	unsigned int i;

	for( i = 0; i <= xdivs; i++ ) {
		if ( fabs( parms[4] < SINGULARITY ) ) {
			temp = A_.table_[i] = 0.0;
		} else {
			temp2 = parms[2] + exp( ( x + parms[3] ) / parms[4] );
			if ( fabs( temp2 ) < SINGULARITY )
				temp = A_.table_[i] = prevAentry;
			else
				temp = A_.table_[i] = ( parms[0] + parms[1] * x) / temp2;
		}
		if ( fabs( parms[9] ) < SINGULARITY ) {
			B_.table_[i] = 0.0;
		} else {
			temp2 = parms[7] + exp( ( x + parms[8] ) / parms[9] );
			if ( fabs( temp2 ) < SINGULARITY )
				B_.table_[i] = prevBentry;
			else
				B_.table_[i] = ( parms[5] + parms[6] * x ) / temp2;
		}
		if ( doTau == 0 )
			B_.table_[i] += temp;

		prevAentry = A_.table_[i];
		prevBentry = B_.table_[i];
		x += dx;
	}

	if ( doTau ) {
		for( i = 0; i <= xdivs; i++ ) {
			temp = A_.table_[i];
			temp2 = B_.table_[i];
			if ( fabs( temp ) < SINGULARITY ) {
				A_.table_[i] = 0.0;
				B_.table_[i] = 0.0;
			} else {
				A_.table_[i] = temp2 / temp;
				B_.table_[i] = temp - temp2;
			}
		}
	}
}
