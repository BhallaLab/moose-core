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

const Cinfo* initHHGateCinfo()
{
	/**
	 * This is a shared message to communicate with the channel.
	 * Receives Vm and gate state.
	 * Sends updated gate state and conductance scale term for gate.
	 */
	static TypeFuncPair gateTypes[] =
	{
		TypeFuncPair( Ftype2< double, double >::global(),
						RFCAST( &HHGate::gateFunc ) ),
		TypeFuncPair( Ftype2< double, double >::global(), 0),
	};
	
	static Finfo* HHGateFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "power", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &HHGate::getPower ),
			RFCAST( &HHGate::setPower )
		), 
		new ValueFinfo( "state", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &HHGate::getState ),
			RFCAST( &HHGate::setState )
		), 
		new ValueFinfo( "instant", ValueFtype1< int >::global(),
			reinterpret_cast< GetFunc >( &HHGate::getInstant ),
			RFCAST( &HHGate::setInstant )
		), 
		//// Stuff needed here.
		/*
		new ObjFinfo< Interpol >(
			"A", &HHGateWrapper::getA,
			&HHGateWrapper::setA, &HHGateWrapper::lookupA, "Interpol"),
		new ObjFinfo< Interpol >(
			"B", &HHGateWrapper::getB,
			&HHGateWrapper::setB, &HHGateWrapper::lookupB, "Interpol"),
			*/
		
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "gate", gateTypes, 2 ),

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "postCreate", Ftype0::global(),
			&HHGate::postCreate ),
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

///\todo: Need to use a functor to handle arbitrary powers
void HHGate::innerSetPower( double power )
{
	power_ = power;
	if ( power_ == 0.0 )
		takePower_ = power0;
	else if ( power_ == 1.0 )
		takePower_ = power1;
	else if ( power_ == 2.0 )
		takePower_ = power2;
	else if ( power_ == 3.0 )
		takePower_ = power3;
	else if ( power_ == 4.0 )
		takePower_ = power4;
	else
		takePower_ = powerN;
}

void HHGate::setPower( const Conn& c, double power )
{
	static_cast< HHGate* >( c.data() )->innerSetPower( power );
}
double HHGate::getPower( const Element* e )
{
	return static_cast< HHGate* >( e->data() )->power_;
}

void HHGate::setState( const Conn& c, double state )
{
	static_cast< HHGate* >( c.data() )->state_ = state;
}
double HHGate::getState( const Element* e )
{
	return static_cast< HHGate* >( e->data() )->state_;
}

void HHGate::setInstant( const Conn& c, int instant )
{
	static_cast< HHGate* >( c.data() )->instant_ = instant;
}
int HHGate::getInstant( const Element* e )
{
	return static_cast< HHGate* >( e->data() )->instant_;
}


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void HHGate::gateFunc(
				const Conn& c, double v, double state, double dt )
{
	static_cast< HHGate *>( c.data() )->innerGateFunc( c, v, state, dt);
}

void HHGate::innerGateFunc(
				const Conn& c, double v, double state, double dt )
{
	if ( instant_ ) {
		state = A_.innerLookup( v ) / B_.innerLookup( v );
	} else {
		double y = B_.innerLookup( v );
		double x = exp( -y * dt );
		state = state * x + ( A_.innerLookup( v ) / y ) * ( 1 - x );
	}

	// This ugly construction returns the info back to sender.
	sendTo2< double, double >( c.targetElement(), gateSlot,
		c.targetIndex(), state, takePower_( state ) );
}

/*
void HHGate::reinitFunc( const Conn& c, double power, double dt, int instant)
{
	static_cast< HHGate *>( c.data() )->
			innerReinitFunc( power, dt, instant );
}
*/

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

	cout << "HHGate::postCreate called\n";
	const Cinfo* ic = initInterpolCinfo();
	Element* A = ic->create( "A", static_cast< void* >( &h->A_ ) );
	e->findFinfo( "childSrc" )->add( e, A, A->findFinfo( "child" ) );

	Element* B = ic->create( "B", static_cast< void* >( &h->B_) );
	e->findFinfo( "childSrc" )->add( e, B, B->findFinfo( "child" ) );
}
