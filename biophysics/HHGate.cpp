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
	// Field definitions. Non needed
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
	Element* A = ic->create( "A", static_cast< void* >( &h->A_ ) );
	e->findFinfo( "childSrc" )->add( e, A, A->findFinfo( "child" ) );

	Element* B = ic->create( "B", static_cast< void* >( &h->B_) );
	e->findFinfo( "childSrc" )->add( e, B, B->findFinfo( "child" ) );
}
