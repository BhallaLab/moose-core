/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include <math.h>

#include "Compartment.h"


const double Compartment::EPSILON = 1.0e-15;

/**
 * The initCompartmentCinfo() function sets up the Compartment class.
 * This function uses the common trick of having an internal
 * static value which is created the first time the function is called.
 * There are several static arrays set up here. The ones which
 * use TypeFuncPairs are for shared messages where multiple kinds
 * of information go along the same connection.
 */
const Cinfo* initCompartmentCinfo()
{
	/**
	 * This is a shared message to receive Process messages from
	 * the scheduler objects.
	 * The first entry is a MsgDest for the Process operation. It
	 * has a single argument, ProcInfo, which holds
	 * lots of information about current time, thread, dt and so on.
	 * The second entry is a MsgDest for the Reinit operation. It
	 * also uses ProcInfo.
	 */
	static TypeFuncPair processTypes[] =
	{
		TypeFuncPair( Ftype1< ProcInfo >::global(),
				RFCAST( &Compartment::processFunc ) ),
		TypeFuncPair( Ftype1< ProcInfo >::global(),
				RFCAST( &Compartment::reinitFunc ) ),
	};
	
	/**
	 * This is a shared message to receive Init messages from
	 * the scheduler objects.
	 * Its job is to separate the compartmental calculations from
	 * the message passing.
	 * It doesn't really need to be shared, as it does not use
	 * the reinit part, but the scheduler objects
	 * expect this form of message for all scheduled output.
	 *
	 * The first entry is a MsgDest for the Process operation. It
	 * has a single argument, ProcInfo, which holds
	 * lots of information about current time, thread, dt and so on.
	 * The second entry is a dummy MsgDest for the Reinit operation. It
	 * also uses ProcInfo.
	 */
	static TypeFuncPair initTypes[] =
	{
		TypeFuncPair( Ftype1< ProcInfo >::global(),
				RFCAST( &Compartment::initFunc ) ),
		TypeFuncPair( Ftype1< ProcInfo >::global(),
				RFCAST( &dummyFunc ) ),
	};

	/**
	 * This is a shared message from a compartment to channels.
	 * The first entry is a MsgDest for the info coming from the channel
	 * It expects Gk and Ek from the channel as args.
	 * The second entry is a MsgSrc sending Vm
	 */
	static TypeFuncPair channelTypes[] =
	{
		TypeFuncPair( Ftype2< double, double >::global(),
				RFCAST( &Compartment::channelFunc ) ),
		TypeFuncPair( Ftype1< double >::global(), 0),
	};

	/**
	 * This is a shared message between asymmetric compartments.
	 * axial messages (this kind) connect up to raxial messages
	 * (defined below).
	 *
	 * The soma should use raxial messages to connect to the axial
	 * message of all the immediately adjacent dendritic compartments.
	 * This puts the (low) somatic resistance in series with these
	 * dendrites.
	 * Dendrites should then use raxial messages to connect on to
	 * more distal dendrites.
	 *
	 * In other words, raxial messages should face outward from the
	 * soma.
	 *
	 * The first entry is a MsgDest for the info coming from the other
	 * compt. It expects Ra and Vm from the other compt as args.
	 * The second is a MsgSrc sending Vm to the axialFunc
	 * of the target compartment.
	 *
	 * Note that the message is named after the source type.
	 */
	static TypeFuncPair axialTypes[] =
	{
		TypeFuncPair( Ftype2< double, double >::global(),
				RFCAST( &Compartment::raxialFunc ) ),
		TypeFuncPair( Ftype1< double >::global(), 0)
	};

	/**
	 * This is a raxial shared message between asymmetric compartments.
	 *
	 * The first entry is a MsgDest for the info coming from the other
	 * compt. It expects Vm from the other compt as an arg.
	 * The second is a MsgSrc sending Ra and Vm to the raxialFunc
	 * of the target compartment.
	 */
	static TypeFuncPair raxialTypes[] =
	{
		TypeFuncPair( Ftype1< double >::global(),
				RFCAST( &Compartment::axialFunc ) ),
		TypeFuncPair( Ftype2< double, double >::global(), 0)
	};
	
	static Finfo* compartmentFinfos[] = 
	{
		new ValueFinfo( "Vm", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getVm ),
			reinterpret_cast< RecvFunc >( &Compartment::setVm )
		),
		new ValueFinfo( "Cm", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getCm ),
			reinterpret_cast< RecvFunc >( &Compartment::setCm )
		),
		new ValueFinfo( "Em", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getEm ),
			reinterpret_cast< RecvFunc >( &Compartment::setEm )
		),
		new ValueFinfo( "Im", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getIm ),
			&dummyFunc
		),
		new ValueFinfo( "initVm", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getInitVm ),
			reinterpret_cast< RecvFunc >( &Compartment::setInitVm )
		),
		new ValueFinfo( "Rm", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getRm ),
			reinterpret_cast< RecvFunc >( &Compartment::setRm )
		),
		new ValueFinfo( "Ra", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getRa ),
			reinterpret_cast< RecvFunc >( &Compartment::setRa )
		),
		new ValueFinfo( "diameter", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getDiameter ),
			reinterpret_cast< RecvFunc >( &Compartment::setDiameter )
		),
		new ValueFinfo( "length", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getLength ),
			reinterpret_cast< RecvFunc >( &Compartment::setLength )
		),

		new SharedFinfo( "process", processTypes, 2 ),
		new SharedFinfo( "init", initTypes, 2 ),
		new SharedFinfo( "channel", channelTypes, 3 ),
		new SharedFinfo( "axial", axialTypes, 2 ),
		new SharedFinfo( "raxial", raxialTypes, 2 ),

		new DestFinfo( "inject", Ftype1< double >::global(),
			RFCAST( &Compartment::injectFunc ) ),
		
		/// Arguments to randInject are probability and current.
		new DestFinfo( "randInject", Ftype2< double, double >::global(),
			RFCAST( &Compartment::randInjectFunc ) ),
	};

	static Cinfo compartmentCinfo(
				"Compartment",
				"Upi Bhalla",
				"Compartment object, for branching neuron models.",
				initNeutralCinfo(),
				compartmentFinfos,
				sizeof( compartmentFinfos ) / sizeof( Finfo* ),
				ValueFtype1< Compartment >::global()
	);

	return &compartmentCinfo;
}

static const Cinfo* compartmentCinfo = initCompartmentCinfo();

static const unsigned int channelSlot =
	initCompartmentCinfo()->getSlotIndex( "channel" );
static const unsigned int axialSlot =
	initCompartmentCinfo()->getSlotIndex( "axial" );
static const unsigned int raxialSlot =
	initCompartmentCinfo()->getSlotIndex( "raxial" );

//////////////////////////////////////////////////////////////////
// Here we put the Compartment class functions.
//////////////////////////////////////////////////////////////////

bool Compartment::rangeWarning( const Conn& c, const string& field, double value )
{
	if ( value < Compartment::EPSILON ) {
		cout << "Warning: Ignored attempt to set " << field <<
				" of compartment " <<
				c.targetElement()->name() << 
				" to less than " << EPSILON << endl;
		return 1;
	}
	return 0;
}

// Value Field access function definitions.
void Compartment::setVm( const Conn& c, double Vm )
{
	static_cast< Compartment* >( c.targetElement()->data() )->Vm_ = Vm;
}

double Compartment::getVm( const Element* e )
{
	return static_cast< Compartment* >( e->data() )->Vm_;
}

void Compartment::setEm( const Conn& c, double Em )
{
	static_cast< Compartment* >( c.targetElement()->data() )->Em_ = Em;
}

double Compartment::getEm( const Element* e )
{
	return static_cast< Compartment* >( e->data() )->Em_;
}

void Compartment::setCm( const Conn& c, double Cm )
{
	if ( rangeWarning( c, "Cm", Cm ) ) return;
	static_cast< Compartment* >( c.targetElement()->data() )->Cm_ = Cm;
}

double Compartment::getCm( const Element* e )
{
	return static_cast< const Compartment* >( e->data() )->Cm_;
}

void Compartment::setRm( const Conn& c, double Rm )
{
	if ( rangeWarning( c, "Rm", Rm ) ) return;
	static_cast< Compartment* >( c.targetElement()->data() )->Rm_ = Rm;
	static_cast< Compartment* >( c.targetElement()->data() )->invRm_ =
			1.0/Rm;
}

double Compartment::getRm( const Element* e )
{
	return static_cast< Compartment* >( e->data() )->Rm_;
}

void Compartment::setRa( const Conn& c, double Ra )
{
	if ( rangeWarning( c, "Ra", Ra ) ) return;
	static_cast< Compartment* >( c.targetElement()->data() )->Ra_ = Ra;
}

double Compartment::getRa( const Element* e )
{
	return static_cast< Compartment* >( e->data() )->Ra_;
}

void Compartment::setIm( const Conn& c, double Im )
{
	static_cast< Compartment* >( c.targetElement()->data() )->Im_ = Im;
}

double Compartment::getIm( const Element* e )
{
	return static_cast< Compartment* >( e->data() )->Im_;
}

void Compartment::setInject( const Conn& c, double Inject )
{
	static_cast< Compartment* >( c.targetElement()->data() )->Inject_ =
			Inject;
}

double Compartment::getInject( const Element* e )
{
	return static_cast< Compartment* >( e->data() )->Inject_;
}

void Compartment::setInitVm( const Conn& c, double initVm )
{
	static_cast< Compartment* >( c.targetElement()->data() )->initVm_ =
			initVm;
}

double Compartment::getInitVm( const Element* e )
{
	return static_cast< Compartment* >( e->data() )->initVm_;
}

void Compartment::setDiameter( const Conn& c, double diameter )
{
}

double Compartment::getDiameter( const Element* e )
{
		return 0.0;
}

void Compartment::setLength( const Conn& c, double length )
{
}

double Compartment::getLength( const Element* e )
{
		return 0.0;
}

//////////////////////////////////////////////////////////////////
// Compartment::Dest function definitions.
//////////////////////////////////////////////////////////////////

void Compartment::innerProcessFunc( Element* e, ProcInfo p )
{
	A_ += Inject_ + sumInject_ + Em_ * invRm_; 
	if ( B_ > EPSILON ) {
		double x = exp( -B_ * p->dt_ / Cm_ );
		Vm_ = Vm_ * x + ( A_ / B_ )  * ( 1.0 - x );
	} else {
		Vm_ += ( A_ - Vm_ * B_ ) * p->dt_ / Cm_;
	}
	A_ = 0.0;
	B_ = invRm_; 
	Im_ = 0.0;
	sumInject_ = 0.0;
	// Send out the channel messages
	send2< double, ProcInfo >( e, channelSlot, Vm_, p );
	// Send out the axial messages
	// send1< double >( e, axialSlot, Vm_ );
	// Send out the raxial messages
	// send2< double >( e, raxialSlot, Ra_, Vm_ );
}

void Compartment::processFunc( const Conn& c, ProcInfo p )
{
	Element* e = c.targetElement();
	static_cast< Compartment* >( e->data() )->innerProcessFunc( e, p );
}

void Compartment::innerReinitFunc( Element* e, ProcInfo p )
{
	Vm_ = initVm_;
	A_ = 0.0;
	B_ = invRm_;
	Im_ = 0.0;
	sumInject_ = 0.0;
	// ChannelReinit no longer done from compartments. 
	// send1< double >( e, 4, Vm_ );
}

void Compartment::reinitFunc( const Conn& c, ProcInfo p )
{
	Element* e = c.targetElement();
	static_cast< Compartment* >( e->data() )->innerReinitFunc( e, p );
}

void Compartment::initFunc( const Conn& c, ProcInfo p )
{
	Element* e = c.targetElement();
	Compartment* compt = static_cast< Compartment* >( e->data() );
	// Send out the axial messages
	send1< double >( e, axialSlot, compt->Vm_ );
	// Send out the raxial messages
	send2< double >( e, raxialSlot, compt->Ra_, compt->Vm_ );
}

void Compartment::dummyInitFunc( const Conn& c, ProcInfo p )
{
		; // nothing happens here.
}

void Compartment::channelFunc( const Conn& c, double Gk, double Ek)
{
	Element* e = c.targetElement();
	Compartment* compt = static_cast< Compartment* >( e->data() );
	compt->A_ += Gk * Ek;
	compt->B_ += Gk;
}

void Compartment::innerRaxialFunc( double Ra, double Vm)
{
	A_ += Vm / Ra;
	B_ += 1.0 / Ra;
	Im_ += ( Vm - Vm_ ) / Ra;
}

void Compartment::raxialFunc( const Conn& c, double Ra, double Vm)
{
	static_cast< Compartment* >( c.targetElement()->data() )->
			innerRaxialFunc( Ra, Vm );
}


void Compartment::innerAxialFunc( double Vm)
{
	A_ += Vm / Ra_;
	B_ += 1.0 / Ra_;
	Im_ += ( Vm - Vm_ ) / Ra_;
}

void Compartment::axialFunc( const Conn& c, double Vm)
{
	static_cast< Compartment* >( c.targetElement()->data() )->
			innerAxialFunc( Vm );
}

void Compartment::injectFunc( const Conn& c, double I)
{
	Compartment* compt = static_cast< Compartment* >(
					c.targetElement()->data() );
	compt->sumInject_ += I;
	compt->Im_ += I;
}

void Compartment::randInjectFunc( const Conn& c, double prob, double I)
{
		/*
	if ( mtrand() < prob * dt_ ) {
		Compartment* compt = static_cast< Compartment* >(
					c.targetElement()->data() );
		compt->sumInject_ += i;
		compt->Im_ += i;
	}
	*/
}

/////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include "../element/Neutral.h"

void testCompartment()
{
	cout << "\nTesting Compartment";

	Element* n = Neutral::create( "Neutral", "n", Element::root() );
	Element* c0 = Neutral::create( "Compartment", "c0", n );
	ASSERT( c0 != 0, "creating compartment" );
	ProcInfoBase p;
	Conn c( c0, 0 );
	p.dt_ = 0.002;
	Compartment::setInject( c, 1.0 );
	Compartment::setRm( c, 1.0 );
	Compartment::setRa( c, 0.0025 );
	Compartment::setCm( c, 1.0 );
	Compartment::setEm( c, 0.0 );
	Compartment::setVm( c, 0.0 );

	// First, test charging curve for a single compartment
	// We want our charging curve to be a nice simple exponential
	// Vm = 1.0 - 1.0 * exp( - t / 1.0 );
	double delta = 0.0;
	double Vm = 0.0;
	double x = 0.0;
	double tau = 1.0;
	double Vmax = 1.0;
	for ( p.currTime_ = 0.0; p.currTime_ < 2.0; p.currTime_ += p.dt_ ) 
	{
		Vm = Compartment::getVm( c0 );
		x = Vmax - Vmax * exp( -p.currTime_ / tau );
		delta += ( Vm - x ) * ( Vm - x );
		Compartment::processFunc( c, &p );
	}
	ASSERT( delta < 1.0e-6, "Testing compartment time" );

	// Second, test the spatial spread of charge.
	// We make the cable long enough to get another nice exponential.
	// Vm = Vm0 * exp( -x/lambda)
	// lambda = sqrt( Rm/Ra ) where these are the actual values, not
	// the values per unit length.
	// So here lambda = 20, so that each compt is lambda/20
	double Rm = 1.0;
	double Ra = 0.0025;
	double lambda = sqrt( Rm / Ra );
	unsigned int i;
	Element* compts[100];
	compts[0] = c0;
	const Finfo* axial = c0->findFinfo( "axial" );
	const Finfo* raxial = c0->findFinfo( "raxial" );
	Compartment::setVm( c, 0.0 );
	Compartment::setInject( c, 20.5 );
	for (i = 1; i < 100; i++ ) {
		char name[20];
		sprintf( name, "c%d", i );
		compts[i] = Neutral::create( "Compartment", "c0", n );
		Conn temp( compts[i], 0 );
		Compartment::setInject( temp, 0.0 );
		Compartment::setRm( temp, Rm );
		Compartment::setRa( temp, Ra );
		Compartment::setCm( temp, 1.0 );
		Compartment::setEm( temp, 0.0 );
		Compartment::setVm( temp, 0.0 );

		assert( raxial->add( compts[i - 1], compts[i], axial ) ); 
	}
	ASSERT( 1, "messaging in compartments" );

	for ( p.currTime_ = 0.0; p.currTime_ < 10.0; p.currTime_ += p.dt_ ) 
	{
		for (i = 0; i < 100; i++ ) {
			Conn temp( compts[i], 0 );
			Compartment::processFunc( temp, &p );
			Compartment::initFunc( temp, &p );
		}
	}

	delta = 0.0;
	// We measure only the first 50 compartments as later we 
	// run into end effects because it is not an infinite cable
	for (i = 0; i < 50; i++ ) {
		Vm = Compartment::getVm( compts[i] );
		x = Vmax * exp( - static_cast< double >( i ) / lambda );
		delta += ( Vm - x ) * ( Vm - x );
	}
	// Error here is larger because it isn't an infinite cable.
	ASSERT( delta < 1.0e-5, "Testing compartment space" );

	// Get rid of all the compartments.
	set( n, "destroy" );
}
#endif
