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

using namespace moose;
const double Compartment::EPSILON = 1.0e-15;

/**
 * The initCompartmentCinfo() function sets up the Compartment class.
 * This function uses the common trick of having an internal
 * static value which is created the first time the function is called.
 * There are several static arrays set up here. The ones which
 * use SharedFinfos are for shared messages where multiple kinds
 * of information go along the same connection.
 */
const Cinfo* initCompartmentCinfo()
{
	
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
				RFCAST( &Compartment::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
				RFCAST( &Compartment::reinitFunc ) ),
	};
	static Finfo* process =	new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo* ),
			"This is a shared message to receive Process messages "
			"from the scheduler objects. The first entry is a MsgDest "
			"for the Process operation. It has a single argument, "
			"ProcInfo, which holds lots of information about current "
			"time, thread, dt and so on. The second entry is a MsgDest "
			"for the Reinit operation. It also uses ProcInfo. " );
	
	 static Finfo* initShared[] =
	{
		new DestFinfo( "init", Ftype1< ProcInfo >::global(),
				RFCAST( &Compartment::initFunc ) ),
		new DestFinfo( "initReinit", Ftype1< ProcInfo >::global(),
				RFCAST( &Compartment::initReinitFunc ) ),
	};
	static Finfo* init = new SharedFinfo( "init", initShared,
			sizeof( initShared ) / sizeof( Finfo* ),
			"This is a shared message to receive Init messages from "
			"the scheduler objects. Its job is to separate the "
			"compartmental calculations from the message passing. "
			"It doesn't really need to be shared, as it does not use "
			"the reinit part, but the scheduler objects expect this "
			"form of message for all scheduled output. The first "
			"entry is a MsgDest for the Process operation. It has a "
			"single argument, ProcInfo, which holds lots of "
			"information about current time, thread, dt and so on. "
			"The second entry is a dummy MsgDest for the Reinit "
			"operation. It also uses ProcInfo. " );

	static Finfo* channelShared[] =
	{
		new DestFinfo( "channel", Ftype2< double, double >::global(),
				RFCAST( &Compartment::channelFunc ) ),
		new SrcFinfo( "Vm", Ftype1< double >::global() ),
	};

	static Finfo* axialShared[] =
	{
		new SrcFinfo( "axialSrc", Ftype1< double >::global() ),
		new DestFinfo( "handleRaxial", Ftype2< double, double >::global(),
				RFCAST( &Compartment::raxialFunc ) ),
	};

	static Finfo* raxialShared[] =
	{
		new DestFinfo( "handleAxial", Ftype1< double >::global(),
				RFCAST( &Compartment::axialFunc ) ),
		new SrcFinfo( "raxialSrc", Ftype2< double, double >::global() )
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
		new ValueFinfo( "inject", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getInject ),
			reinterpret_cast< RecvFunc >( &Compartment::setInject )
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
		new ValueFinfo( "x0", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getX0 ),
			reinterpret_cast< RecvFunc >( &Compartment::setX0 )
		),
		new ValueFinfo( "y0", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getY0 ),
			reinterpret_cast< RecvFunc >( &Compartment::setY0 )
		),
		new ValueFinfo( "z0", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getZ0 ),
			reinterpret_cast< RecvFunc >( &Compartment::setZ0 )
		),
		new ValueFinfo( "x", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getX ),
			reinterpret_cast< RecvFunc >( &Compartment::setX )
		),
		new ValueFinfo( "y", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getY ),
			reinterpret_cast< RecvFunc >( &Compartment::setY )
		),
		new ValueFinfo( "z", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &Compartment::getZ ),
			reinterpret_cast< RecvFunc >( &Compartment::setZ )
		),

	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		// Sends out the membrane potential. Used for SpikeGen.
		new SrcFinfo( "VmSrc", Ftype1< double >::global() ),
                new SrcFinfo( "ImSrc", Ftype1< double >::global() ),
	//////////////////////////////////////////////////////////////////
	// SharedFinfo definitions
	//////////////////////////////////////////////////////////////////
		process,
		init,
		/*
		new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "init", initShared,
			sizeof( initShared ) / sizeof( Finfo* ) ),
		*/
		new SharedFinfo( "axial", axialShared,
			sizeof( axialShared ) / sizeof( Finfo* ),
			"This is a shared message between asymmetric compartments. "
			"axial messages (this kind) connect up to raxial "
			"messages (defined below). The soma should use raxial "
			"messages to connect to the axial message of all the "
			"immediately adjacent dendritic compartments.This puts "
			"the (low) somatic resistance in series with these "
			"dendrites. Dendrites should then use raxial messages to"
			"connect on to more distal dendrites. In other words, "
			"raxial messages should face outward from the soma. "
			"The first entry is a MsgSrc sending Vm to the axialFunc"
			"of the target compartment. The second entry is a MsgDest "
			"for the info coming from the other compt. It expects "
			"Ra and Vm from the other compt as args. Note that the "
			"message is named after the source type. " ),
		new SharedFinfo( "raxial", raxialShared,
			sizeof( raxialShared ) / sizeof( Finfo* ),
			"This is a raxial shared message between asymmetric "
			"compartments. The first entry is a MsgDest for the info "
			"coming from the other compt. It expects Vm from the "
			"other compt as an arg. The second is a MsgSrc sending "
			"Ra and Vm to the raxialFunc of the target compartment. " ),
		new SharedFinfo( "channel", channelShared,
			sizeof( channelShared ) / sizeof( Finfo* ),
			"This is a shared message from a compartment to channels. "
			"The first entry is a MsgDest for the info coming from "
			"the channel. It expects Gk and Ek from the channel "
			"as args. The second entry is a MsgSrc sending Vm " ),
		/*
		new SharedFinfo( "process", processTypes, 2 ),
		new SharedFinfo( "init", initTypes, 2 ),
		new SharedFinfo( "channel", channelTypes, 2 ),
		new SharedFinfo( "axial", axialTypes, 2 ),
		new SharedFinfo( "raxial", raxialTypes, 2 ),
		*/

	//////////////////////////////////////////////////////////////////
	// DestFinfo definitions
	//////////////////////////////////////////////////////////////////
		new DestFinfo( "injectMsg", Ftype1< double >::global(),
			RFCAST( &Compartment::injectMsgFunc ), 
			"The injectMsg corresponds to the INJECT message in the "
			"GENESIS compartment. It does different things from the "
			"inject field, and perhaps should just be merged in. In "
			"particular, it needs to be updated every dt to have an effect. " ),
		
		new DestFinfo( "randInject", Ftype2< double, double >::global(),
			RFCAST( &Compartment::randInjectFunc ),
			"Arguments to randInject are probability and current." ),
		new DestFinfo( "cable", Ftype0::global(),&dummyFunc,
			"message from compartment to its cable" ),
			
	};

	// This sets up two clocks: first a process clock at stage 0, tick 0,
	// then an init clock at stage 0, tick 1.
	static SchedInfo schedInfo[] = { { process, 0, 0 }, { init, 0, 1 } };

	static string doc[] =
	{
		"Name", "Compartment",
		"Author", "Upi Bhalla",
		"Description", "Compartment object, for branching neuron models.",
	};	
	static Cinfo compartmentCinfo(
				doc,
				sizeof( doc ) / sizeof( string ),
				initNeutralCinfo(),
				compartmentFinfos,
				sizeof( compartmentFinfos ) / sizeof( Finfo* ),
				ValueFtype1< Compartment >::global(),
				schedInfo, 2
	);

	return &compartmentCinfo;
}

static const Cinfo* compartmentCinfo = initCompartmentCinfo();

static const Slot channelSlot =
	initCompartmentCinfo()->getSlot( "channel.Vm" );
static const Slot axialSlot =
	initCompartmentCinfo()->getSlot( "axial.axialSrc" );
static const Slot raxialSlot =
	initCompartmentCinfo()->getSlot( "raxial.raxialSrc" );
static const Slot VmSlot =
	initCompartmentCinfo()->getSlot( "VmSrc" );
static const Slot ImSlot =
	initCompartmentCinfo()->getSlot( "ImSrc" );

//////////////////////////////////////////////////////////////////
// Here we put the Compartment class functions.
//////////////////////////////////////////////////////////////////

Compartment::Compartment()
{
	Vm_ = -0.06;
	Em_ = -0.06;
	Cm_ = 1.0;
	Rm_ = 1.0;
	invRm_ = 1.0;
	Ra_ = 1.0;
	Im_ = 0.0;
	Inject_ = 0.0;
	sumInject_ = 0.0;
	initVm_ = -0.06;
	A_ = 0.0;
	B_ = 0.0;
	x_ = 0.0;
	y_ = 0.0;
	z_ = 0.0;
	x0_ = 0.0;
	y0_ = 0.0;
	z0_ = 0.0;
	diameter_ = 0.0;
	length_ = 0.0;
}

bool Compartment::rangeWarning( const Conn* c, const string& field, double value )
{
	if ( value < Compartment::EPSILON ) {
		cout << "Warning: Ignored attempt to set " << field <<
				" of compartment " <<
				c->target().e->name() << 
				" to less than " << EPSILON << endl;
		return 1;
	}
	return 0;
}

// Value Field access function definitions.
void Compartment::setVm( const Conn* c, double Vm )
{
	static_cast< Compartment* >( c->data() )->Vm_ = Vm;
}

double Compartment::getVm( Eref e )
{
	return static_cast< Compartment* >( e.data() )->Vm_;
}

void Compartment::setEm( const Conn* c, double Em )
{
	static_cast< Compartment* >( c->data() )->Em_ = Em;
}

double Compartment::getEm( Eref e )
{
	return static_cast< Compartment* >( e.data() )->Em_;
}

void Compartment::setCm( const Conn* c, double Cm )
{
	if ( rangeWarning( c, "Cm", Cm ) ) return;
	static_cast< Compartment* >( c->data() )->Cm_ = Cm;
}

double Compartment::getCm( Eref e )
{
	return static_cast< const Compartment* >( e.data() )->Cm_;
}

void Compartment::setRm( const Conn* c, double Rm )
{
	if ( rangeWarning( c, "Rm", Rm ) ) return;
	static_cast< Compartment* >( c->data() )->Rm_ = Rm;
	static_cast< Compartment* >( c->data() )->invRm_ =
			1.0/Rm;
}

double Compartment::getRm( Eref e )
{
	return static_cast< Compartment* >( e.data() )->Rm_;
}

void Compartment::setRa( const Conn* c, double Ra )
{
	if ( rangeWarning( c, "Ra", Ra ) ) return;
	static_cast< Compartment* >( c->data() )->Ra_ = Ra;
}

double Compartment::getRa( Eref e )
{
	return static_cast< Compartment* >( e.data() )->Ra_;
}

void Compartment::setIm( const Conn* c, double Im )
{
	static_cast< Compartment* >( c->data() )->Im_ = Im;
}

double Compartment::getIm( Eref e )
{
	return static_cast< Compartment* >( e.data() )->Im_;
}

void Compartment::setInject( const Conn* c, double Inject )
{
	static_cast< Compartment* >( c->data() )->Inject_ =
			Inject;
}

double Compartment::getInject( Eref e )
{
	return static_cast< Compartment* >( e.data() )->Inject_;
}

void Compartment::setInitVm( const Conn* c, double initVm )
{
	static_cast< Compartment* >( c->data() )->initVm_ =
			initVm;
}

double Compartment::getInitVm( Eref e )
{
	return static_cast< Compartment* >( e.data() )->initVm_;
}

void Compartment::setDiameter( const Conn* c, double value )
{
	static_cast< Compartment* >( c->data() )->
			diameter_ = value;
}

double Compartment::getDiameter( Eref e )
{
	return static_cast< Compartment* >( e.data() )->diameter_;
}

void Compartment::setLength( const Conn* c, double value )
{
	static_cast< Compartment* >( c->data() )->length_ =
			value;
}

double Compartment::getLength( Eref e )
{
	return static_cast< Compartment* >( e.data() )->length_;
}

void Compartment::setX0( const Conn* c, double value )
{
	static_cast< Compartment* >( c->data() )->x0_ =
			value;
}

double Compartment::getX0( Eref e )
{
	return static_cast< Compartment* >( e.data() )->x0_;
}

void Compartment::setY0( const Conn* c, double value )
{
	static_cast< Compartment* >( c->data() )->y0_ =
			value;
}

double Compartment::getY0( Eref e )
{
	return static_cast< Compartment* >( e.data() )->y0_;
}

void Compartment::setZ0( const Conn* c, double value )
{
	static_cast< Compartment* >( c->data() )->z0_ =
			value;
}

double Compartment::getZ0( Eref e )
{
	return static_cast< Compartment* >( e.data() )->z0_;
}

void Compartment::setX( const Conn* c, double value )
{
	static_cast< Compartment* >( c->data() )->x_ =
			value;
}

double Compartment::getX( Eref e )
{
	return static_cast< Compartment* >( e.data() )->x_;
}

void Compartment::setY( const Conn* c, double value )
{
	static_cast< Compartment* >( c->data() )->y_ =
			value;
}

double Compartment::getY( Eref e )
{
	return static_cast< Compartment* >( e.data() )->y_;
}

void Compartment::setZ( const Conn* c, double value )
{
	static_cast< Compartment* >( c->data() )->z_ =
			value;
}

double Compartment::getZ( Eref e )
{
	return static_cast< Compartment* >( e.data() )->z_;
}

//////////////////////////////////////////////////////////////////
// Compartment::Dest function definitions.
//////////////////////////////////////////////////////////////////

void Compartment::innerProcessFunc( Eref e, ProcInfo p )
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
	sumInject_ = 0.0;
        send1< double >(e, ImSlot, Im_); // for efield objects
	Im_ = 0.0;
	// Send out the channel messages
	send1< double >( e, channelSlot, Vm_ );
	// Send out the message to any SpikeGens.
	send1< double >( e, VmSlot, Vm_ );
	// Send out the axial messages
	// send1< double >( e, axialSlot, Vm_ );
	// Send out the raxial messages
	// send2< double >( e, raxialSlot, Ra_, Vm_ );
}

void Compartment::processFunc( const Conn* c, ProcInfo p )
{
	static_cast< Compartment* >( c->data() )->
		innerProcessFunc( c->target(), p );
}

void Compartment::innerReinitFunc(  Eref e, ProcInfo p )
{
#ifndef NDEBUG
   // cout << "Compartment::innerReinitFunc:: " << e.id().path() << endl;
#endif
	Vm_ = initVm_;
	A_ = 0.0;
	B_ = invRm_;
	Im_ = 0.0;
	sumInject_ = 0.0;
	// Send the Vm over to the channels at reset.
	send1< double >( e, channelSlot, Vm_ );
	// Send the Vm over to the SpikeGen
	send1< double >( e, VmSlot, Vm_ );
}

void Compartment::reinitFunc( const Conn* c, ProcInfo p )
{
	// cout << "Compt reinit: " << c->target().name() << endl << flush;
	static_cast< Compartment* >( c->data() )->
		innerReinitFunc( c->target(), p );
}

void Compartment::initFunc( const Conn* c, ProcInfo p )
{
	static_cast< Compartment* >( c->data() )->
		innerInitFunc( c->target(), p );
}

void Compartment::innerInitFunc( Eref e, ProcInfo p )
{
	// Send out the axial messages
	send1< double >( e, axialSlot, Vm_ );
	// Send out the raxial messages
	send2< double >( e, raxialSlot, Ra_, Vm_ );
}

void Compartment::initReinitFunc( const Conn* c, ProcInfo p )
{
	static_cast< Compartment* >( c->data() )->
		innerInitReinitFunc( c->target(), p );
}

void Compartment::innerInitReinitFunc( Eref e, ProcInfo p )
{
	; // Nothing happens here
}

void Compartment::channelFunc( const Conn* c, double Gk, double Ek)
{
	Compartment* compt = static_cast< Compartment* >( c->data() );
	compt->A_ += Gk * Ek;
	compt->B_ += Gk;
}

void Compartment::innerRaxialFunc( double Ra, double Vm)
{
	A_ += Vm / Ra;
	B_ += 1.0 / Ra;
	Im_ += ( Vm - Vm_ ) / Ra;
}

void Compartment::raxialFunc( const Conn* c, double Ra, double Vm)
{
	static_cast< Compartment* >( c->data() )->
			innerRaxialFunc( Ra, Vm );
}


void Compartment::innerAxialFunc( double Vm)
{
	A_ += Vm / Ra_;
	B_ += 1.0 / Ra_;
	Im_ += ( Vm - Vm_ ) / Ra_;
}

void Compartment::axialFunc( const Conn* c, double Vm)
{
	static_cast< Compartment* >( c->data() )->
			innerAxialFunc( Vm );
}

void Compartment::injectMsgFunc( const Conn* c, double I)
{
	Compartment* compt = static_cast< Compartment* >(
					c->data() );
	compt->sumInject_ += I;
	compt->Im_ += I;
}

void Compartment::randInjectFunc( const Conn* c, double prob, double I)
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
// Comment out this define if it takes too long (about 5 seconds on
// a modest machine, but could be much longer with valgrind)
// #define DO_SPATIAL_TESTS
#include "../element/Neutral.h"

void testCompartment()
{
	cout << "\nTesting Compartment" << flush;

	Element* n = Neutral::create( "Neutral", "n", Element::root()->id(), 
		Id::scratchId() );
	Element* c0 = Neutral::create( "Compartment", "c0", n->id(), 
		Id::scratchId() );
	ASSERT( c0 != 0, "creating compartment" );
	ProcInfoBase p;
	SetConn c( c0, 0 );
	p.dt_ = 0.002;
	Compartment::setInject( &c, 1.0 );
	Compartment::setRm( &c, 1.0 );
	Compartment::setRa( &c, 0.0025 );
	Compartment::setCm( &c, 1.0 );
	Compartment::setEm( &c, 0.0 );
	Compartment::setVm( &c, 0.0 );

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
		Compartment::processFunc( &c, &p );
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
	unsigned int i;
	Element* compts[100];
	compts[0] = c0;
	// const Finfo* axial = c0->findFinfo( "axial" );
	// const Finfo* raxial = c0->findFinfo( "raxial" );
	Compartment::setVm( &c, 0.0 );
	Compartment::setInject( &c, 20.5 );
	bool ret;
	for (i = 1; i < 100; i++ ) {
		char name[20];
		sprintf( name, "c%d", i );
		compts[i] = Neutral::create( "Compartment", name, n->id(), 
			Id::scratchId() );
		SetConn temp( compts[i], 0 );
		Compartment::setInject( &temp, 0.0 );
		Compartment::setRm( &temp, Rm );
		Compartment::setRa( &temp, Ra );
		Compartment::setCm( &temp, 1.0 );
		Compartment::setEm( &temp, 0.0 );
		Compartment::setVm( &temp, 0.0 );

		ret = Eref( compts[i - 1] ).add( "raxial", compts[i], "axial" ); 
		// ret = raxial->add( compts[i - 1], compts[i], axial ); 
		assert( ret );
	}
	ASSERT( 1, "messaging in compartments" );
	ASSERT( n->numTargets( "childSrc" ) == 100, "Check children" );

#ifdef DO_SPATIAL_TESTS
	double lambda = sqrt( Rm / Ra );

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
#endif
	// Get rid of all the compartments.
	set( n, "destroy" );
}
#endif
