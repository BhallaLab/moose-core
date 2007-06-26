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
#include "HHChannel.h"
#include "../element/Neutral.h"
#include "DeletionMarkerFinfo.h"
#include "GlobalMarkerFinfo.h"

const double HHChannel::EPSILON = 1.0e-10;
const int HHChannel::INSTANT_X = 1;
const int HHChannel::INSTANT_Y = 2;
const int HHChannel::INSTANT_Z = 4;

const Cinfo* initHHChannelCinfo()
{
	/** 
	 * This is a shared message to receive Process message from
	 * the scheduler.
	 * The first entry is a MsgDest for the Process operation. It
	 * has a single argument, ProcInfo, which holds
	 * lots of information about current time, thread, dt and so on.
	 * The second entry is a MsgDest for the Reinit operation. It
	 * also uses ProcInfo.
	 */
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
				RFCAST( &HHChannel::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
				RFCAST( &HHChannel::reinitFunc ) ),
	};

	/**
	 * This is a shared message to couple channel to compartment.
	 * The first entry is a MsgSrc to send Gk and Ek to the compartment
	 * The second entry is a MsgDest for Vm from the compartment.
	 */
	static Finfo* channelShared[] =
	{
		new SrcFinfo( "channel", Ftype2< double, double >::global() ),
		new DestFinfo( "Vm", Ftype1< double >::global(), 
				RFCAST( &HHChannel::channelFunc ) ),
	};

	/**
	 * This is a shared message to communicate with the X gate.
	 * Sends out Vm.
	 * Receives lookedup up values A and B for the Vm.
	 * The A term is the alpha term from HH equations.
	 * The B term is actually alpha + beta, precalculated.
	 */
	static Finfo* xGateShared[] =
	{
		new SrcFinfo( "Vm", Ftype1< double >::global() ),
		new DestFinfo( "gate", Ftype2< double, double >::global(),
				RFCAST( &HHChannel::xGateFunc ) ),
	};

	/**
	 * Shared message for Y gate. Fields as in X gate.
	 */
	static Finfo* yGateShared[] =
	{
		new SrcFinfo( "Vm", Ftype1< double >::global() ),
		new DestFinfo( "gate", Ftype2< double, double >::global(),
				RFCAST( &HHChannel::yGateFunc ) ),
	};

	/**
	 * Shared message for Z gate. Fields as in X gate.
	 */
	static Finfo* zGateShared[] =
	{
		new SrcFinfo( "Vm", Ftype1< double >::global() ),
		new DestFinfo( "gate", Ftype2< double, double >::global(),
			RFCAST( &HHChannel::zGateFunc ) ),
	};

///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////

	static Finfo* HHChannelFinfos[] =
	{
		new ValueFinfo( "Gbar", ValueFtype1< double >::global(),
			GFCAST( &HHChannel::getGbar ), 
			RFCAST( &HHChannel::setGbar )
		),
		new ValueFinfo( "Ek", ValueFtype1< double >::global(),
			GFCAST( &HHChannel::getEk ), 
			RFCAST( &HHChannel::setEk )
		),
		new ValueFinfo( "Xpower", ValueFtype1< double >::global(),
			GFCAST( &HHChannel::getXpower ), 
			RFCAST( &HHChannel::setXpower )
		),
		new ValueFinfo( "Ypower", ValueFtype1< double >::global(),
			GFCAST( &HHChannel::getYpower ), 
			RFCAST( &HHChannel::setYpower )
		),
		new ValueFinfo( "Zpower", ValueFtype1< double >::global(),
			GFCAST( &HHChannel::getZpower ), 
			RFCAST( &HHChannel::setZpower )
		),
		new ValueFinfo( "instant", ValueFtype1< int >::global(),
			GFCAST( &HHChannel::getInstant ), 
			RFCAST( &HHChannel::setInstant )
		),
		new ValueFinfo( "Gk", ValueFtype1< double >::global(),
			GFCAST( &HHChannel::getGk ), 
			RFCAST( &HHChannel::setGk )
		),
		new ValueFinfo( "Ik", ValueFtype1< double >::global(),
			GFCAST( &HHChannel::getIk ), 
			RFCAST( &HHChannel::setIk )
		),
		new ValueFinfo( "X", ValueFtype1< double >::global(),
			GFCAST( &HHChannel::getX ), 
			RFCAST( &HHChannel::setX )
		),
		new ValueFinfo( "Y", ValueFtype1< double >::global(),
			GFCAST( &HHChannel::getY ), 
			RFCAST( &HHChannel::setY )
		),
		new ValueFinfo( "Z", ValueFtype1< double >::global(),
			GFCAST( &HHChannel::getZ ), 
			RFCAST( &HHChannel::setZ )
		),
		new ValueFinfo( "useConcentration",
			ValueFtype1< int >::global(),
			GFCAST( &HHChannel::getUseConcentration ), 
			&dummyFunc
		),
///////////////////////////////////////////////////////
// Shared message definitions
///////////////////////////////////////////////////////
		new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "channel", channelShared,
			sizeof( channelShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "xGate", xGateShared,
			sizeof( xGateShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "yGate", yGateShared,
			sizeof( yGateShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "zGate", zGateShared,
			sizeof( zGateShared ) / sizeof( Finfo* ) ),

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
		new SrcFinfo( "IkSrc", Ftype1< double >::global() ),

///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
		new DestFinfo( "concen", Ftype1< double >::global(),
				RFCAST( &HHChannel::concFunc ) ),
	};

	static Cinfo HHChannelCinfo(
		"HHChannel",
		"Upinder S. Bhalla, 2007, NCBS",
		"HHChannel: Hodgkin-Huxley type voltage-gated Ion channel. Something\nlike the old tabchannel from GENESIS, but also presents\na similar interface as hhchan from GENESIS. ",
		initNeutralCinfo(),
		HHChannelFinfos,
		sizeof( HHChannelFinfos )/sizeof(Finfo *),
		ValueFtype1< HHChannel >::global()
	);

	return &HHChannelCinfo;
}

static const Cinfo* hhChannelCinfo = initHHChannelCinfo();

static const unsigned int channelSlot =
	initHHChannelCinfo()->getSlotIndex( "channel.channel" );
static const unsigned int xGateSlot =
	initHHChannelCinfo()->getSlotIndex( "xGate.Vm" );
static const unsigned int yGateSlot =
	initHHChannelCinfo()->getSlotIndex( "yGate.Vm" );
static const unsigned int zGateSlot =
	initHHChannelCinfo()->getSlotIndex( "zGate.Vm" );
static const unsigned int ikSlot =
	initHHChannelCinfo()->getSlotIndex( "IkSrc" );


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void HHChannel::setGbar( const Conn& c, double Gbar )
{
	static_cast< HHChannel* >( c.data() )->Gbar_ = Gbar;
}
double HHChannel::getGbar( const Element* e )
{
	return static_cast< HHChannel* >( e->data() )->Gbar_;
}

void HHChannel::setEk( const Conn& c, double Ek )
{
	static_cast< HHChannel* >( c.data() )->Ek_ = Ek;
}
double HHChannel::getEk( const Element* e )
{
	return static_cast< HHChannel* >( e->data() )->Ek_;
}

/**
 * Assigns the power for a specific gate, identified by the Finfo.
 * Assumes that this is a different power from the old one.
 * 
 * If the gate exists and has only this element for input, then change
 * the gate power.
 * If the gate exists and has multiple parents, then make a new gate,
 * 	set its power.
 * If the gate does not exist, make a new gate, set its power.
 *
 * Note that if the power is zero, then the gate has to be removed.
 *
 * The function is designed with the idea that if copies of this
 * channel are made, then they all point back to the original HHGate.
 * (Unless they are cross-node copies).
 * It is only if we subsequently alter the HHGate of this channel that
 * we need to make our own variant of the HHGate, or disconnect from
 * an existing one.
 */

void HHChannel::makeGate( Element *e, const Finfo* f, double power )
{
	Element* gate = 0;	
	vector< Conn > list;
	unsigned int numGates = f->outgoingConns( e, list );
	assert( numGates <= 1 );
	if ( power <= 0 ) {
		// If gate exists, remove it.
		if ( numGates == 1 ) {
			gate = list[0].targetElement();
			unsigned int numChans =
					gate->findFinfo( "gate" )->numOutgoing( gate );
			assert( numChans > 0 );
			if ( numChans > 1 ) {
				// Here we have multiple channels using this gate. So
				// we don't mess with the original.
				// Get rid of local connection to gate, but don't delete
				f->dropAll( e );
			} else { // Delete the single gate.
				bool ret = set( gate, "destroy" );
				assert( ret );
			}
		}
		return;
	}

	if ( numGates == 1 ) {
		gate = list[0].targetElement();
		unsigned int numChans =
				gate->findFinfo( "gate" )->numOutgoing( gate );
		assert( numChans > 0 );
		if ( numChans > 1 ) {
			// Here we have multiple channels using this gate. So
			// we don't mess with the original.
			// make a new gate which we can change.
			gate = Neutral::create( "HHGate", "xGate", e );
			gate->addFinfo( GlobalMarkerFinfo::global() );
			bool ret = f->add( e, gate, gate->findFinfo( "gate" ) );
			assert( ret );
		}
	} else { // No gate, make a new one.
		gate = Neutral::create( "HHGate", f->name(), e );
		// Make it a global so that duplicates do not happen unless
		// the table values change.
		gate->addFinfo( GlobalMarkerFinfo::global() );
		bool ret = f->add( e, gate, gate->findFinfo( "gate" ) );
		assert( ret );
	}
}

/**
 * Assigns the Xpower for this gate. If the gate exists and has
 * only this element for input, then change the gate value.
 * If the gate exists and has multiple parents, then make a new gate.
 * If the gate does not exist, make a new gate
 */
void HHChannel::setXpower( const Conn& c, double Xpower )
{
	Element* e = c.targetElement();
	HHChannel* chan = static_cast< HHChannel* >( c.data() );

	if ( Xpower == chan->Xpower_ ) return;
	makeGate( e, e->findFinfo( "xGate" ), Xpower );
	chan->Xpower_ = Xpower;
	chan->takeXpower_ = selectPower( Xpower );
}
double HHChannel::getXpower( const Element* e )
{
	return static_cast< HHChannel* >( e->data() )->Xpower_;
}

void HHChannel::setYpower( const Conn& c, double Ypower )
{
	Element* e = c.targetElement();
	HHChannel* chan = static_cast< HHChannel* >( c.data() );

	if ( Ypower == chan->Ypower_ ) return;
	makeGate( e, e->findFinfo( "yGate" ), Ypower );
	chan->Ypower_ = Ypower;
	chan->takeYpower_ = selectPower( Ypower );
}
double HHChannel::getYpower( const Element* e )
{
	return static_cast< HHChannel* >( e->data() )->Ypower_;
}

void HHChannel::setZpower( const Conn& c, double Zpower )
{
	Element* e = c.targetElement();
	HHChannel* chan = static_cast< HHChannel* >( c.data() );

	if ( Zpower == chan->Zpower_ ) return;
	makeGate( e, e->findFinfo( "zGate" ), Zpower );
	chan->Zpower_ = Zpower;
	chan->takeZpower_ = selectPower( Zpower );
}
double HHChannel::getZpower( const Element* e )
{
	return static_cast< HHChannel* >( e->data() )->Zpower_;
}


void HHChannel::setInstant( const Conn& c, int instant )
{
	static_cast< HHChannel* >( c.data() )->instant_ = instant;
}
int HHChannel::getInstant( const Element* e )
{
	return static_cast< HHChannel* >( e->data() )->instant_;
}

void HHChannel::setGk( const Conn& c, double Gk )
{
	static_cast< HHChannel* >( c.data() )->Gk_ = Gk;
}
double HHChannel::getGk( const Element* e )
{
	return static_cast< HHChannel* >( e->data() )->Gk_;
}

void HHChannel::setIk( const Conn& c, double Ik )
{
	static_cast< HHChannel* >( c.data() )->Ik_ = Ik;
}
double HHChannel::getIk( const Element* e )
{
	return static_cast< HHChannel* >( e->data() )->Ik_;
}

void HHChannel::setX( const Conn& c, double X )
{
	static_cast< HHChannel* >( c.data() )->X_ = X;
}
double HHChannel::getX( const Element* e )
{
	return static_cast< HHChannel* >( e->data() )->X_;
}

void HHChannel::setY( const Conn& c, double Y )
{
	static_cast< HHChannel* >( c.data() )->Y_ = Y;
}
double HHChannel::getY( const Element* e )
{
	return static_cast< HHChannel* >( e->data() )->Y_;
}

void HHChannel::setZ( const Conn& c, double Z )
{
	static_cast< HHChannel* >( c.data() )->Z_ = Z;
}
double HHChannel::getZ( const Element* e )
{
	return static_cast< HHChannel* >( e->data() )->Z_;
}

int HHChannel::getUseConcentration( const Element* e )
{
	return static_cast< HHChannel* >( e->data() )->useConcentration_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

/**
 * Returns the state variable for the new timestep based on
 * the internal variables A_ and B_ which were passed in from the gate.
 */
double HHChannel::integrate( double state, double dt )
{
	if ( B_ > EPSILON ) {
		double x = exp( -B_ * dt );
		return state * x + ( A_ / B_ ) * ( 1 - x );
	}
	return state + A_ * dt ;
}

void HHChannel::processFunc( const Conn& c, ProcInfo p )
{
	Element* e = c.targetElement();
	static_cast< HHChannel* >( e->data() )->innerProcessFunc( e, p );
}

void HHChannel::innerProcessFunc( Element* e, ProcInfo info )
{
	g_ += Gbar_;
	if ( Xpower_ > 0 ) {
		// The looked up table values A_ and B_ come back from the gate
		// right away after these 'send' calls.
		send1< double >( e, xGateSlot, Vm_ );
		if ( instant_ & INSTANT_X )
			X_ = A_/B_;
		else 
			X_ = integrate( X_, info->dt_ );
		g_ *= takeXpower_( X_, Xpower_ );
	}

	if ( Ypower_ > 0 ) {
		send1< double >( e, yGateSlot, Vm_ );
		if ( instant_ & INSTANT_Y )
			Y_ = A_/B_;
		else 
			Y_ = integrate( Y_, info->dt_ );

		g_ *= takeYpower_( Y_, Ypower_ );
	}

	if ( Zpower_ > 0 ) {
		if ( useConcentration_ )
			send1< double >( e, zGateSlot, conc_ );
		else
			send1< double >( e, zGateSlot, Vm_ );

		if ( instant_ & INSTANT_Z )
			Z_ = A_/B_;
		else 
			Z_ = integrate( Z_, info->dt_ );

		g_ *= takeZpower_( Z_, Zpower_ );
	}

	Gk_ = g_;
	send2< double, double >( e, channelSlot, Gk_, Ek_ );
	Ik_ = ( Ek_ - Vm_ ) * g_;

	// This is used if the channel connects up to a conc pool and
	// handles influx of ions giving rise to a concentration change.
	send1< double >( e, ikSlot, Ik_ );
	g_ = 0.0;
}

void HHChannel::reinitFunc( const Conn& c, ProcInfo p )
{
	Element* e = c.targetElement();
	static_cast< HHChannel* >( e->data() )->innerReinitFunc( e, p );
}

/**
 * Here we get the steady-state values for the gate (the 'instant'
 * calculation) as A_/B_.
 */
void HHChannel::innerReinitFunc( Element* e, ProcInfo info )
{
	g_ = Gbar_;

	if ( Xpower_ > 0 ) {
		// The looked up table values A_ and B_ come back from the gate
		// right away after these 'send' calls.
		send1< double >( e, xGateSlot, Vm_ );
		if ( B_ < EPSILON ) {
			cout << "Warning: B_ value for " << e->name() <<
					" is ~0. Check X table\n";
			return;
		}
		X_ = A_/B_;
		g_ *= takeXpower_( X_, Xpower_ );
	}

	if ( Ypower_ > 0 ) {
		send1< double >( e, yGateSlot, Vm_ );
		if ( B_ < EPSILON ) {
			cout << "Warning: B_ value for " << e->name() <<
					" is ~0. Check Y table\n";
			return;
		}
		Y_ = A_/B_;
		g_ *= takeYpower_( Y_, Ypower_ );
	}

	if ( Zpower_ > 0 ) {
		if ( useConcentration_ )
			send1< double >( e, zGateSlot, conc_ );
		else
			send1< double >( e, zGateSlot, Vm_ );
		if ( B_ < EPSILON ) {
			cout << "Warning: B_ value for " << e->name() <<
					" is ~0. Check Z table\n";
			return;
		}
		Z_ = A_/B_;
		g_ *= takeZpower_( Z_, Zpower_ );
	}

	Gk_ = g_;

	send2< double, double >( e, channelSlot, Gk_, Ek_ );
	// channelSrc_.send( Gk_, Ek_ );
	Ik_ = ( Ek_ - Vm_ ) * g_;
	g_ = 0.0;
}

void HHChannel::channelFunc( const Conn& c, double Vm )
{
	static_cast< HHChannel* >( c.data() )->Vm_ = Vm;
}

void HHChannel::concFunc( const Conn& c, double conc )
{
	Element* e = c.targetElement();
	static_cast< HHChannel* >( e->data() )->conc_ = conc;
}

void HHChannel::xGateFunc( const Conn& c, double A, double B )
{
	HHChannel* h =
			static_cast< HHChannel* >( c.targetElement()->data() );
	h->A_ = A;
	h->B_ = B;
}

void HHChannel::yGateFunc( const Conn& c, double A, double B )
{
	HHChannel* h =
			static_cast< HHChannel* >( c.targetElement()->data() );
	h->A_ = A;
	h->B_ = B;
}

void HHChannel::zGateFunc( const Conn& c, double A, double B )
{
	HHChannel* h =
			static_cast< HHChannel* >( c.targetElement()->data() );
	h->A_ = A;
	h->B_ = B;
}

///////////////////////////////////////////////////
// Utility function
///////////////////////////////////////////////////
double HHChannel::powerN( double x, double p )
{
	if ( x > 0.0 )
		return exp( p * log( x ) );
	return 0.0;
}

PFDD HHChannel::selectPower( double power )
{
	if ( power == 0.0 )
		return powerN;
	else if ( power == 1.0 )
		return power1;
	else if ( power == 2.0 )
		return power2;
	else if ( power == 3.0 )
		return power3;
	else if ( power == 4.0 )
		return power4;
	else
		return powerN;
}

///////////////////////////////////////////////////
// Unit tests
///////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include "Compartment.h"

static const double EREST = -0.07;

// AP measured in millivolts wrt EREST, at a sample interval of 
// 100 usec
static double actionPotl[] = { 0,
1.2315, 2.39872, 3.51917, 4.61015, 5.69088, 6.78432, 7.91934, 9.13413,
10.482, 12.0417, 13.9374, 16.3785, 19.7462, 24.7909, 33.0981, 47.967,
73.3833, 98.7377, 105.652, 104.663, 101.815, 97.9996, 93.5261, 88.6248,
83.4831, 78.2458, 73.0175, 67.8684, 62.8405, 57.9549, 53.217, 48.6206,
44.1488, 39.772, 35.4416, 31.0812, 26.5764, 21.7708, 16.4853, 10.6048,
4.30989, -1.60635, -5.965, -8.34834, -9.3682, -9.72711,
-9.81085, -9.78371, -9.71023, -9.61556, -9.50965, -9.39655,
-9.27795, -9.15458, -9.02674, -8.89458, -8.75814, -8.61746,
-8.47254, -8.32341, -8.17008, -8.01258, -7.85093, -7.68517,
-7.51537, -7.34157, -7.16384, -6.98227, -6.79695, -6.60796,
-6.41542, -6.21944, -6.02016, -5.81769, -5.61219, -5.40381,
-5.19269, -4.979, -4.76291, -4.54459, -4.32422, -4.10197,
-3.87804, -3.65259, -3.42582, -3.19791, -2.96904, -2.7394,
-2.50915, -2.27848, -2.04755, -1.81652, -1.58556, -1.3548,
-1.12439, -0.894457, -0.665128, -0.436511, -0.208708, 0.0181928,
};

void lset( 
	Element* gate, const Finfo* table, double val, unsigned int i )
{
	lookupSet< double, unsigned int >( gate, table, val, i );
}

double Na_m_A( double v )
{
	if ( fabs( EREST + 0.025 - v ) < 1e-6 )
		v += 1e-6;
	return  0.1e6 * ( EREST + 0.025 - v ) / ( exp ( ( EREST + 0.025 - v )/ 0.01 ) - 1.0 );
}

double Na_m_B( double v )
{
	return 4.0e3 * exp ( ( EREST - v ) / 0.018 );
}

double Na_h_A( double v )
{
	return 70.0 * exp ( ( EREST - v ) / 0.020 );
}

double Na_h_B( double v )
{
	return 1.0e3 / ( exp ( ( 0.030 + EREST - v )/ 0.01 )  + 1.0 );
}

double K_n_A( double v )
{
	if ( fabs( EREST + 0.025 - v ) < 1e-6 )
		v += 1e-6;
	
	return ( 1.0e4 * ( 0.01 + EREST - v ) ) / ( exp ( ( 0.01 + EREST         - v ) / 0.01 ) - 1.0 );
}

double K_n_B( double v )
{
	return 0.125e3 * exp ( (EREST - v ) / 0.08 );
}

void testHHChannel()
{
	cout << "\nTesting HHChannel";
	// Check message construction with compartment
	Element* n = Neutral::create( "Neutral", "n", Element::root() );
	Element* compt = Neutral::create( "Compartment", "compt", n );
	Element* chan = Neutral::create( "HHChannel", "Na", compt );

	ASSERT( compt->findFinfo( "channel" )->add( compt, chan,
					chan->findFinfo( "channel" ) ),
					"Setting up channel" );

	// Check gate construction and removal when powers are assigned
	
	ASSERT( chan->findFinfo( "childSrc" )->numOutgoing( chan ) == 0,
					"Creating xGate");
	set< double >( chan, "Xpower", 2.0 );
	ASSERT( chan->findFinfo( "xGate" )->numIncoming( chan ) == 1,
					"Creating xGate");
	ASSERT( chan->findFinfo( "childSrc" )->numOutgoing( chan ) == 1,
					"Creating xGate");

	set< double >( chan, "Xpower", 0.0 );
	ASSERT( chan->findFinfo( "childSrc" )->numOutgoing( chan ) == 0,
					"Removing xGate");
	ASSERT( chan->findFinfo( "xGate" )->numIncoming( chan ) == 0,
					"Removing xGate");
	set< double >( chan, "Xpower", 3.0 );
	ASSERT( chan->findFinfo( "xGate" )->numIncoming( chan ) == 1,
					"Creating xGate again");

	Id xGateId;
	bool ret = lookupGet< Id, string >(
		chan, "lookupChild", xGateId, "xGate" );
	ASSERT( ret, "Look up xGate");
	ASSERT( !xGateId.zero() && !xGateId.bad(), "Lookup xGate" );

	Element* xGate = xGateId();

	set< double >( chan, "Ypower", 1.0 );
	ASSERT( chan->findFinfo( "yGate" )->numIncoming( chan ) == 1, "Creating yGate");
	
	// Check steady-state calculation for channel cond on reinit
	// Here we start with Gbar = 1, Vm set ahead of time to 0,
	// xGate giving an X_ state of 2 and yGate of 10 for Vm == 0.
	Id temp;

	////////////////////////////////////////////////////////////////
	// Set up X gates
	////////////////////////////////////////////////////////////////
	// Set up X gate A
	ret = lookupGet< Id, string >( xGate, "lookupChild", temp, "A" );
	ASSERT( ret, "Check gate table" );
	ASSERT( !temp.zero() && !temp.bad(), "xGate_A" );
	Element* xGate_A = temp();
	set< double >( xGate_A, "xmin", -2.0 );
	set< double >( xGate_A, "xmax", 2.0 );
	set< int >( xGate_A, "xdivs", 1 );
	lookupSet< double, unsigned int >( xGate_A, "table", 0.0, 0 );
	lookupSet< double, unsigned int >( xGate_A, "table", 4.0, 1 );

	// Set up X gate B
	ret = lookupGet< Id, string >( xGate, "lookupChild", temp, "B" );
	ASSERT( ret, "Check gate table" );
	ASSERT( !temp.zero() && !temp.bad(), "xGate_B" );
	Element* xGate_B = temp();
	set< double >( xGate_B, "xmin", -1.0 );
	set< double >( xGate_B, "xmax", 1.0 );
	set< int >( xGate_B, "xdivs", 1 );
	lookupSet< double, unsigned int >( xGate_B, "table", 0.0, 0 );
	lookupSet< double, unsigned int >( xGate_B, "table", 2.0, 1 );

	////////////////////////////////////////////////////////////////
	// Set up Y gates
	////////////////////////////////////////////////////////////////
	Id yGateId;
	ret = lookupGet< Id, string >( chan, "lookupChild", yGateId, "yGate" );
	ASSERT( ret, "Look up yGate");
	ASSERT( !yGateId.zero() && !yGateId.bad(), "Lookup yGate" );

	Element* yGate = yGateId();
	ret = lookupGet< Id, string >( yGate, "lookupChild", temp, "A" );
	ASSERT( ret, "Check gate table" );
	ASSERT( !temp.zero() && !temp.bad(), "yGate_A" );
	Element* yGate_A = temp();
	set< double >( yGate_A, "xmin", -2.0 );
	set< double >( yGate_A, "xmax", 2.0 );
	set< int >( yGate_A, "xdivs", 1 );
	lookupSet< double, unsigned int >( yGate_A, "table", 20.0, 0 );
	lookupSet< double, unsigned int >( yGate_A, "table", 0.0, 1 );

	ret = lookupGet< Id, string >( yGate, "lookupChild", temp, "B" );
	ASSERT( ret, "Check gate table" );
	ASSERT( !temp.zero() && !temp.bad(), "yGate_B" );
	Element* yGate_B = temp();
	set< double >( yGate_B, "xmin", -1.0 );
	set< double >( yGate_B, "xmax", 1.0 );
	set< int >( yGate_B, "xdivs", 1 );
	lookupSet< double, unsigned int >( yGate_B, "table", 0.0, 0 );
	lookupSet< double, unsigned int >( yGate_B, "table", 2.0, 1 );

	////////////////////////////////////////////////////////////////
	// Do the Reinit.
	////////////////////////////////////////////////////////////////
	set< double >( chan, "Gbar", 1.0 );
	set< double >( chan, "Ek", 0.0 );
	ProcInfoBase pb;
	pb.dt_ = 0.001;
	Conn c( chan, 0 );
	HHChannel* Na = static_cast< HHChannel* >( c.data() );
	Na->Vm_ = 0.0;

	// This function should do all the reinit steps.
	HHChannel::reinitFunc( c, &pb );
	ASSERT( Na->Gk_ == 80, "Gk_" );
	ASSERT( Na->X_ == 2, "X_" );
	ASSERT( Na->Y_ == 10, "Y_" );

	////////////////////////////////////////////////////////////////
	// Check construction and result of HH squid simulation
	////////////////////////////////////////////////////////////////
	
	Element* kchan = Neutral::create( "HHChannel", "K", compt );
	ASSERT( compt->findFinfo( "channel" )->add( compt, kchan,
					kchan->findFinfo( "channel" ) ),
					"Setting up K channel" );
	static const double VMIN = -0.1;
	static const double VMAX = 0.05;
	static const unsigned int XDIVS = 150;

	set< double >( compt, "Cm", 0.007854e-6 );
	set< double >( compt, "Ra", 7639.44e3 ); // does it matter?
	set< double >( compt, "Rm", 424.4e3 );
	set< double >( compt, "Em", EREST + 0.010613 );
	set< double >( compt, "inject", 0.1e-6 );
	set< double >( chan, "Gbar", 0.94248e-3 );
	set< double >( chan, "Ek", EREST + 0.115 );
	set< double >( kchan, "Gbar", 0.282743e-3 );
	set< double >( kchan, "Ek", EREST - 0.012 );
	set< double >( kchan, "Xpower", 4.0 );

	Id kGateId;
	ret = lookupGet< Id, string >( kchan, "lookupChild", kGateId, "xGate" );
	ASSERT( ret, "Look up kGate");
	ASSERT( !kGateId.zero() && !kGateId.bad(), "Lookup kGate" );

	Element* kGate = kGateId();
	ret = lookupGet< Id, string >( kGate, "lookupChild", temp, "A" );
	ASSERT( ret, "Check gate table" );
	ASSERT( !temp.zero() && !temp.bad(), "kGate_A" );
	Element* kGate_A = temp();
	ret = lookupGet< Id, string >( kGate, "lookupChild", temp, "B" );
	ASSERT( ret, "Check gate table" );
	ASSERT( !temp.zero() && !temp.bad(), "kGate_B" );
	Element* kGate_B = temp();

	assert( set< double >( xGate_A, "xmin", VMIN ) );
	assert( set< double >( xGate_B, "xmin", VMIN ) );
	assert( set< double >( yGate_A, "xmin", VMIN ) );
	assert( set< double >( yGate_B, "xmin", VMIN ) );
	assert( set< double >( kGate_A, "xmin", VMIN ) );
	assert( set< double >( kGate_B, "xmin", VMIN ) );

	assert( set< double >( xGate_A, "xmax", VMAX ) );
	assert( set< double >( xGate_B, "xmax", VMAX ) );
	assert( set< double >( yGate_A, "xmax", VMAX ) );
	assert( set< double >( yGate_B, "xmax", VMAX ) );
	assert( set< double >( kGate_A, "xmax", VMAX ) );
	assert( set< double >( kGate_B, "xmax", VMAX ) );

	assert( set< int >( xGate_A, "xdivs", XDIVS ) );
	assert( set< int >( xGate_B, "xdivs", XDIVS ) );
	assert( set< int >( yGate_A, "xdivs", XDIVS ) );
	assert( set< int >( yGate_B, "xdivs", XDIVS ) );
	assert( set< int >( kGate_A, "xdivs", XDIVS ) );
	assert( set< int >( kGate_B, "xdivs", XDIVS ) );

	assert( set< int >( xGate_A, "mode", 1 ) );
	assert( set< int >( xGate_B, "mode", 1 ) );
	assert( set< int >( yGate_A, "mode", 1 ) );
	assert( set< int >( yGate_B, "mode", 1 ) );
	assert( set< int >( kGate_A, "mode", 1 ) );
	assert( set< int >( kGate_B, "mode", 1 ) );

	double v = VMIN;
	double dv = ( VMAX - VMIN ) / XDIVS;
	const Finfo* table = xGate_A->findFinfo( "table" );
	for (unsigned int i = 0 ; i <= XDIVS; i++ ) {
		lset( xGate_A, table, Na_m_A( v ), i );
		lset( xGate_B, table, Na_m_A( v ) + Na_m_B( v ), i );
		lset( yGate_A, table, Na_h_A( v ), i );
		lset( yGate_B, table, Na_h_A( v ) + Na_h_B( v ), i );
		lset( kGate_A, table, K_n_A( v ), i );
		lset( kGate_B, table, K_n_A( v ) + K_n_B( v ), i );
		v = v + dv;
	}

	assert( set< double >( compt, "initVm", EREST ) );

	pb.dt_ = 1.0e-5;
	pb.currTime_ = 0.0;
	Conn c1( compt, 0 );
	Conn c2( chan, 0 );
	Conn c3( kchan, 0 );

	Compartment::reinitFunc( c1, &pb );
	HHChannel::reinitFunc( c2, &pb );
	HHChannel::reinitFunc( c3, &pb );

	unsigned int sample = 0;
	double delta = 0.0;
	for ( pb.currTime_ = 0.0; pb.currTime_ < 0.01;
			pb.currTime_ += pb.dt_ )
	{
		Compartment::processFunc( c1, &pb );
		HHChannel::processFunc( c2, &pb );
		HHChannel::processFunc( c3, &pb );
		if ( static_cast< int >( pb.currTime_ * 1e5 ) % 10 == 0 ) {
			get< double >( compt, "Vm", v );
			// cout << v << endl;
			v -= EREST + actionPotl[ sample++ ] * 0.001;
			delta += v * v;
		}
	}

	ASSERT( delta < 5e-4, "Action potl unit test\n" );
	
	////////////////////////////////////////////////////////////////
	// Clear it all up
	////////////////////////////////////////////////////////////////
	set( n, "destroy" );
}
#endif 
