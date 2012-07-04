/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "HHGate.h"
#include "ChanBase.h"
#include "HHChannel.h"
#include "../shell/Shell.h"

const double HHChannel::EPSILON = 1.0e-10;
const int HHChannel::INSTANT_X = 1;
const int HHChannel::INSTANT_Y = 2;
const int HHChannel::INSTANT_Z = 4;

const Cinfo* HHChannel::initCinfo()
{
	/////////////////////////////////////////////////////////////////////
	// Shared messages
	/////////////////////////////////////////////////////////////////////
	static DestFinfo process( "process", 
		"Handles process call",
		new ProcOpFunc< HHChannel >( &HHChannel::process ) );
	static DestFinfo reinit( "reinit", 
		"Handles reinit call",
		new ProcOpFunc< HHChannel >( &HHChannel::reinit ) );
	static Finfo* processShared[] =
	{
		&process, &reinit
	};
	static SharedFinfo proc( "proc", 
			"This is a shared message to receive Process message from the"
			"scheduler. The first entry is a MsgDest for the Process "
			"operation. It has a single argument, ProcInfo, which "
			"holds lots of information about current time, thread, dt and"
			"so on.\n The second entry is a MsgDest for the Reinit "
			"operation. It also uses ProcInfo.",
		processShared, sizeof( processShared ) / sizeof( Finfo* )
	);

	/////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
		static ElementValueFinfo< HHChannel, double > Xpower( "Xpower",
			"Power for X gate",
			&HHChannel::setXpower,
			&HHChannel::getXpower
		);
		static ElementValueFinfo< HHChannel, double > Ypower( "Ypower",
			"Power for Y gate",
			&HHChannel::setYpower,
			&HHChannel::getYpower
		);
		static ElementValueFinfo< HHChannel, double > Zpower( "Zpower",
			"Power for Z gate",
			&HHChannel::setZpower,
			&HHChannel::getZpower
		);
		static ValueFinfo< HHChannel, int > instant( "instant",
			"Bitmapped flag: bit 0 = Xgate, bit 1 = Ygate, bit 2 = Zgate"
			"When true, specifies that the lookup table value should be"
			"used directly as the state of the channel, rather than used"
			"as a rate term for numerical integration for the state",
			&HHChannel::setInstant,
			&HHChannel::getInstant
		);
		static ValueFinfo< HHChannel, double > X( "X", 
			"State variable for X gate",
			&HHChannel::setX,
			&HHChannel::getX
		);
		static ValueFinfo< HHChannel, double > Y( "Y",
			"State variable for Y gate",
			&HHChannel::setY,
			&HHChannel::getY
		);
		static ValueFinfo< HHChannel, double > Z( "Z",
			"State variable for Y gate",
			&HHChannel::setZ,
			&HHChannel::getZ
		);
		static ValueFinfo< HHChannel, int > useConcentration( 
			"useConcentration",
			"Flag: when true, use concentration message rather than Vm to"
			"control Z gate",
			&HHChannel::setUseConcentration,
			&HHChannel::getUseConcentration
		);

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	// IkOut SrcFinfo defined above.

///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
		static DestFinfo concen( "concen", 
			"Incoming message from Concen object to specific conc to use"
			"in the Z gate calculations",
			new OpFunc1< HHChannel, double >( &HHChannel::handleConc )
		);
		static DestFinfo createGate( "createGate",
			"Function to create specified gate."
			"Argument: Gate type [X Y Z]",
			new EpFunc1< HHChannel, string >( &HHChannel::createGate )
		);
///////////////////////////////////////////////////////
// FieldElementFinfo definition for HHGates. Note that these are made
// with the deferCreate flag off, so that the HHGates are created 
// right away even if they are empty.
// Assume only a single entry allocated in each gate.
///////////////////////////////////////////////////////
		static FieldElementFinfo< HHChannel, HHGate > gateX( "gateX",
			"Sets up HHGate X for channel",
			HHGate::initCinfo(),
			&HHChannel::getXgate,
			&HHChannel::setNumGates,
			&HHChannel::getNumXgates,
			1
		);
		static FieldElementFinfo< HHChannel, HHGate > gateY( "gateY",
			"Sets up HHGate Y for channel",
			HHGate::initCinfo(),
			&HHChannel::getYgate,
			&HHChannel::setNumGates,
			&HHChannel::getNumYgates,
			1
		);
		static FieldElementFinfo< HHChannel, HHGate > gateZ( "gateZ",
			"Sets up HHGate Z for channel",
			HHGate::initCinfo(),
			&HHChannel::getZgate,
			&HHChannel::setNumGates,
			&HHChannel::getNumZgates,
			1
		);
	
///////////////////////////////////////////////////////
	static Finfo* HHChannelFinfos[] =
	{
		&proc,				// Shared
		&Xpower,			// Value
		&Ypower,			// Value
		&Zpower,			// Value
		&instant,			// Value
		&X,					// Value
		&Y,					// Value
		&Z,					// Value
		&useConcentration,	// Value
		&concen,			// Dest
		&createGate,		// Dest
		&gateX,				// FieldElement
		&gateY,				// FieldElement
		&gateZ				// FieldElement
	};
	
	static string doc[] =
	{
		"Name", "HHChannel",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "HHChannel: Hodgkin-Huxley type voltage-gated Ion channel. Something "
		"like the old tabchannel from GENESIS, but also presents "
		"a similar interface as hhchan from GENESIS. ",
	};

	static Cinfo HHChannelCinfo(
		"HHChannel",
		ChanBase::initCinfo(),
		HHChannelFinfos,
		sizeof( HHChannelFinfos )/sizeof(Finfo *),
		new Dinfo< HHChannel >()
	);

	return &HHChannelCinfo;
}

static const Cinfo* hhChannelCinfo = HHChannel::initCinfo();
//////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////
HHChannel::HHChannel()
			: Xpower_( 0.0 ), Ypower_( 0.0 ), Zpower_( 0.0 ),
			conc_( 0.0 ),
			instant_( 0 ),
			X_( 0.0 ), Y_( 0.0 ), Z_( 0.0 ),
            xInited_( false ), yInited_( false ), zInited_( false ),
			g_( 0.0 ),
			useConcentration_( 0 ),
			xGate_( 0 ),
			yGate_( 0 ),
			zGate_( 0 ),
			myId_()
{
	;
}

HHChannel::~HHChannel()
{
	if ( xGate_ && reinterpret_cast< char* >( this ) == 
		ObjId( xGate_->originalChannelId(), 0 ).data() )
		delete xGate_;
	if ( yGate_ && reinterpret_cast< char* >( this ) == 
		ObjId( yGate_->originalChannelId(), 0 ).data() )
		delete yGate_;
	if ( zGate_ && reinterpret_cast< char* >( this ) == 
		ObjId( zGate_->originalChannelId(), 0 ).data() )
		delete zGate_;
}

bool HHChannel::setGatePower( const Eref& e, const Qinfo* q, double power,
	double *assignee, const string& gateType )
{
	if ( power < 0 ) {
		cout << "Error: HHChannel::set" << gateType << 
			"power: Cannot use negative power: " << power << endl;
		return 0;
	}

	if ( doubleEq( power, *assignee ) )
		return 0;

	if ( doubleEq( *assignee, 0.0 ) && power > 0 ) {
		createGate( e, q, gateType );
	} else if ( doubleEq( power, 0.0 ) ) {
		destroyGate( e, q, gateType );
	}
	
	*assignee = power;
	return 1;
}

/**
 * Assigns the Xpower for this gate. If the gate exists and has
 * only this element for input, then change the gate value.
 * If the gate exists and has multiple parents, then make a new gate.
 * If the gate does not exist, make a new gate
 */
void HHChannel::setXpower( const Eref& e, const Qinfo* q, double Xpower )
{
	if ( setGatePower( e, q, Xpower, &Xpower_, "X" ) )
		takeXpower_ = selectPower( Xpower );
}

void HHChannel::setYpower( const Eref& e, const Qinfo* q, double Ypower )
{
	if ( setGatePower( e, q, Ypower, &Ypower_, "Y" ) )
		takeYpower_ = selectPower( Ypower );
}

void HHChannel::setZpower( const Eref& e, const Qinfo* q, double Zpower )
{
	if ( setGatePower( e, q, Zpower, &Zpower_, "Z" ) ) {
		takeZpower_ = selectPower( Zpower );
		useConcentration_ = 1; // Not sure about this.
	}
}

/**
 * If the gate exists and has only this element for input, then change
 * the gate power.
 * If the gate exists and has multiple parents, then make a new gate,
 * 	set its power.
 * If the gate does not exist, make a new gate, set its power.
 *
 * The function is designed with the idea that if copies of this
 * channel are made, then they all point back to the original HHGate.
 * (Unless they are cross-node copies).
 * It is only if we subsequently alter the HHGate of this channel that
 * we need to make our own variant of the HHGate, or disconnect from
 * an existing one.
 * \todo: May need to convert to handling arrays and Erefs.
 */
// Assuming that the elements are simple elements. Use Eref for 
// general case

bool HHChannel::checkOriginal( Id chanId ) const
{
	bool isOriginal = 1;
	if ( xGate_ ) {
		isOriginal = xGate_->isOriginalChannel( chanId );
	} else if ( yGate_ ) {
		isOriginal = yGate_->isOriginalChannel( chanId );
	} else if ( zGate_ ) {
		isOriginal = zGate_->isOriginalChannel( chanId );
	}
	return isOriginal;
}

void HHChannel::innerCreateGate( const string& gateName, 
	HHGate** gatePtr, Id chanId, Id gateId )
{
	//Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	if ( *gatePtr ) {
		cout << "Warning: HHChannel::createGate: '" << gateName <<
			"' on Element '" << chanId.path() << "' already present\n";
		return;
	}
	*gatePtr = new HHGate( chanId, gateId );
}

void HHChannel::createGate( const Eref& e, const Qinfo* q, 
	string gateType )
{
	if ( !checkOriginal( e.id() ) ) {
		cout << "Warning: HHChannel::createGate: Not allowed from copied channel:\n" << e.id().path() << "\n";
		return;
	}

	if ( gateType == "X" )
		innerCreateGate( "xGate", &xGate_, e.id(), Id(e.id().value() + 1) );
	else if ( gateType == "Y" )
		innerCreateGate( "yGate", &yGate_, e.id(), Id(e.id().value() + 2) );
	else if ( gateType == "Z" )
		innerCreateGate( "zGate", &zGate_, e.id(), Id(e.id().value() + 3) );
	else
		cout << "Warning: HHChannel::createGate: Unknown gate type '" <<
			gateType << "'. Ignored\n";
}

void HHChannel::innerDestroyGate( const string& gateName, 
	HHGate** gatePtr, Id chanId )
{
	if ( *gatePtr == 0 ) {
		cout << "Warning: HHChannel::destroyGate: '" << gateName <<
			"' on Element '" << chanId.path() << "' not present\n";
		return;
	}
	delete (*gatePtr);
	*gatePtr = 0;
}

void HHChannel::destroyGate( const Eref& e, const Qinfo* q,
	string gateType )
{
	if ( !checkOriginal( e.id() ) ) {
		cout << "Warning: HHChannel::destroyGate: Not allowed from copied channel:\n" << e.id().path() << "\n";
		return;
	}
	
	if ( gateType == "X" )
		innerDestroyGate( "xGate", &xGate_, e.id() );
	else if ( gateType == "Y" )
		innerDestroyGate( "yGate", &yGate_, e.id() );
	else if ( gateType == "Z" )
		innerDestroyGate( "zGate", &zGate_, e.id() );
	else
		cout << "Warning: HHChannel::destroyGate: Unknown gate type '" <<
			gateType << "'. Ignored\n";
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
/**
 * Assigns the Xpower for this gate. If the gate exists and has
 * only this element for input, then change the gate value.
 * If the gate exists and has multiple parents, then make a new gate.
 * If the gate does not exist, make a new gate
 */

double HHChannel::getXpower( const Eref& e, const Qinfo* q ) const
{
	return Xpower_;
}

double HHChannel::getYpower( const Eref& e, const Qinfo* q ) const
{
	return Ypower_;
}

double HHChannel::getZpower( const Eref& e, const Qinfo* q ) const
{
	return Zpower_;
}

void HHChannel::setInstant( int instant )
{
	instant_ = instant;
}
int HHChannel::getInstant() const
{
	return instant_;
}

void HHChannel::setX( double X )
{
        X_ = X;
        xInited_ = true;        
}
double HHChannel::getX() const
{
	return X_;
}

void HHChannel::setY( double Y )
{
        Y_ = Y;
        yInited_ = true;        
}
double HHChannel::getY() const
{
	return Y_;
}

void HHChannel::setZ( double Z )
{
        Z_ = Z;
        zInited_ = true;        
}
double HHChannel::getZ() const
{
	return Z_;
}

void HHChannel::setUseConcentration( int value )
{
	useConcentration_ = value;
}

int HHChannel::getUseConcentration() const
{
	return useConcentration_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

/**
 * Returns the state variable for the new timestep based on
 * the internal variables A_ and B_ which were passed in from the gate.
 */
double HHChannel::integrate( double state, double dt, double A, double B )
{
	if ( B > EPSILON ) {
		double x = exp( -B * dt );
		return state * x + ( A / B ) * ( 1 - x );
	}
	return state + A * dt ;
}

void HHChannel::process( const Eref& e, ProcPtr info )
{
	g_ += ChanBase::getGbar();
	double A = 0;
	double B = 0;
	if ( Xpower_ > 0 ) {
		xGate_->lookupBoth( Vm_, &A, &B );
		if ( instant_ & INSTANT_X )
			X_ = A/B;
		else 
			X_ = integrate( X_, info->dt, A, B );
		g_ *= takeXpower_( X_, Xpower_ );
	}

	if ( Ypower_ > 0 ) {
		yGate_->lookupBoth( Vm_, &A, &B );
		if ( instant_ & INSTANT_Y )
			Y_ = A/B;
		else 
			Y_ = integrate( Y_, info->dt, A, B );

		g_ *= takeYpower_( Y_, Ypower_ );
	}

	if ( Zpower_ > 0 ) {
		if ( useConcentration_ )
			zGate_->lookupBoth( conc_, &A, &B );
		else
			zGate_->lookupBoth( Vm_, &A, &B );
		if ( instant_ & INSTANT_Z )
			Z_ = A/B;
		else 
			Z_ = integrate( Z_, info->dt, A, B );

		g_ *= takeZpower_( Z_, Zpower_ );
	}

	ChanBase::setGk( g_ );
	ChanBase::updateIk();
	// Gk_ = g_;
	// Ik_ = ( Ek_ - Vm_ ) * g_;

	// Send out the relevant channel messages.
	ChanBase::process( e, info );
	/*
	channelOut.send( e, info, Gk_, Ek_ );
	
	// This is used if the channel connects up to a conc pool and
	// handles influx of ions giving rise to a concentration change.
	IkOut.send( e, info, Ik_ );
	
	// Needed by GHK-type objects
	permeability.send( e, info, Gk_ );
	*/
	
	g_ = 0.0;
    
}

/**
 * Here we get the steady-state values for the gate (the 'instant'
 * calculation) as A_/B_.
 */
void HHChannel::reinit( const Eref& er, ProcPtr info )
{
	g_ = ChanBase::getGbar();
	Element* e = er.element();

	double A = 0.0;
	double B = 0.0;
	if ( Xpower_ > 0 ) {
		assert( xGate_ );
		xGate_->lookupBoth( Vm_, &A, &B );
		if ( B < EPSILON ) {
			cout << "Warning: B_ value for " << e->getName() <<
					" is ~0. Check X table\n";
			return;
		}
                if (!xInited_)
                    X_ = A/B;
		g_ *= takeXpower_( X_, Xpower_ );
	}

	if ( Ypower_ > 0 ) {
		assert( yGate_ );
		yGate_->lookupBoth( Vm_, &A, &B );
		if ( B < EPSILON ) {
			cout << "Warning: B value for " << e->getName() <<
					" is ~0. Check Y table\n";
			return;
		}
                if (!yInited_)
                    Y_ = A/B;
		g_ *= takeYpower_( Y_, Ypower_ );
	}

	if ( Zpower_ > 0 ) {
		assert( zGate_ );
		if ( useConcentration_ )
			zGate_->lookupBoth( conc_, &A, &B );
		else
			zGate_->lookupBoth( Vm_, &A, &B );
		if ( B < EPSILON ) {
			cout << "Warning: B value for " << e->getName() <<
					" is ~0. Check Z table\n";
			return;
		}
                if (!zInited_)
                    Z_ = A/B;
		g_ *= takeZpower_( Z_, Zpower_ );
	}

	ChanBase::setGk( g_ );
	ChanBase::updateIk();
	// Gk_ = g_;
	// Ik_ = ( Ek_ - Vm_ ) * g_;

	// Send out the relevant channel messages.
	// Same for reinit as for process.
	ChanBase::reinit( er, info );

	/*
	channelOut.send( er, info, Gk_, Ek_ );
	// Needed by GHK-type objects
	permeability.send( er, info, Gk_ );
	*/
	
	g_ = 0.0;
}

void HHChannel::handleConc( double conc )
{
	conc_ = conc;
}


///////////////////////////////////////////////////
// HHGate functions
///////////////////////////////////////////////////


HHGate* HHChannel::getXgate( unsigned int i )
{
	return xGate_;
}

HHGate* HHChannel::getYgate( unsigned int i )
{
	return yGate_;
}

HHGate* HHChannel::getZgate( unsigned int i )
{
	return zGate_;
}

void HHChannel::setNumGates( unsigned int num ) 
{ ; }

unsigned int  HHChannel::getNumXgates() const
{
	return ( xGate_ != 0 );
}

unsigned int  HHChannel::getNumYgates() const
{
	return ( yGate_ != 0 );
}

unsigned int  HHChannel::getNumZgates() const
{
	return ( zGate_ != 0 );
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
