/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "../randnum/randnum.h"
#include "Compartment.h"

using namespace moose;
const double Compartment::EPSILON = 1.0e-15;

static SrcFinfo1< double >* VmOut() {
	static SrcFinfo1< double > VmOut( "VmOut", 
		"Sends out Vm value of compartment on each timestep" );
	return &VmOut;
}

// Here we send out Vm, but to a different subset of targets. So we
// have to define a new SrcFinfo even though the contents of the msg
// are still only Vm.

static SrcFinfo1< double >* axialOut() {
	static SrcFinfo1< double > axialOut( "axialOut", 
		"Sends out Vm value of compartment to adjacent compartments,"
		"on each timestep" );
	return &axialOut;
}

static SrcFinfo2< double, double >* raxialOut()
{
	static SrcFinfo2< double, double > raxialOut( "raxialOut", 
		"Sends out Raxial information on each timestep, "
		"fields are Ra and Vm" );
	return &raxialOut;
}

/**
 * The initCinfo() function sets up the Compartment class.
 * This function uses the common trick of having an internal
 * static value which is created the first time the function is called.
 * There are several static arrays set up here. The ones which
 * use SharedFinfos are for shared messages where multiple kinds
 * of information go along the same connection.
 */
const Cinfo* Compartment::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Shared messages
	///////////////////////////////////////////////////////////////////
	static DestFinfo process( "process", 
		"Handles 'process' call",
		new ProcOpFunc< Compartment >( &Compartment::process ) );

	static DestFinfo reinit( "reinit", 
		"Handles 'reinit' call",
		new ProcOpFunc< Compartment >( &Compartment::reinit ) );
	
	static Finfo* processShared[] =
	{
		&process, &reinit
	};

	static SharedFinfo proc( "proc",
		"This is a shared message to receive Process messages "
		"from the scheduler objects. The Process should be called "
		"_second_ in each clock tick, after the Init message."
		"The first entry in the shared msg is a MsgDest "
		"for the Process operation. It has a single argument, "
		"ProcInfo, which holds lots of information about current "
		"time, thread, dt and so on. The second entry is a MsgDest "
		"for the Reinit operation. It also uses ProcInfo. ",
		processShared, sizeof( processShared ) / sizeof( Finfo* )
	);
	///////////////////////////////////////////////////////////////////
	
	static DestFinfo initProc( "initProc", 
		"Handles Process call for the 'init' phase of the Compartment "
		"calculations. These occur as a separate Tick cycle from the "
		"regular proc cycle, and should be called before the proc msg.",
		new ProcOpFunc< Compartment >( &Compartment::initProc ) );
	static DestFinfo initReinit( "initReinit", 
		"Handles Reinit call for the 'init' phase of the Compartment "
		"calculations.",
		new ProcOpFunc< Compartment >( &Compartment::initReinit ) );
	static Finfo* initShared[] =
	{
		&initProc, &initReinit
	};

	static SharedFinfo init( "init", 
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
			"operation. It also uses ProcInfo. ",
		initShared, sizeof( initShared ) / sizeof( Finfo* )
	);

	///////////////////////////////////////////////////////////////////

	static DestFinfo handleChannel( "handleChannel", 
		"Handles conductance and Reversal potential arguments from Channel",
		new OpFunc2< Compartment, double, double >( &Compartment::handleChannel ) );
	// VmOut is declared above as it needs to be in scope for later funcs.

	static Finfo* channelShared[] =
	{
		&handleChannel, VmOut()
	};
	static SharedFinfo channel( "channel", 
			"This is a shared message from a compartment to channels. "
			"The first entry is a MsgDest for the info coming from "
			"the channel. It expects Gk and Ek from the channel "
			"as args. The second entry is a MsgSrc sending Vm ",
		channelShared, sizeof( channelShared ) / sizeof( Finfo* )
	);
	///////////////////////////////////////////////////////////////////
	// axialOut declared above as it is needed in file scope
	static DestFinfo handleRaxial( "handleRaxial", 
		"Handles Raxial info: arguments are Ra and Vm.",
		new OpFunc2< Compartment, double, double >( 
			&Compartment::handleRaxial )
	);

	static Finfo* axialShared[] =
	{
		axialOut(), &handleRaxial
	};
	static SharedFinfo axial( "axial", 
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
			"message is named after the source type. ",
		axialShared, sizeof( axialShared ) / sizeof( Finfo* )
	);

	///////////////////////////////////////////////////////////////////
	static DestFinfo handleAxial( "handleAxial", 
		"Handles Axial information. Argument is just Vm.",
		new OpFunc1< Compartment, double >( &Compartment::handleAxial ) );
	// rxialOut declared above as it is needed in file scope
	static Finfo* raxialShared[] =
	{
		&handleAxial, raxialOut()
	};
	static SharedFinfo raxial( "raxial", 
			"This is a raxial shared message between asymmetric "
			"compartments. The first entry is a MsgDest for the info "
			"coming from the other compt. It expects Vm from the "
			"other compt as an arg. The second is a MsgSrc sending "
			"Ra and Vm to the raxialFunc of the target compartment. ",
			raxialShared, sizeof( raxialShared ) / sizeof( Finfo* )
	);
	///////////////////////////////////////////////////////////////////
	// Value Finfos.
	///////////////////////////////////////////////////////////////////

		static ValueFinfo< Compartment, double > Vm( "Vm", 
			"membrane potential",
			&Compartment::setVm,
			&Compartment::getVm
		);
		static ValueFinfo< Compartment, double > Cm( "Cm", 
			"Membrane capacitance",
			 &Compartment::setCm,
			&Compartment::getCm
		);
		static ValueFinfo< Compartment, double > Em( "Em", 
			"Resting membrane potential",
			 &Compartment::setEm,
			&Compartment::getEm
		);
		static ReadOnlyValueFinfo< Compartment, double > Im( "Im", 
			"Current going through membrane",
			&Compartment::getIm
		);
		static ValueFinfo< Compartment, double > inject( "inject", 
			"Current injection to deliver into compartment",
			&Compartment::setInject,
			&Compartment::getInject
		);
		static ValueFinfo< Compartment, double > initVm( "initVm", 
			"Initial value for membrane potential",
			&Compartment::setInitVm,
			&Compartment::getInitVm
		);
		static ValueFinfo< Compartment, double > Rm( "Rm", 
			"Membrane resistance",
			&Compartment::setRm,
			&Compartment::getRm
		);
		static ValueFinfo< Compartment, double > Ra( "Ra", 
			"Axial resistance of compartment",
			&Compartment::setRa,
			&Compartment::getRa
		);
		static ValueFinfo< Compartment, double > diameter( "diameter", 
			"Diameter of compartment",
			&Compartment::setDiameter,
			&Compartment::getDiameter
		);
		static ValueFinfo< Compartment, double > length( "length", 
			"Length of compartment",
			&Compartment::setLength,
			&Compartment::getLength
		);
		static ValueFinfo< Compartment, double > x0( "x0", 
			"X coordinate of start of compartment",
			&Compartment::setX0,
			&Compartment::getX0
		);
		static ValueFinfo< Compartment, double > y0( "y0", 
			"Y coordinate of start of compartment",
			&Compartment::setY0,
			&Compartment::getY0
		);
		static ValueFinfo< Compartment, double > z0( "z0", 
			"Z coordinate of start of compartment",
			&Compartment::setZ0,
			&Compartment::getZ0
		);
		static ValueFinfo< Compartment, double > x( "x",
			"x coordinate of end of compartment",
			&Compartment::setX,
			&Compartment::getX
		);
		static ValueFinfo< Compartment, double > y( "y",
			"y coordinate of end of compartment",
			&Compartment::setY,
			&Compartment::getY
		);
		static ValueFinfo< Compartment, double > z( "z", 
			"z coordinate of end of compartment",
			&Compartment::setZ,
			&Compartment::getZ
		);
	
	//////////////////////////////////////////////////////////////////
	// DestFinfo definitions
	//////////////////////////////////////////////////////////////////
		static DestFinfo injectMsg( "injectMsg", 
			"The injectMsg corresponds to the INJECT message in the "
			"GENESIS compartment. Unlike the 'inject' field, any value "
			"assigned by handleInject applies only for a single timestep."
			"So it needs to be updated every dt for a steady (or varying)"
			"injection current",
			new OpFunc1< Compartment,  double >( &Compartment::injectMsg )
		);
		
		static DestFinfo randInject( "randInject",
			"Sends a random injection current to the compartment. Must be"
			"updated each timestep."
			"Arguments to randInject are probability and current.",
			new OpFunc2< Compartment, double, double > (
				&Compartment::randInject ) );

		static DestFinfo cable( "cable", 
			"Message for organizing compartments into groups, called"
			"cables. Doesn't do anything.",
			new OpFunc0< Compartment >( &Compartment::cable )
		);
	///////////////////////////////////////////////////////////////////
	static Finfo* compartmentFinfos[] = 
	{
		&Vm,				// Value
		&Cm,				// Value
		&Em,				// Value
		&Im,				// ReadOnlyValue
		&inject,			// Value
		&initVm,			// Value
		&Rm,				// Value
		&Ra,				// Value
		&diameter,			// Value
		&length,			// Value
		&x0,				// Value
		&y0,				// Value
		&z0,				// Value
		&x,					// Value
		&y,					// Value
		&z,					// Value
		&injectMsg,			// DestFinfo
		&randInject,		// DestFinfo
		&injectMsg,			// DestFinfo
		&cable,				// DestFinfo
		&proc,				// SharedFinfo
		&init,				// SharedFinfo
		&channel,			// SharedFinfo
		&axial,				// SharedFinfo
		&raxial				// SharedFinfo
	};

	static string doc[] =
	{
		"Name", "Compartment",
		"Author", "Upi Bhalla",
		"Description", "Compartment object, for branching neuron models.",
	};	
	static Cinfo compartmentCinfo(
				"Compartment",
				Neutral::initCinfo(),
				compartmentFinfos,
				sizeof( compartmentFinfos ) / sizeof( Finfo* ),
				new Dinfo< Compartment >()
	);

	return &compartmentCinfo;
}

static const Cinfo* compartmentCinfo = Compartment::initCinfo();

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
        lastIm_ = 0.0;
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

Compartment::~Compartment()
{
	;
}

bool Compartment::rangeWarning( const string& field, double value )
{
	if ( value < Compartment::EPSILON ) {
		cout << "Warning: Ignored attempt to set " << field <<
				" of compartment " <<
				// c->target().e->name() << 
				" to less than " << EPSILON << endl;
		return 1;
	}
	return 0;
}

// Value Field access function definitions.
void Compartment::setVm( double Vm )
{
	Vm_ = Vm;
}

double Compartment::getVm() const
{
	return Vm_;
}

void Compartment::setEm( double Em )
{
	Em_ = Em;
}

double Compartment::getEm() const
{
	return Em_;
}

void Compartment::setCm( double Cm )
{
	if ( rangeWarning( "Cm", Cm ) ) return;
	Cm_ = Cm;
}

double Compartment::getCm() const
{
	return Cm_;
}

void Compartment::setRm( double Rm )
{
	if ( rangeWarning( "Rm", Rm ) ) return;
	Rm_ = Rm;
	invRm_ = 1.0/Rm;
}

double Compartment::getRm() const
{
	return Rm_;
}

void Compartment::setRa( double Ra )
{
	if ( rangeWarning( "Ra", Ra ) ) return;
	Ra_ = Ra;
}

double Compartment::getRa() const
{
	return Ra_;
}

void Compartment::setIm( double Im )
{
	Im_ = Im;
}

double Compartment::getIm() const
{
	return lastIm_;
}

void Compartment::setInject( double Inject )
{
	Inject_ = Inject;
}

double Compartment::getInject() const
{
	return Inject_;
}

void Compartment::setInitVm( double initVm )
{
	initVm_ = initVm;
}

double Compartment::getInitVm() const
{
	return initVm_;
}

void Compartment::setDiameter( double value )
{
	diameter_ = value;
}

double Compartment::getDiameter() const
{
	return diameter_;
}

void Compartment::setLength( double value )
{
	length_ = value;
}

double Compartment::getLength() const
{
	return length_;
}

void Compartment::setX0( double value )
{
	x0_ = value;
}

double Compartment::getX0() const
{
	return x0_;
}

void Compartment::setY0( double value )
{
	y0_ = value;
}

double Compartment::getY0() const
{
	return y0_;
}

void Compartment::setZ0( double value )
{
	z0_ = value;
}

double Compartment::getZ0() const
{
	return z0_;
}

void Compartment::setX( double value )
{
	x_ = value;
}

double Compartment::getX() const
{
	return x_;
}

void Compartment::setY( double value )
{
	y_ = value;
}

double Compartment::getY() const
{
	return y_;
}

void Compartment::setZ( double value )
{
	z_ = value;
}

double Compartment::getZ() const
{
	return z_;
}

//////////////////////////////////////////////////////////////////
// Compartment::Dest function definitions.
//////////////////////////////////////////////////////////////////

void Compartment::process( const Eref& e, ProcPtr p )
{
	A_ += Inject_ + sumInject_ + Em_ * invRm_; 
	if ( B_ > EPSILON ) {
		double x = exp( -B_ * p->dt / Cm_ );
		Vm_ = Vm_ * x + ( A_ / B_ )  * ( 1.0 - x );
	} else {
		Vm_ += ( A_ - Vm_ * B_ ) * p->dt / Cm_;
	}
	A_ = 0.0;
	B_ = invRm_;
        lastIm_ = Im_;
	Im_ = 0.0;
	sumInject_ = 0.0;
	// Send out Vm to channels, SpikeGens, etc.
	VmOut()->send( e, p->threadIndexInGroup, Vm_ );

	// The axial/raxial messages go out in the 'init' phase.
}

void Compartment::reinit(  const Eref& e, ProcPtr p )
{
	this->innerReinit( e, p );
}

void Compartment::innerReinit(  const Eref& e, ProcPtr p )
{
	Vm_ = initVm_;
	A_ = 0.0;
	B_ = invRm_;
	Im_ = 0.0;
        lastIm_ = 0.0;
	sumInject_ = 0.0;
	dt_ = p->dt;
	
	// Send out the resting Vm to channels, SpikeGens, etc.
	VmOut()->send( e, p->threadIndexInGroup, Vm_ );
}

void Compartment::initProc( const Eref& e, ProcPtr p )
{
	// Separate variants for regular and SymCompartment
	this->innerInitProc( e, p ); 
}

void Compartment::innerInitProc( const Eref& e, ProcPtr p )
{
	// Send out the axial messages
	axialOut()->send( e, p->threadIndexInGroup, Vm_ );

	// Send out the raxial messages
	raxialOut()->send( e, p->threadIndexInGroup, Ra_, Vm_ );
}

void Compartment::initReinit( const Eref& e, ProcPtr p )
{
	this->innerInitReinit( e, p );
}

void Compartment::innerInitReinit( const Eref& e, ProcPtr p )
{
	; // Nothing happens here
}

void Compartment::handleChannel( double Gk, double Ek)
{
	A_ += Gk * Ek;
	B_ += Gk;
}

void Compartment::handleRaxial( double Ra, double Vm)
{
	A_ += Vm / Ra;
	B_ += 1.0 / Ra;
	Im_ += ( Vm - Vm_ ) / Ra;
}

void Compartment::handleAxial( double Vm)
{
	A_ += Vm / Ra_;
	B_ += 1.0 / Ra_;
	Im_ += ( Vm - Vm_ ) / Ra_;
}

void Compartment::injectMsg( double current)
{
	sumInject_ += current;
	Im_ += current;
}

void Compartment::randInject( double prob, double current)
{
	if ( mtrand() < prob * dt_ ) {
		sumInject_ += current;
		Im_ += current;
	}
}

void Compartment::cable()
{
	;
}

/////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS

void testCompartment()
{
	const Cinfo* comptCinfo = Compartment::initCinfo();
	Id comptId = Id::nextId();
	vector< DimInfo > dims;
	Element* n = new Element( comptId, comptCinfo, "compt", dims, 1, true );
	assert( n != 0 );
	Eref compter( n, 0 );
	Compartment* c = reinterpret_cast< Compartment* >( compter.data() );
	ProcInfo p;
	p.dt = 0.002;
	c->setInject( 1.0 );
	c->setRm( 1.0 );
	c->setRa( 0.0025 );
	c->setCm( 1.0 );
	c->setEm( 0.0 );
	c->setVm( 0.0 );

	// First, test charging curve for a single compartment
	// We want our charging curve to be a nice simple exponential
	// Vm = 1.0 - 1.0 * exp( - t / 1.0 );
	double delta = 0.0;
	double Vm = 0.0;
	double tau = 1.0;
	double Vmax = 1.0;
	for ( p.currTime = 0.0; p.currTime < 2.0; p.currTime += p.dt ) 
	{
		Vm = c->getVm();
		double x = Vmax - Vmax * exp( -p.currTime / tau );
		delta += ( Vm - x ) * ( Vm - x );
		c->process( compter, &p );
	}
	assert( delta < 1.0e-6 );

	comptId.destroy();
	cout << "." << flush;
}

// Comment out this define if it takes too long (about 5 seconds on
// a modest machine, but could be much longer with valgrind)
#define DO_SPATIAL_TESTS
/**
 * Function to test the spatial spread of charge.
 * We make the cable long enough to get another nice exponential.
 * Vm = Vm0 * exp( -x/lambda)
 * lambda = sqrt( Rm/Ra ) where these are the actual values, not
 * the values per unit length.
 * So here lambda = 20, so that each compt is lambda/20
 */
#include "../shell/Shell.h"
void testCompartmentProcess()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	unsigned int size = 100;
	vector< int > dims( 1, size );

	double Rm = 1.0;
	double Ra = 0.01;
	double Cm = 1.0;
	double dt = 0.01;
	double runtime = 10;
	double lambda = sqrt( Rm / Ra );

	Id cid = shell->doCreate( "Compartment", Id(), "compt", dims );
	assert( cid != Id() );
	assert( cid()->dataHandler()->totalEntries() == size );

	bool ret = Field< double >::setRepeat( cid, "initVm", 0.0 );
	assert( ret );
	Field< double >::setRepeat( cid, "inject", 0 );
	// Only apply current injection in first compartment
	Field< double >::set( ObjId( cid, 0 ), "inject", 1.0 ); 
	Field< double >::setRepeat( cid, "Rm", Rm );
	Field< double >::setRepeat( cid, "Ra", Ra );
	Field< double >::setRepeat( cid, "Cm", Cm );
	Field< double >::setRepeat( cid, "Em", 0 );
	Field< double >::setRepeat( cid, "Vm", 0 );

	// The diagonal message has a default stride of 1, so it connects
	// successive compartments.
	// Note that the src and dest elements here are identical, so we cannot
	// use a shared message. The messaging system will get confused about
	// direction to send data. So we split up the shared message that we
	// might have used, below, into two individual messages.
	// MsgId mid = shell->doAddMsg( "Diagonal", ObjId( cid ), "raxial", ObjId( cid ), "axial" );
	MsgId mid = shell->doAddMsg( "Diagonal", ObjId( cid ), "axialOut", ObjId( cid ), "handleAxial" );
	assert( mid != Msg::bad);
	// mid = shell->doAddMsg( "Diagonal", ObjId( cid ), "handleRaxial", ObjId( cid ), "raxialOut" );
	mid = shell->doAddMsg( "Diagonal", ObjId( cid ), "raxialOut", ObjId( cid ), "handleRaxial" );
	assert( mid != Msg::bad );
	ObjId managerId = Msg::getMsg( mid )->manager().objId();
	// Make the raxial data go from high to lower index compartments.
	Field< int >::set( managerId, "stride", -1 );

#ifdef DO_SPATIAL_TESTS
	shell->doSetClock( 0, dt );
	shell->doSetClock( 1, dt );
	shell->doUseClock( "/compt", "init", 0 );
	shell->doUseClock( "/compt", "process", 1 );

	shell->doReinit();
	shell->doStart( runtime );

	double Vmax = Field< double >::get( ObjId( cid, 0 ), "Vm" );

	double delta = 0.0;
	// We measure only the first 50 compartments as later we 
	// run into end effects because it is not an infinite cable
	for ( unsigned int i = 0; i < 50; i++ ) {
		double Vm = Field< double >::get( ObjId( cid, i ), "Vm" );
		double x = Vmax * exp( - static_cast< double >( i ) / lambda );
		delta += ( Vm - x ) * ( Vm - x );
	 	// cout << i << " (x, Vm) = ( " << x << ", " << Vm << " )\n";
	}
	assert( delta < 1.0e-5 );
#endif // DO_SPATIAL_TESTS
	shell->doDelete( cid );
	cout << "." << flush;
}
#endif // DO_UNIT_TESTS
