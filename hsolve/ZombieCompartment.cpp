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
//~ #include "../randnum/randnum.h"
#include "../biophysics/Compartment.h"
#include "HinesMatrix.h"
#include "HSolveStruct.h"
#include "HSolvePassive.h"
#include "RateLookup.h"
#include "HSolveActive.h"
#include "HSolve.h"
#include "ZombieCompartment.h"

using namespace moose;
const double ZombieCompartment::EPSILON = 1.0e-15;

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
const Cinfo* ZombieCompartment::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Shared messages
	///////////////////////////////////////////////////////////////////
	static DestFinfo process( "process", 
		"Handles 'process' call",
		new ProcOpFunc< ZombieCompartment >( &ZombieCompartment::process ) );
	
	static DestFinfo reinit( "reinit", 
		"Handles 'reinit' call",
		new ProcOpFunc< ZombieCompartment >( &ZombieCompartment::reinit ) );
	
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
		new ProcOpFunc< ZombieCompartment >( &ZombieCompartment::initProc ) );
	static DestFinfo initReinit( "initReinit", 
		"Handles Reinit call for the 'init' phase of the Compartment "
		"calculations.",
		new ProcOpFunc< ZombieCompartment >( &ZombieCompartment::initReinit ) );
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
		new EpFunc2< ZombieCompartment, double, double >( &ZombieCompartment::handleChannel ) );
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
		new OpFunc2< ZombieCompartment, double, double >( 
			&ZombieCompartment::handleRaxial )
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
		new OpFunc1< ZombieCompartment, double >( &ZombieCompartment::handleAxial ) );
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
	
		static ElementValueFinfo< ZombieCompartment, double > Vm( "Vm", 
			"membrane potential",
			&ZombieCompartment::setVm,
			&ZombieCompartment::getVm
		);
		static ElementValueFinfo< ZombieCompartment, double > Cm( "Cm", 
			"Membrane capacitance",
			&ZombieCompartment::setCm,
			&ZombieCompartment::getCm
		);
		static ElementValueFinfo< ZombieCompartment, double > Em( "Em", 
			"Resting membrane potential",
			 &ZombieCompartment::setEm,
			&ZombieCompartment::getEm
		);
		static ReadOnlyElementValueFinfo< ZombieCompartment, double > Im( "Im", 
			"Current going through membrane",
			&ZombieCompartment::getIm
		);
		static ElementValueFinfo< ZombieCompartment, double > inject( "inject", 
			"Current injection to deliver into compartment",
			&ZombieCompartment::setInject,
			&ZombieCompartment::getInject
		);
		static ElementValueFinfo< ZombieCompartment, double > initVm( "initVm", 
			"Initial value for membrane potential",
			&ZombieCompartment::setInitVm,
			&ZombieCompartment::getInitVm
		);
		static ElementValueFinfo< ZombieCompartment, double > Rm( "Rm", 
			"Membrane resistance",
			&ZombieCompartment::setRm,
			&ZombieCompartment::getRm
		);
		static ElementValueFinfo< ZombieCompartment, double > Ra( "Ra", 
			"Axial resistance of compartment",
			&ZombieCompartment::setRa,
			&ZombieCompartment::getRa
		);
		static ValueFinfo< ZombieCompartment, double > diameter( "diameter", 
			"Diameter of compartment",
			&ZombieCompartment::setDiameter,
			&ZombieCompartment::getDiameter
		);
		static ValueFinfo< ZombieCompartment, double > length( "length", 
			"Length of compartment",
			&ZombieCompartment::setLength,
			&ZombieCompartment::getLength
		);
		static ValueFinfo< ZombieCompartment, double > x0( "x0", 
			"X coordinate of start of compartment",
			&ZombieCompartment::setX0,
			&ZombieCompartment::getX0
		);
		static ValueFinfo< ZombieCompartment, double > y0( "y0", 
			"Y coordinate of start of compartment",
			&ZombieCompartment::setY0,
			&ZombieCompartment::getY0
		);
		static ValueFinfo< ZombieCompartment, double > z0( "z0", 
			"Z coordinate of start of compartment",
			&ZombieCompartment::setZ0,
			&ZombieCompartment::getZ0
		);
		static ValueFinfo< ZombieCompartment, double > x( "x",
			"x coordinate of end of compartment",
			&ZombieCompartment::setX,
			&ZombieCompartment::getX
		);
		static ValueFinfo< ZombieCompartment, double > y( "y",
			"y coordinate of end of compartment",
			&ZombieCompartment::setY,
			&ZombieCompartment::getY
		);
		static ValueFinfo< ZombieCompartment, double > z( "z", 
			"z coordinate of end of compartment",
			&ZombieCompartment::setZ,
			&ZombieCompartment::getZ
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
			new OpFunc1< ZombieCompartment, double >( &ZombieCompartment::injectMsg )
		);
		
		static DestFinfo randInject( "randInject",
			"Sends a random injection current to the compartment. Must be"
			"updated each timestep."
			"Arguments to randInject are probability and current.",
			new OpFunc2< ZombieCompartment, double, double > (
				&ZombieCompartment::randInject ) );
		
		static DestFinfo cable( "cable", 
			"Message for organizing compartments into groups, called"
			"cables. Doesn't do anything.",
			new OpFunc0< ZombieCompartment >( &ZombieCompartment::cable )
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
		"Name", "ZombieCompartment",
		"Author", "Upi Bhalla",
		"Description", "Compartment object, for branching neuron models.",
	};
	
	static Cinfo zombieCompartmentCinfo(
				"ZombieCompartment",
				Neutral::initCinfo(),
				compartmentFinfos,
				sizeof( compartmentFinfos ) / sizeof( Finfo* ),
				new Dinfo< ZombieCompartment >()
	);
	
	return &zombieCompartmentCinfo;
}

static const Cinfo* zombieCompartmentCinfo = ZombieCompartment::initCinfo();

//////////////////////////////////////////////////////////////////
// Here we put the Compartment class functions.
//////////////////////////////////////////////////////////////////

ZombieCompartment::ZombieCompartment()
{
	hsolve_   = NULL;
	
	diameter_ = 0.0;
	length_   = 0.0;
	x_        = 0.0;
	y_        = 0.0;
	z_        = 0.0;
	x0_       = 0.0;
	y0_       = 0.0;
	z0_       = 0.0;
}

ZombieCompartment::~ZombieCompartment()
{
	;
}

void ZombieCompartment::copyFields( Compartment* c )
{
	diameter_ = c->getDiameter();
	length_   = c->getLength();
	x0_       = c->getX0();
	y0_       = c->getY0();
	z0_       = c->getZ0();
	x_        = c->getX();
	y_        = c->getY();
	z_        = c->getZ();
}

bool ZombieCompartment::rangeWarning( const string& field, double value )
{
	if ( value < ZombieCompartment::EPSILON ) {
		cout << "Warning: Ignored attempt to set " << field <<
				" of compartment " <<
				// c->target().e->name() << 
				" to less than " << EPSILON << endl;
		return 1;
	}
	return 0;
}

// Value Field access function definitions.
void ZombieCompartment::setVm( const Eref& e, const Qinfo* q, double Vm )
{
	hsolve_->setVm( e.id(), Vm );
}

double ZombieCompartment::getVm( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getVm( e.id() );
}

void ZombieCompartment::setEm( const Eref& e, const Qinfo* q, double Em )
{
	hsolve_->setEm( e.id(), Em );
}

double ZombieCompartment::getEm( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getEm( e.id() );
}

void ZombieCompartment::setCm( const Eref& e, const Qinfo* q, double Cm )
{
	if ( rangeWarning( "Cm", Cm ) ) return;
	hsolve_->setCm( e.id(), Cm );
}

double ZombieCompartment::getCm( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getCm( e.id() );
}

void ZombieCompartment::setRm( const Eref& e, const Qinfo* q, double Rm )
{
	if ( rangeWarning( "Rm", Rm ) ) return;
	hsolve_->setRm( e.id(), Rm );
}

double ZombieCompartment::getRm( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getRm( e.id() );
}

void ZombieCompartment::setRa( const Eref& e, const Qinfo* q, double Ra )
{
	if ( rangeWarning( "Ra", Ra ) ) return;
	hsolve_->setRa( e.id(), Ra );
}

double ZombieCompartment::getRa( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getRa( e.id() );
}

//~ void ZombieCompartment::setIm( const Eref& e, const Qinfo* q, double Im )
//~ {
	//~ Im_ = Im;
//~ }

double ZombieCompartment::getIm( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getIm( e.id() );
}

void ZombieCompartment::setInject( const Eref& e, const Qinfo* q, double Inject )
{
	hsolve_->setInject( e.id(), Inject );
}

double ZombieCompartment::getInject( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getInject( e.id() );
}

void ZombieCompartment::setInitVm( const Eref& e, const Qinfo* q, double initVm )
{
	hsolve_->setInitVm( e.id(), initVm );
}

double ZombieCompartment::getInitVm( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getInitVm( e.id() );
}

void ZombieCompartment::setDiameter( double value )
{
	diameter_ = value;
}

double ZombieCompartment::getDiameter() const
{
	return diameter_;
}

void ZombieCompartment::setLength( double value )
{
	length_ = value;
}

double ZombieCompartment::getLength() const
{
	return length_;
}

void ZombieCompartment::setX0( double value )
{
	x0_ = value;
}

double ZombieCompartment::getX0() const
{
	return x0_;
}

void ZombieCompartment::setY0( double value )
{
	y0_ = value;
}

double ZombieCompartment::getY0() const
{
	return y0_;
}

void ZombieCompartment::setZ0( double value )
{
	z0_ = value;
}

double ZombieCompartment::getZ0() const
{
	return z0_;
}

void ZombieCompartment::setX( double value )
{
	x_ = value;
}

double ZombieCompartment::getX() const
{
	return x_;
}

void ZombieCompartment::setY( double value )
{
	y_ = value;
}

double ZombieCompartment::getY() const
{
	return y_;
}

void ZombieCompartment::setZ( double value )
{
	z_ = value;
}

double ZombieCompartment::getZ() const
{
	return z_;
}

/*
void ZombieCompartment::setDiameter( const Eref& e, const Qinfo* q, double value )
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	c->setDiameter( value );
}

double ZombieCompartment::getDiameter( const Eref& e, const Qinfo* q ) const
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	return c->getDiameter();
}

void ZombieCompartment::setLength( const Eref& e, const Qinfo* q, double value )
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	c->setLength( value );
}

double ZombieCompartment::getLength( const Eref& e, const Qinfo* q ) const
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	return c->getLength();
}

void ZombieCompartment::setX0( const Eref& e, const Qinfo* q, double value )
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	c->setX0( value );
}

double ZombieCompartment::getX0( const Eref& e, const Qinfo* q ) const
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	return c->getX0();
}

void ZombieCompartment::setY0( const Eref& e, const Qinfo* q, double value )
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	c->setY0( value );
}

double ZombieCompartment::getY0( const Eref& e, const Qinfo* q ) const
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	return c->getY0();
}

void ZombieCompartment::setZ0( const Eref& e, const Qinfo* q, double value )
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	c->setZ0( value );
}

double ZombieCompartment::getZ0( const Eref& e, const Qinfo* q ) const
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	return c->getZ0();
}

void ZombieCompartment::setX( const Eref& e, const Qinfo* q, double value )
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	c->setX( value );
}

double ZombieCompartment::getX( const Eref& e, const Qinfo* q ) const
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	return c->getX();
}

void ZombieCompartment::setY( const Eref& e, const Qinfo* q, double value )
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	c->setY( value );
}

double ZombieCompartment::getY( const Eref& e, const Qinfo* q ) const
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	return c->getY();
}

void ZombieCompartment::setZ( const Eref& e, const Qinfo* q, double value )
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	c->setZ( value );
}

double ZombieCompartment::getZ( const Eref& e, const Qinfo* q ) const
{
	Compartment* c = reinterpret_cast< Compartment* >(
		e.element()->dataHandler()->data( 0 ) );
	
	return c->getZ();
}
*/

//////////////////////////////////////////////////////////////////
// ZombieCompartment::Dest function definitions.
//////////////////////////////////////////////////////////////////

void ZombieCompartment::dummy( const Eref& e, ProcPtr p )
{ ; }

void ZombieCompartment::process( const Eref& e, ProcPtr p )
{ ; }

void ZombieCompartment::reinit(  const Eref& e, ProcPtr p )
{ ; }

void ZombieCompartment::innerReinit(  const Eref& e, ProcPtr p )
{ ; }

void ZombieCompartment::initProc( const Eref& e, ProcPtr p )
{ ; }

void ZombieCompartment::initReinit( const Eref& e, ProcPtr p )
{ ; }

void ZombieCompartment::handleChannel( const Eref& e, const Qinfo* q, double Gk, double Ek )
{
	hsolve_->addGkEk( e.id(), Gk, Ek );
}

void ZombieCompartment::handleRaxial( double Ra, double Vm )
{ ; }

void ZombieCompartment::handleAxial( double Vm )
{ ; }

void ZombieCompartment::injectMsg( double current )
{
	hsolve_->addInject( e.id(), current );
}

void ZombieCompartment::randInject( double prob, double current )
{
	//~ if ( mtrand() < prob * dt_ ) {
		//~ sumInject_ += current;
	//~ }
}

void ZombieCompartment::cable()
{ ; }

//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////

// static func
void ZombieCompartment::zombify( Element* solver, Element* orig )
{
	// Delete "process" msg.
	static const Finfo* procDest = Compartment::initCinfo()->findFinfo( "process");
	assert( procDest );
	
	const DestFinfo* df = dynamic_cast< const DestFinfo* >( procDest );
	assert( df );
	MsgId mid = orig->findCaller( df->getFid() );
	if ( mid != Msg::bad )
		Msg::deleteMsg( mid );
	
	// Create zombie.
	//~ Element ze( orig->id(), zombieCompartmentCinfo, solver->dataHandler() );
	//~ Eref zer( &ze, 0 );
	DataHandler* dh = orig->dataHandler()->copyUsingNewDinfo(
		ZombieCompartment::initCinfo()->dinfo() );
	Eref oer( orig, 0 );
	Eref ser( solver, 0 );
	//~ ZombieCompartment* zd = reinterpret_cast< ZombieCompartment* >( zer.data() );
	ZombieCompartment* zd = reinterpret_cast< ZombieCompartment* >( dh->data( 0 ) );
	Compartment* od = reinterpret_cast< Compartment* >( oer.data() );
	HSolve* sd = reinterpret_cast< HSolve* >( ser.data() );
	zd->hsolve_ = sd;
	zd->copyFields( od );
	
	//~ unsigned int numEntries = orig->dataHandler()->localEntries();
	
	//~ DataHandler* zh = new ZombieHandler(
		//~ solver->dataHandler(), 
		//~ orig->dataHandler(),
		//~ 0,
		//~ numEntries );
	
	//~ orig->zombieSwap( zombieCompartmentCinfo, zh );
	orig->zombieSwap( zombieCompartmentCinfo, dh );
}

// static func
void ZombieCompartment::unzombify( Element* zombie )
{
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );
	
	//~ ZombieCompartment* z = reinterpret_cast< ZombieCompartment* >( zer.data() );
	
	// Creating data handler for original left for later.
	DataHandler* dh = 0;
	
	zombie->zombieSwap( Compartment::initCinfo(), dh );
}
