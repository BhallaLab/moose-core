/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**		   Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "../biophysics/Compartment.h"
#include "HinesMatrix.h"
#include "HSolveStruct.h"
#include "HSolvePassive.h"
#include "RateLookup.h"
#include "HSolveActive.h"
#include "HSolve.h"
#include "../biophysics/HHGate.h"
#include "../biophysics/ChanBase.h"
#include "../biophysics/HHChannel.h"
#include "ZombieHHChannel.h"

const Cinfo* ZombieHHChannel::initCinfo()
{
	/////////////////////////////////////////////////////////////////////
	// Shared messages
	/////////////////////////////////////////////////////////////////////
	static DestFinfo process( "process", 
		"Handles process call",
		new ProcOpFunc< ZombieHHChannel >( &ZombieHHChannel::process ) );
	static DestFinfo reinit( "reinit", 
		"Handles reinit call",
		new ProcOpFunc< ZombieHHChannel >( &ZombieHHChannel::reinit ) );
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
		static ElementValueFinfo< ZombieHHChannel, double > Gbar( "Gbar",
			"Maximal channel conductance",
			&ZombieHHChannel::setGbar,
			&ZombieHHChannel::getGbar
		);
		static ElementValueFinfo< ZombieHHChannel, double > Ek( "Ek", 
			"Reversal potential of channel",
			&ZombieHHChannel::setEk,
			&ZombieHHChannel::getEk
		);
		static ElementValueFinfo< ZombieHHChannel, double > Gk( "Gk",
			"Channel conductance variable",
			&ZombieHHChannel::setGk,
			&ZombieHHChannel::getGk
		);
		static ReadOnlyElementValueFinfo< ZombieHHChannel, double > Ik( "Ik",
			"Channel current variable",
			&ZombieHHChannel::getIk
		);
		static ElementValueFinfo< ZombieHHChannel, double > Xpower( "Xpower",
			"Power for X gate",
			&ZombieHHChannel::setXpower,
			&ZombieHHChannel::getXpower
		);
		static ElementValueFinfo< ZombieHHChannel, double > Ypower( "Ypower",
			"Power for Y gate",
			&ZombieHHChannel::setYpower,
			&ZombieHHChannel::getYpower
		);
		static ElementValueFinfo< ZombieHHChannel, double > Zpower( "Zpower",
			"Power for Z gate",
			&ZombieHHChannel::setZpower,
			&ZombieHHChannel::getZpower
		);
		static ElementValueFinfo< ZombieHHChannel, int > instant( "instant",
			"Bitmapped flag: bit 0 = Xgate, bit 1 = Ygate, bit 2 = Zgate"
			"When true, specifies that the lookup table value should be"
			"used directly as the state of the channel, rather than used"
			"as a rate term for numerical integration for the state",
			&ZombieHHChannel::setInstant,
			&ZombieHHChannel::getInstant
		);
		static ElementValueFinfo< ZombieHHChannel, double > X( "X", 
			"State variable for X gate",
			&ZombieHHChannel::setX,
			&ZombieHHChannel::getX
		);
		static ElementValueFinfo< ZombieHHChannel, double > Y( "Y",
			"State variable for Y gate",
			&ZombieHHChannel::setY,
			&ZombieHHChannel::getY
		);
		static ElementValueFinfo< ZombieHHChannel, double > Z( "Z",
			"State variable for Y gate",
			&ZombieHHChannel::setZ,
			&ZombieHHChannel::getZ
		);
		static ValueFinfo< ZombieHHChannel, int > useConcentration( 
			"useConcentration",
			"Flag: when true, use concentration message rather than Vm to"
			"control Z gate",
			&ZombieHHChannel::setUseConcentration,
			&ZombieHHChannel::getUseConcentration
		);
		
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
		 static DestFinfo concen( "concen", 
			 "Incoming message from Concen object to specific conc to use"
			 "in the Z gate calculations",
			 new OpFunc1< ZombieHHChannel, double >( &ZombieHHChannel::handleConc )
		 );
		 static DestFinfo createGate( "createGate",
			 "Function to create specified gate."
			 "Argument: Gate type [X Y Z]",
			 new EpFunc1< ZombieHHChannel, string >( &ZombieHHChannel::createGate )
		 );
///////////////////////////////////////////////////////
// FieldElementFinfo definition for HHGates. Note that these are made
// with the deferCreate flag off, so that the HHGates are created 
// right away even if they are empty.
// Assume only a single entry allocated in each gate.
///////////////////////////////////////////////////////
		 static FieldElementFinfo< ZombieHHChannel, HHGate > gateX( "gateX",
			 "Sets up HHGate X for channel",
			 HHGate::initCinfo(),
			 &ZombieHHChannel::getXgate,
			 &ZombieHHChannel::setNumGates,
			 &ZombieHHChannel::getNumXgates,
			 1
		 );
		 static FieldElementFinfo< ZombieHHChannel, HHGate > gateY( "gateY",
			 "Sets up HHGate Y for channel",
			 HHGate::initCinfo(),
			 &ZombieHHChannel::getYgate,
			 &ZombieHHChannel::setNumGates,
			 &ZombieHHChannel::getNumYgates,
			 1
		 );
		 static FieldElementFinfo< ZombieHHChannel, HHGate > gateZ( "gateZ",
			 "Sets up HHGate Z for channel",
			 HHGate::initCinfo(),
			 &ZombieHHChannel::getZgate,
			 &ZombieHHChannel::setNumGates,
			 &ZombieHHChannel::getNumZgates,
			 1
		 );
	
///////////////////////////////////////////////////////
	static Finfo* zombieHHChannelFinfos[] =
	{
		&proc,				// Shared
		&Gbar,				// Value
		&Ek,				// Value
		&Gk,				// Value
		&Ik,				// Value
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
		"Name", "ZombieHHChannel",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "ZombieHHChannel: Hodgkin-Huxley type voltage-gated Ion channel. Something "
		"like the old tabchannel from GENESIS, but also presents "
		"a similar interface as hhchan from GENESIS. ",
	};
	
	/*
	 * ZombieHHChannel derives directly from Neutral, unlike the regular
	 * HHChannel which derives from ChanBase. ChanBase handles fields like
	 * Gbar, Gk, Ek, Ik, which are common to HHChannel, SynChan, etc. On the
	 * other hand, these fields are stored separately for HHChannel and SynChan
	 * in the HSolver. Hence we cannot have a ZombieChanBase which does, for
	 * example:
	 *           hsolve_->setGk( id, Gk );
	 * Instead we must have ZombieHHChannel and ZombieSynChan which do:
	 *           hsolve_->setHHChannelGk( id, Gk );
	 * and:
	 *           hsolve_->setSynChanGk( id, Gk );
	 * respectively.
	 */
	static Cinfo zombieHHChannelCinfo(
		"ZombieHHChannel",
		ChanBase::initCinfo(),
		zombieHHChannelFinfos,
		sizeof( zombieHHChannelFinfos )/sizeof(Finfo *),
		new Dinfo< ZombieHHChannel >()
	);
	
	return &zombieHHChannelCinfo;
}

static const Cinfo* zombieHHChannelCinfo = ZombieHHChannel::initCinfo();
//////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////
ZombieHHChannel::ZombieHHChannel()
	:
	Xpower_( 0.0 ),
	Ypower_( 0.0 ),
	Zpower_( 0.0 ),
	useConcentration_( 0 )
	//~ xGate_( 0 ),
	//~ yGate_( 0 ),
	//~ zGate_( 0 )
{ ; }

void ZombieHHChannel::copyFields( Id chanId, HSolve* hsolve_ )
{
	//~ Xpower_           = Field< double >::get( chanId, "Xpower" );
	//~ Ypower_           = Field< double >::get( chanId, "Ypower" );
	//~ Zpower_           = Field< double >::get( chanId, "Zpower" );
	//~ useConcentration_ = Field< double >::get( chanId, "useConcentration" );
	//~ 
	//~ hsolve_->setPowers( chanId, Xpower_, Ypower_, Zpower_ );
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void ZombieHHChannel::setXpower( const Eref& e, const Qinfo* q, double Xpower )
{
	Xpower_ = Xpower;
	hsolve_->setPowers( e.id(), Xpower_, Ypower_, Zpower_ );
}

double ZombieHHChannel::getXpower( const Eref& e, const Qinfo* q ) const
{
	return Xpower_;
}

void ZombieHHChannel::setYpower( const Eref& e, const Qinfo* q, double Ypower )
{
	Ypower_ = Ypower;
	hsolve_->setPowers( e.id(), Xpower_, Ypower_, Zpower_ );
}

double ZombieHHChannel::getYpower( const Eref& e, const Qinfo* q ) const
{
	return Ypower_;
}

void ZombieHHChannel::setZpower( const Eref& e, const Qinfo* q, double Zpower )
{
	Zpower_ = Zpower;
	hsolve_->setPowers( e.id(), Xpower_, Ypower_, Zpower_ );
}

double ZombieHHChannel::getZpower( const Eref& e, const Qinfo* q ) const
{
	return Zpower_;
}

void ZombieHHChannel::setGbar( const Eref& e, const Qinfo* q, double Gbar )
{
	// cout << "in ZombieHHChannel::setGbar( " << e.id().path() << ", " << Gbar << " )\n";
	hsolve_->setHHChannelGbar( e.id(), Gbar );
}

double ZombieHHChannel::getGbar( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getHHChannelGbar( e.id() );
}

void ZombieHHChannel::setGk( const Eref& e, const Qinfo* q, double Gk )
{
	hsolve_->setGk( e.id(), Gk );
}

double ZombieHHChannel::getGk( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getGk( e.id() );
}

void ZombieHHChannel::setEk( const Eref& e, const Qinfo* q, double Ek )
{
	hsolve_->setEk( e.id(), Ek );
}

double ZombieHHChannel::getEk( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getEk( e.id() );
}

double ZombieHHChannel::getIk( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getIk( e.id() );
}

void ZombieHHChannel::setInstant( const Eref& e, const Qinfo* q, int instant )
{
	hsolve_->setInstant( e.id(), instant );
}

int ZombieHHChannel::getInstant( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getInstant( e.id() );
}

void ZombieHHChannel::setX( const Eref& e, const Qinfo* q, double X )
{
	hsolve_->setX( e.id(), X );
}

double ZombieHHChannel::getX( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getX( e.id() );
}

void ZombieHHChannel::setY( const Eref& e, const Qinfo* q, double Y )
{
	hsolve_->setY( e.id(), Y );
}

double ZombieHHChannel::getY( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getY( e.id() );
}

void ZombieHHChannel::setZ( const Eref& e, const Qinfo* q, double Z )
{
	hsolve_->setZ( e.id(), Z );
}

double ZombieHHChannel::getZ( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getZ( e.id() );
}

void ZombieHHChannel::setUseConcentration( int value )
{
	cerr << "Error: HSolve::setUseConcentration(): Cannot change "
		"'useConcentration' once HSolve has been setup.\n";
}

int ZombieHHChannel::getUseConcentration() const
{
	return useConcentration_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void ZombieHHChannel::process( const Eref& e, ProcPtr info )
{ ; }

void ZombieHHChannel::reinit( const Eref& er, ProcPtr info )
{ ; }

void ZombieHHChannel::handleConc( double conc )
{
    ;
}

void ZombieHHChannel::createGate(const Eref& e, const Qinfo * q, string name)
{
    ;
}

///////////////////////////////////////////////////
// HHGate functions
///////////////////////////////////////////////////


 HHGate* ZombieHHChannel::getXgate( unsigned int i )
 {
	 return NULL;
 }
 
 HHGate* ZombieHHChannel::getYgate( unsigned int i )
 {
	 return NULL;
 }
 
 HHGate* ZombieHHChannel::getZgate( unsigned int i )
 {
	 return NULL;
 }

void ZombieHHChannel::setNumGates(unsigned int num)
{;}

unsigned int ZombieHHChannel::getNumXgates() const
{
    return -1;
}
unsigned int ZombieHHChannel::getNumYgates() const
{
    return -1;
}
unsigned int ZombieHHChannel::getNumZgates() const
{
    return -1;
}

        
//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////

// static func
void ZombieHHChannel::zombify( Element* solver, Element* orig )
{
	// Delete "process" msg.
	static const Finfo* procDest = HHChannel::initCinfo()->findFinfo( "process");
	assert( procDest );
	
	const DestFinfo* df = dynamic_cast< const DestFinfo* >( procDest );
	assert( df );
	MsgId mid = orig->findCaller( df->getFid() );
	if ( mid != Msg::bad )
		Msg::deleteMsg( mid );

    // NOTE: the following line can be uncommented to remove messages
    // lying within the realm of HSolve. But HSolve will need to
    // maintain a datastructure for putting back the messages at
    // unzombify.

    HSolve::deleteIncomingMessages(orig, "concen");
    HSolve::deleteIncomingMessages(orig, "Vm");

	// Create zombie.
	DataHandler* dh = orig->dataHandler()->copyUsingNewDinfo(
		ZombieHHChannel::initCinfo()->dinfo() );
	Eref oer( orig, 0 );
	Eref ser( solver, 0 );
	ZombieHHChannel* zd = reinterpret_cast< ZombieHHChannel* >( dh->data( 0 ) );
	//~ HHChannel* od = reinterpret_cast< HHChannel* >( oer.data() );
	HSolve* sd = reinterpret_cast< HSolve* >( ser.data() );
	zd->hsolve_ = sd;
	zd->copyFields( oer.id(), sd );
	orig->zombieSwap( zombieHHChannelCinfo, dh );
}

// static func
void ZombieHHChannel::unzombify( Element* zombie )
{
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );
	
	//~ ZombieHHChannel* z = reinterpret_cast< ZombieHHChannel* >( zer.data() );
	
	// Creating data handler for original left for later.
	DataHandler* dh = 0;
	
	zombie->zombieSwap( HHChannel::initCinfo(), dh );
}
