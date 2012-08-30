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
#include "../biophysics/CaConc.h"
#include "HinesMatrix.h"
#include "HSolveStruct.h"
#include "HSolvePassive.h"
#include "RateLookup.h"
#include "HSolveActive.h"
#include "HSolve.h"
#include "ZombieCaConc.h"


///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
/*
 * Static function.
 * 
 * This Finfo is used to send out CaConc to channels.
 * The original CaConc sends this itself, whereas the HSolve
 * sends on behalf of the Zombie.
 */
SrcFinfo1< double >* ZombieCaConc::concOut() {
	static SrcFinfo1< double > concOut( "concOut", 
			"Concentration of Ca in pool" );
	return &concOut;
}

const Cinfo* ZombieCaConc::initCinfo()
{
	///////////////////////////////////////////////////////
	// Shared message definitions
	///////////////////////////////////////////////////////
	static DestFinfo process( "process", 
		"Handles process call",
		new ProcOpFunc< ZombieCaConc >( &ZombieCaConc::process ) );
	static DestFinfo reinit( "reinit", 
		"Handles reinit call",
		new ProcOpFunc< ZombieCaConc >( &ZombieCaConc::reinit ) );
	
	static Finfo* processShared[] =
	{
		&process, &reinit
	};
	
	static SharedFinfo proc( "proc", 
		"Shared message to receive Process message from scheduler",
		processShared, sizeof( processShared ) / sizeof( Finfo* ) );
		
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	static ElementValueFinfo< ZombieCaConc, double > Ca( "Ca",
		"Calcium concentration.",
		&ZombieCaConc::setCa,
		&ZombieCaConc::getCa
	);
	static ElementValueFinfo< ZombieCaConc, double > CaBasal( "CaBasal",
		"Basal Calcium concentration.",
		&ZombieCaConc::setCaBasal,
		&ZombieCaConc::getCaBasal
	);
	static ElementValueFinfo< ZombieCaConc, double > Ca_base( "Ca_base",
		"Basal Calcium concentration, synonym for CaBasal",
		&ZombieCaConc::setCaBasal,
		&ZombieCaConc::getCaBasal
	);
	static ElementValueFinfo< ZombieCaConc, double > tau( "tau",
		"Settling time for Ca concentration",
		&ZombieCaConc::setTau,
		&ZombieCaConc::getTau
	);
	static ElementValueFinfo< ZombieCaConc, double > B( "B",
		"Volume scaling factor",
		&ZombieCaConc::setB,
		&ZombieCaConc::getB
	);
	// Local field, hence 'ElementValueFinfo' not needed.
	static ValueFinfo< ZombieCaConc, double > thick( "thick",
		"Thickness of Ca shell.",
		&ZombieCaConc::setThickness,
		&ZombieCaConc::getThickness
	);
	static ElementValueFinfo< ZombieCaConc, double > ceiling( "ceiling",
		"Ceiling value for Ca concentration. If Ca > ceiling, Ca = ceiling. If ceiling <= 0.0, there is no upper limit on Ca concentration value.",
		&ZombieCaConc::setCeiling,
		&ZombieCaConc::getCeiling
	);
	static ElementValueFinfo< ZombieCaConc, double > floor( "floor",
		"Floor value for Ca concentration. If Ca < floor, Ca = floor",
		&ZombieCaConc::setFloor,
		&ZombieCaConc::getFloor
	);
	
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	
	static DestFinfo current( "current", 
		"Calcium Ion current, due to be converted to conc.",
		new OpFunc1< ZombieCaConc, double >( &ZombieCaConc::current )
	);
	
	static DestFinfo currentFraction( "currentFraction", 
		"Fraction of total Ion current, that is carried by Ca2+.",
		new OpFunc2< ZombieCaConc, double, double >( &ZombieCaConc::currentFraction )
	);
	
	static DestFinfo increase( "increase", 
		"Any input current that increases the concentration.",
		new OpFunc1< ZombieCaConc, double >( &ZombieCaConc::increase )
	);
	
	static DestFinfo decrease( "decrease", 
		"Any input current that decreases the concentration.",
		new OpFunc1< ZombieCaConc, double >( &ZombieCaConc::decrease )
	);
	
	static DestFinfo basal( "basal", 
		"Synonym for assignment of basal conc.",
		new EpFunc1< ZombieCaConc, double >( &ZombieCaConc::setCaBasal )
	);
	
	static Finfo* caConcFinfos[] =
	{
		&proc,		// Shared 
		concOut(),	// Src
		&Ca,		// Value
		&CaBasal,	// Value
		&Ca_base,	// Value
		&tau,		// Value
		&B,			// Value
		&thick,		// Value
		&ceiling,	// Value
		&floor,		// Value
		&current,	// Dest
		&currentFraction,	// Dest
		&increase,	// Dest
		&decrease,	// Dest
		&basal,		// Dest
	};
	
	static string doc[] =
	{
		"Name", "ZombieCaConc",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "ZombieCaConc: Calcium concentration pool. Takes current from a "
				"channel and keeps track of calcium buildup and depletion by a "
				"single exponential process. ",
	};	
	static Cinfo zombieCaConcCinfo(
		"ZombieCaConc",
		Neutral::initCinfo(),
		caConcFinfos,
		sizeof( caConcFinfos )/sizeof(Finfo *),
		new Dinfo< ZombieCaConc >()
	);
	
	return &zombieCaConcCinfo;
}
///////////////////////////////////////////////////

static const Cinfo* zombieCaConcCinfo = ZombieCaConc::initCinfo();

void ZombieCaConc::copyFields( CaConc* c )
{
	tau_       = c->getTau();
	B_         = c->getB();
	thickness_ = c->getThickness();
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void ZombieCaConc::setCa( const Eref& e, const Qinfo* q, double Ca )
{
	hsolve_->setCa( e.id(), Ca );
}

double ZombieCaConc::getCa( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getCa( e.id() );
}

void ZombieCaConc::setCaBasal( const Eref& e, const Qinfo* q, double CaBasal )
{
	hsolve_->setCa( e.id(), CaBasal );
}

double ZombieCaConc::getCaBasal( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getCaBasal( e.id() );
}

void ZombieCaConc::setTau( const Eref& e, const Qinfo* q, double tau )
{
	tau_ = tau;
	hsolve_->setTauB( e.id(), tau_, B_ );
}

double ZombieCaConc::getTau( const Eref& e, const Qinfo* q ) const
{
	return tau_;
}

void ZombieCaConc::setB( const Eref& e, const Qinfo* q, double B )
{
	B_ = B;
	hsolve_->setTauB( e.id(), tau_, B_ );
}

double ZombieCaConc::getB( const Eref& e, const Qinfo* q ) const
{
	return B_;
}

void ZombieCaConc::setCeiling( const Eref& e, const Qinfo* q, double ceiling )
{
	hsolve_->setCaCeiling( e.id(), ceiling );
}

double ZombieCaConc::getCeiling( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getCaCeiling( e.id() );
}

void ZombieCaConc::setFloor( const Eref& e, const Qinfo* q, double floor )
{
	hsolve_->setCaFloor( e.id(), floor );
}

double ZombieCaConc::getFloor( const Eref& e, const Qinfo* q ) const
{
	return hsolve_->getCaFloor( e.id() );
}

void ZombieCaConc::setThickness( double thickness )
{
	thickness_ = thickness;
}

double ZombieCaConc::getThickness() const
{
	return thickness_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void ZombieCaConc::reinit( const Eref& e, ProcPtr p )
{
	;
}

void ZombieCaConc::process( const Eref& e, ProcPtr p )
{
	;
}

void ZombieCaConc::current( double I )
{
	//~ activation_ += I;
}

void ZombieCaConc::currentFraction( double I, double fraction )
{
	//~ activation_ += I * fraction;
}

void ZombieCaConc::increase( double I )
{
	//~ activation_ += fabs( I );
}

void ZombieCaConc::decrease( double I )
{
	//~ activation_ -= fabs( I );
}

//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////

// static func
void ZombieCaConc::zombify( Element* solver, Element* orig )
{
	// Delete "process" msg.
	static const Finfo* procDest = CaConc::initCinfo()->findFinfo( "process");
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
    
    // HSolve::deleteIncomingMessages(orig, "current");

	// Create zombie.
	DataHandler* dh = orig->dataHandler()->copyUsingNewDinfo(
		ZombieCaConc::initCinfo()->dinfo() );
	Eref oer( orig, 0 );
	Eref ser( solver, 0 );
	ZombieCaConc* zd = reinterpret_cast< ZombieCaConc* >( dh->data( 0 ) );
	CaConc* od = reinterpret_cast< CaConc* >( oer.data() );
	HSolve* sd = reinterpret_cast< HSolve* >( ser.data() );
	zd->hsolve_ = sd;
	zd->copyFields( od );
	
	orig->zombieSwap( zombieCaConcCinfo, dh );
}

// static func
void ZombieCaConc::unzombify( Element* zombie )
{
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );
	
	//~ ZombieCaConc* z = reinterpret_cast< ZombieCaConc* >( zer.data() );
	
	// Creating data handler for original left for later.
	DataHandler* dh = 0;
	
	zombie->zombieSwap( CaConc::initCinfo(), dh );
}
