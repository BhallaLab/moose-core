/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <math.h>
#include <cfloat>
#include "moose.h"
#include "CaConc.h"
#include "../element/Neutral.h"

const Cinfo* initCaConcCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
				RFCAST( &CaConc::processFunc ) ),
	    new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
				RFCAST( &CaConc::reinitFunc ) ),
	};
	static Finfo* process =	new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo* ),
			"This is a shared message to receive Process message from the scheduler.The first entry is a MsgDest "
			"for the Process operation. It has a single argument, ProcInfo, which holds lots of information about "
			"current time, thread, dt and so on. The second entry is a MsgDest for the Reinit operation. "
			"It also uses ProcInfo. ");

///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////

	static Finfo* CaConcFinfos[] =
	{
		new ValueFinfo( "Ca", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &CaConc::getCa ), 
			RFCAST( &CaConc::setCa )
		),
		new ValueFinfo( "CaBasal", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &CaConc::getCaBasal ), 
			RFCAST( &CaConc::setCaBasal )
		),
		new ValueFinfo( "Ca_base", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &CaConc::getCaBasal ), 
			RFCAST( &CaConc::setCaBasal )
		),
		new ValueFinfo( "tau", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &CaConc::getTau ), 
			RFCAST( &CaConc::setTau )
		),
		new ValueFinfo( "B", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &CaConc::getB ), 
			RFCAST( &CaConc::setB )
		),
                new ValueFinfo( "thick", ValueFtype1< double >::global(),
                                reinterpret_cast< GetFunc >( &CaConc::getThickness ), 
                                RFCAST( &CaConc::setThickness ),
                                "Thickness of the shell for Ca concentration model."
		),
                new ValueFinfo( "ceiling", ValueFtype1< double >::global(),
                                reinterpret_cast< GetFunc >( &CaConc::getCeiling ), 
                                RFCAST( &CaConc::setCeiling ),
                                "Ceiling value for Ca concentration. If Ca > ceiling, Ca = ceiling. If ceiling <= 0.0, there is no upper limit on Ca concentration value."
		),
                new ValueFinfo( "floor", ValueFtype1< double >::global(),
                                reinterpret_cast< GetFunc >( &CaConc::getFloor ), 
                                RFCAST( &CaConc::setFloor ),
                                "Floor value for Ca concentration. If Ca < floor, Ca = floor."
		),

                
///////////////////////////////////////////////////////
// Shared message definitions
///////////////////////////////////////////////////////
		process,
		/*
		new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo * ) ),
			*/

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
		new SrcFinfo( "concSrc", Ftype1< double >::global() ),

///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
		new DestFinfo( "current", Ftype1< double >::global(),
				RFCAST( &CaConc::currentFunc ) ),
		new DestFinfo( "currentFraction", 
				Ftype2< double, double >::global(),
				RFCAST( &CaConc::currentFractionFunc ) ),
		new DestFinfo( "increase", Ftype1< double >::global(),
				RFCAST( &CaConc::increaseFunc ) ),
		new DestFinfo( "decrease", Ftype1< double >::global(),
				RFCAST( &CaConc::decreaseFunc ) ),
		new DestFinfo( "basalMsg", Ftype1< double >::global(),
				RFCAST( &CaConc::basalMsgFunc ) ),
	};

	// We want the Ca updates before channel updates, so along with compts.
	static SchedInfo schedInfo[] = { { process, 0, 0 } };

	static string doc[] =
	{
		"Name", "CaConc",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "CaConc: Calcium concentration pool. Takes current from a "
				"channel and keeps track of calcium buildup and depletion by a "
				"single exponential process. ",
	};	
	static Cinfo CaConcCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		CaConcFinfos,
		sizeof( CaConcFinfos )/sizeof(Finfo *),
		ValueFtype1< CaConc >::global(),
		schedInfo, 1
	);

	return &CaConcCinfo;
}

static const Cinfo* caConcCinfo = initCaConcCinfo();

static const Slot concSlot =
	initCaConcCinfo()->getSlot( "concSrc" );

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void CaConc::setCa( const Conn* c, double Ca )
{
	static_cast< CaConc* >( c->data() )->Ca_ = Ca;
}
double CaConc::getCa( Eref e )
{
	return static_cast< CaConc* >( e.data() )->Ca_;
}

void CaConc::setCaBasal( const Conn* c, double CaBasal )
{
	static_cast< CaConc* >( c->data() )->CaBasal_ = CaBasal;
}
double CaConc::getCaBasal( Eref e )
{
	return static_cast< CaConc* >( e.data() )->CaBasal_;
}

void CaConc::setTau( const Conn* c, double tau )
{
	static_cast< CaConc* >( c->data() )->tau_ = tau;
}
double CaConc::getTau( Eref e )
{
	return static_cast< CaConc* >( e.data() )->tau_;
}

void CaConc::setB( const Conn* c, double B )
{
	static_cast< CaConc* >( c->data() )->B_ = B;
}
double CaConc::getB( Eref e )
{
	return static_cast< CaConc* >( e.data() )->B_;
}
void CaConc::setThickness( const Conn * c, double thickness )
{
    static_cast< CaConc* >( c->data() )->thickness_ = thickness;
}
double CaConc::getThickness( Eref e )
{
	return static_cast< CaConc* >( e.data() )->thickness_;
}
void CaConc::setCeiling( const Conn * c, double ceiling )
{
    static_cast< CaConc* >( c->data() )->ceiling_ = ceiling;
}
double CaConc::getCeiling( Eref e )
{
	return static_cast< CaConc* >( e.data() )->ceiling_;
}

void CaConc::setFloor( const Conn * c, double floor )
{
    static_cast< CaConc* >( c->data() )->floor_ = floor;
}
double CaConc::getFloor( Eref e )
{
	return static_cast< CaConc* >( e.data() )->floor_;
}


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void CaConc::reinitFunc( const Conn* c, ProcInfo info )
{
	static_cast< CaConc* >( c->data() )->innerReinitFunc( c );
}

void CaConc::innerReinitFunc( const Conn* c )
{
	activation_ = 0.0;
	c_ = 0.0;
	Ca_ = CaBasal_;
	send1< double >( c->target(), concSlot, Ca_ );
}

void CaConc::processFunc( const Conn* c, ProcInfo info )
{
	static_cast< CaConc* >( c->data() )->innerProcessFunc( c, info );
}

void CaConc::innerProcessFunc( const Conn* conn, ProcInfo info )
{
	double x = exp( -info->dt_ / tau_ );
        Ca_ = CaBasal_ + c_ * x + ( B_ * activation_ * tau_ )  * ( 1.0 - x );
        if (Ca_ > ceiling_){
            Ca_ = ceiling_;
        } else if ( Ca_ < floor_ ){
            Ca_ = floor_;
        }
        c_ = Ca_ - CaBasal_;
	send1< double >( conn->target(), concSlot, Ca_ );
	activation_ = 0;
}


void CaConc::currentFunc( const Conn* c, double I )
{
	static_cast< CaConc* >( c->data() )->activation_ += I;
}

void CaConc::currentFractionFunc(
				const Conn* c, double I, double fraction )
{
	static_cast< CaConc* >( c->data() )->activation_ += I * fraction;
}

void CaConc::increaseFunc( const Conn* c, double I )
{
	static_cast< CaConc* >( c->data() )->activation_ += fabs( I );
}

void CaConc::decreaseFunc( const Conn* c, double I )
{
	static_cast< CaConc* >( c->data() )->activation_ -= fabs( I );
}

void CaConc::basalMsgFunc( const Conn* c, double value )
{
	static_cast< CaConc* >( c->data() )->CaBasal_ = value;
}

///////////////////////////////////////////////////
// Unit tests
///////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
void testCaConc()
{
	cout << "\nTesting CaConc" << flush;

	Element* n = Neutral::create( "Neutral", "n", Element::root()->id(), 
		Id::scratchId() );
	Element* Ca = Neutral::create( "CaConc", "Ca", n->id(), 
		Id::scratchId() );
	ASSERT( Ca != 0, "creating CaConc" );
	ProcInfoBase p;
	SetConn c( Ca, 0 );
	double tau = 0.10;
	double basal = 0.0001;
	CaConc::setCa( &c, basal );
	CaConc::setCaBasal( &c, basal );
	CaConc::setTau( &c, tau );
	// Here we use a volume of 1e-15 m^3, i.e., a 10 micron cube.
	CaConc::setB( &c, 5.2e-6 / 1e-15 );
	// Faraday constant = 96485.3415 s A / mol
	// Use a 1 pA input current. This should give (0.5e-12/F) moles/sec
	// influx, because Ca has valence of 2. 
	// So we get 5.2e-18 moles/sec coming in.
	// Our volume is 1e-15 m^3
	// So our buildup should be at 5.2e-3 moles/m^3/sec = 5.2 uM/sec
	double curr = 1e-12;

	// This will settle when efflux = influx
	// dC/dt = B*Ik - C/tau = 0.
	// so Ca = CaBasal + tau * B * Ik = 
	// 0.0001 + 0.1 * 5.2e-6 * 1e3 = 0.000626
	
	p.dt_ = 0.001;
	p.currTime_ = 0.0;
	CaConc::reinitFunc( &c, &p );
	double y;
	double conc;
	double delta = 0.0;
	for ( p.currTime_ = 0.0; p.currTime_ < 0.5; p.currTime_ += p.dt_ )
	{
		CaConc::currentFunc( &c, curr );
		CaConc::processFunc( &c, &p );
		y = basal + 526.0e-6 * ( 1.0 - exp( -p.currTime_ / tau ) );
		conc = CaConc::getCa( Ca );
		delta += ( y - conc ) * ( y - conc );
	}
	ASSERT( delta < 1e-6, "CaConc unit test" );
	// Get rid of the test stuff
	set( n, "destroy" );
}
#endif 
