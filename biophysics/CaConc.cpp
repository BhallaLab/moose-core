/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// #include <cfloat>
#include "header.h"
#include "CaConc.h"

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
static SrcFinfo1< double > *concOut() {
	static SrcFinfo1< double > concOut( "concOut", 
			"Concentration of Ca in pool" );
	return &concOut;
}

const Cinfo* CaConc::initCinfo()
{
	///////////////////////////////////////////////////////
	// Shared message definitions
	///////////////////////////////////////////////////////
	static DestFinfo process( "process", 
		"Handles process call",
		new ProcOpFunc< CaConc >( &CaConc::process ) );
	static DestFinfo reinit( "reinit", 
		"Handles reinit call",
		new ProcOpFunc< CaConc >( &CaConc::reinit ) );

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
	static ValueFinfo< CaConc, double > Ca( "Ca",
		"Calcium concentration.",
        &CaConc::setCa,
		&CaConc::getCa
	);
	static ValueFinfo< CaConc, double > CaBasal( "CaBasal",
		"Basal Calcium concentration.",
        &CaConc::setCaBasal,
		&CaConc::getCaBasal
	);
	static ValueFinfo< CaConc, double > Ca_base( "Ca_base",
		"Basal Calcium concentration, synonym for CaBasal",
        &CaConc::setCaBasal,
		&CaConc::getCaBasal
	);
	static ValueFinfo< CaConc, double > tau( "tau",
		"Settling time for Ca concentration",
        &CaConc::setTau,
		&CaConc::getTau
	);
	static ValueFinfo< CaConc, double > B( "B",
		"Volume scaling factor",
        &CaConc::setB,
		&CaConc::getB
	);
	static ValueFinfo< CaConc, double > thick( "thick",
		"Thickness of Ca shell.",
        &CaConc::setThickness,
		&CaConc::getThickness
	);
	static ValueFinfo< CaConc, double > ceiling( "ceiling",
		"Ceiling value for Ca concentration. If Ca > ceiling, Ca = ceiling. If ceiling <= 0.0, there is no upper limit on Ca concentration value.",
        &CaConc::setCeiling,
		&CaConc::getCeiling
	);
	static ValueFinfo< CaConc, double > floor( "floor",
		"Floor value for Ca concentration. If Ca < floor, Ca = floor",
        &CaConc::setFloor,
		&CaConc::getFloor
	);

///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////

	static DestFinfo current( "current", 
		"Calcium Ion current, due to be converted to conc.",
		new OpFunc1< CaConc, double >( &CaConc::current )
	);

	static DestFinfo currentFraction( "currentFraction", 
		"Fraction of total Ion current, that is carried by Ca2+.",
		new OpFunc2< CaConc, double, double >( &CaConc::currentFraction )
	);

	static DestFinfo increase( "increase", 
		"Any input current that increases the concentration.",
		new OpFunc1< CaConc, double >( &CaConc::increase )
	);

	static DestFinfo decrease( "decrease", 
		"Any input current that decreases the concentration.",
		new OpFunc1< CaConc, double >( &CaConc::decrease )
	);

	static DestFinfo basal( "basal", 
		"Synonym for assignment of basal conc.",
		new OpFunc1< CaConc, double >( &CaConc::setCaBasal )
	);

	static Finfo* CaConcFinfos[] =
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

	// We want the Ca updates before channel updates, so along with compts.
	// static SchedInfo schedInfo[] = { { process, 0, 0 } };

	static string doc[] =
	{
		"Name", "CaConc",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "CaConc: Calcium concentration pool. Takes current from a "
				"channel and keeps track of calcium buildup and depletion by a "
				"single exponential process. ",
	};	
	static Cinfo CaConcCinfo(
		"CaConc",
		Neutral::initCinfo(),
		CaConcFinfos,
		sizeof( CaConcFinfos )/sizeof(Finfo *),
		new Dinfo< CaConc >()
	);

	return &CaConcCinfo;
}
///////////////////////////////////////////////////

static const Cinfo* caConcCinfo = CaConc::initCinfo();

CaConc::CaConc()
	:
		Ca_( 0.0 ),
		CaBasal_( 0.0 ),
		tau_( 1.0 ),
		B_( 1.0 ),
		thickness_( 0.0 ),
		ceiling_( 1e9 ),
		floor_( -1e9 )
{;}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void CaConc::setCa( double Ca )
{
	Ca_ = Ca;
}
double CaConc::getCa() const
{
	return Ca_;
}

void CaConc::setCaBasal( double CaBasal )
{
	CaBasal_ = CaBasal;
}
double CaConc::getCaBasal() const
{
	return CaBasal_;
}

void CaConc::setTau( double tau )
{
	tau_ = tau;
}
double CaConc::getTau() const
{
	return tau_;
}

void CaConc::setB( double B )
{
	B_ = B;
}
double CaConc::getB() const
{
	return B_;
}
void CaConc::setThickness( double thickness )
{
    thickness_ = thickness;
}
double CaConc::getThickness() const
{
	return thickness_;
}
void CaConc::setCeiling( double ceiling )
{
    ceiling_ = ceiling;
}
double CaConc::getCeiling() const
{
	return ceiling_;
}

void CaConc::setFloor( double floor )
{
    floor_ = floor;
}
double CaConc::getFloor() const
{
	return floor_;
}


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void CaConc::reinit( const Eref& e, ProcPtr p )
{
	activation_ = 0.0;
	c_ = 0.0;
	Ca_ = CaBasal_;
	concOut()->send( e, p->threadIndexInGroup, Ca_ );
}

void CaConc::process( const Eref& e, ProcPtr p )
{
	double x = exp( -p->dt / tau_ );
	Ca_ = CaBasal_ + c_ * x + ( B_ * activation_ * tau_ )  * (1.0 - x);
	if (Ca_ > ceiling_){
		Ca_ = ceiling_;
	} else if ( Ca_ < floor_ ){
		Ca_ = floor_;
	}
	c_ = Ca_ - CaBasal_;
	concOut()->send( e, p->threadIndexInGroup, Ca_ );
	activation_ = 0;
}


void CaConc::current( double I )
{
	activation_ += I;
}

void CaConc::currentFraction( double I, double fraction )
{
	activation_ += I * fraction;
}

void CaConc::increase( double I )
{
	activation_ += fabs( I );
}

void CaConc::decrease( double I )
{
	activation_ -= fabs( I );
}

///////////////////////////////////////////////////
// Unit tests
///////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
void testCaConc()
{
	CaConc cc;
	double tau = 0.10;
	double basal = 0.0001;

	cc.setCa( basal );
	cc.setCaBasal( basal );
	cc.setTau( tau );
	// Here we use a volume of 1e-15 m^3, i.e., a 10 micron cube.
	cc.setB( 5.2e-6 / 1e-15 );
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
	
	ProcInfo p;
	p.dt = 0.001;
	p.currTime = 0.0;
	Id tempId = Id::nextId();
	Element temp( tempId, CaConc::initCinfo(), "temp", 0 );
	Eref er( &temp, 0 );
	cc.reinit( er, &p );

	double y;
	double conc;
	double delta = 0.0;
	for ( p.currTime = 0.0; p.currTime < 0.5; p.currTime += p.dt )
	{
		cc.current( curr );
		cc.process( er, &p );
		y = basal + 526.0e-6 * ( 1.0 - exp( -p.currTime / tau ) );
		conc = cc.getCa();
		delta += ( y - conc ) * ( y - conc );
	}
	assert( delta < 1e-6 );
	cout << "." << flush;
}
#endif 
