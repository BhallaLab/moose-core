/**********************************************************************
** This program is part of 'MOOSE', the
** Multiscale Object Oriented Simulation Environment.
**           copyright (C) 2003-2008
**           Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "DifShell.h"
#include <cmath>

const Cinfo* initDifShellCinfo()
{
	static Finfo* processShared_0[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &DifShell::process_0 ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &DifShell::reinit_0 ),
			"Reinit happens only in stage 0" ),
	};
	static Finfo* process_0 = new SharedFinfo( "process_0", processShared_0,
		sizeof( processShared_0 ) / sizeof( Finfo* ),
		"Here we create 2 shared finfos to attach with the Ticks. This is because we want to perform DifShell "
		"computations in 2 stages, much as in the Compartment object. "
		"In the first stage we send out the concentration value to other DifShells and Buffer elements. We also "
		"receive fluxes and currents and sum them up to compute ( dC / dt ). "
		"In the second stage we find the new C value using an explicit integration method. "
		"This 2-stage procedure eliminates the need to store and send prev_C values, as was common in GENESIS."	);
	
	static Finfo* processShared_1[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &DifShell::process_1 ) ),
		new DestFinfo( "reinit", Ftype0::global(),
			&dummyFunc, 
			"Reinit happens only in stage 0" ),
	};
	static Finfo* process_1 = new SharedFinfo( "process_1", processShared_1,
		sizeof( processShared_1 ) / sizeof( Finfo* ) );
	
	static Finfo* bufferShared[] =
	{
		new SrcFinfo( "concentration", Ftype1< double >::global() ),
		new DestFinfo( "reaction",
			Ftype4< double, double, double, double >::global(),
			RFCAST( &DifShell::buffer ),
			"Here the DifShell receives reaction rates (forward and backward), and concentrations for the "
			"free-buffer and bound-buffer molecules."	),
	};

	static Finfo* innerDifShared[] =
	{
		new SrcFinfo( "source", Ftype2< double, double >::global() ),
		new DestFinfo( "dest", Ftype2< double, double >::global(),
			RFCAST( &DifShell::fluxFromOut ) ),
	};

	static Finfo* outerDifShared[] =
	{
		new DestFinfo( "dest", Ftype2< double, double >::global(),
			RFCAST( &DifShell::fluxFromIn ) ),
		new SrcFinfo( "source", Ftype2< double, double >::global() ),
	};

	static Finfo* difShellFinfos[] = 
	{
	//////////////////////////////////////////////////////////////////
	// Field definitions
	//////////////////////////////////////////////////////////////////
		new ValueFinfo( "C", ValueFtype1< double >::global(),
			GFCAST( &DifShell::getC ),
			&dummyFunc,
			"Concentration C is computed by the DifShell and is read-only"
		),
		new ValueFinfo( "Ceq", ValueFtype1< double >::global(),
			GFCAST( &DifShell::getCeq ),
			RFCAST( &DifShell::setCeq )
		),
		new ValueFinfo( "D", ValueFtype1< double >::global(),
			GFCAST( &DifShell::getD ),
			RFCAST( &DifShell::setD )
		),
		new ValueFinfo( "valence", ValueFtype1< double >::global(),
			GFCAST( &DifShell::getValence ),
			RFCAST( &DifShell::setValence )
		),
		new ValueFinfo( "leak", ValueFtype1< double >::global(),
			GFCAST( &DifShell::getLeak ),
			RFCAST( &DifShell::setLeak )
		),
		new ValueFinfo( "shapeMode", ValueFtype1< unsigned int >::global(),
			GFCAST( &DifShell::getShapeMode ),
			RFCAST( &DifShell::setShapeMode )
		),
		new ValueFinfo( "length", ValueFtype1< double >::global(),
			GFCAST( &DifShell::getLength ),
			RFCAST( &DifShell::setLength )
		),
		new ValueFinfo( "diameter", ValueFtype1< double >::global(),
			GFCAST( &DifShell::getDiameter ),
			RFCAST( &DifShell::setDiameter )
		),
		new ValueFinfo( "thickness", ValueFtype1< double >::global(),
			GFCAST( &DifShell::getThickness ),
			RFCAST( &DifShell::setThickness )
		),
		new ValueFinfo( "volume", ValueFtype1< double >::global(),
			GFCAST( &DifShell::getVolume ),
			RFCAST( &DifShell::setVolume )
		),
		new ValueFinfo( "outerArea", ValueFtype1< double >::global(),
			GFCAST( &DifShell::getOuterArea ),
			RFCAST( &DifShell::setOuterArea )
		),
		new ValueFinfo( "innerArea", ValueFtype1< double >::global(),
			GFCAST( &DifShell::getInnerArea ),
			RFCAST( &DifShell::setInnerArea )
		),
	//////////////////////////////////////////////////////////////////
	// MsgSrc definitions
	//////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////
	// SharedFinfo definitions
	//////////////////////////////////////////////////////////////////
		process_0,
		process_1,
		new SharedFinfo( "buffer", bufferShared,
			sizeof( bufferShared ) / sizeof( Finfo* ),
			"This is a shared message from a DifShell to a Buffer (FixBuffer or DifBuffer). "
			"During stage 0:\n "
			"- DifShell sends ion concentration \n"
			"- Buffer updates buffer concentration and sends it back immediately using a call-back.\n"
			"- DifShell updates the time-derivative ( dC / dt ) \n"
	 		"During stage 1: \n"
			"- DifShell advances concentration C \n"
			"This scheme means that the Buffer does not need to be scheduled, and it does its computations when "
			"it receives a cue from the DifShell. May not be the best idea, but it saves us from doing the above "
			"computations in 3 stages instead of 2." ),
		new SharedFinfo( "innerDif", innerDifShared,
			sizeof( innerDifShared ) / sizeof( Finfo* ),
			"This shared message (and the next) is between DifShells: adjoining shells exchange information to "
			"find out the flux between them. "
			"Using this message, an inner shell sends to, and receives from its outer shell." ),
		new SharedFinfo( "outerDif", outerDifShared,
			sizeof( outerDifShared ) / sizeof( Finfo* ),
			"Using this message, an outer shell sends to, and receives from its inner shell." ),

	//////////////////////////////////////////////////////////////////
	// DestFinfo definitions
	//////////////////////////////////////////////////////////////////
		new DestFinfo( "influx",
			Ftype1< double >::global(),
			RFCAST( &DifShell::influx ) ),
		new DestFinfo( "outflux",
			Ftype1< double >::global(),
			RFCAST( &DifShell::influx ) ),
		new DestFinfo( "fInflux",
			Ftype2< double, double >::global(),
			RFCAST( &DifShell::fInflux ) ),
		new DestFinfo( "fOutflux",
			Ftype2< double, double >::global(),
			RFCAST( &DifShell::fOutflux ) ),
		new DestFinfo( "storeInflux",
			Ftype1< double >::global(),
			RFCAST( &DifShell::storeInflux ) ),
		new DestFinfo( "storeOutflux",
			Ftype1< double >::global(),
			RFCAST( &DifShell::storeOutflux ) ),
		new DestFinfo( "tauPump",
			Ftype2< double, double >::global(),
			RFCAST( &DifShell::tauPump ) ),
		new DestFinfo( "eqTauPump",
			Ftype1< double >::global(),
			RFCAST( &DifShell::eqTauPump ) ),
		new DestFinfo( "mmPump",
			Ftype2< double, double >::global(),
			RFCAST( &DifShell::mmPump ) ),
		new DestFinfo( "hillPump",
			Ftype3< double, double, unsigned int >::global(),
			RFCAST( &DifShell::hillPump ) ),
	};

	static SchedInfo schedInfo[] = { { process_0, 0, 0 }, { process_1, 0, 1 } };

	static string doc[] =
	{
		"Name", "DifShell",
		"Author", "Niraj Dudani",
		"Description", "DifShell object: Models diffusion of an ion (typically calcium) within an "
				"electric compartment. A DifShell is an iso-concentration region with respect to "
				"the ion. Adjoining DifShells exchange flux of this ion, and also keep track of "
				"changes in concentration due to pumping, buffering and channel currents, by "
				"talking to the appropriate objects.",
	};	
	static Cinfo difShellCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		difShellFinfos,
		sizeof( difShellFinfos ) / sizeof( Finfo* ),
		ValueFtype1< DifShell >::global(),
		schedInfo, 2
	);

	return &difShellCinfo;
}

static const Cinfo* difShellCinfo = initDifShellCinfo();

static const Slot bufferSlot =
	initDifShellCinfo()->getSlot( "buffer.concentration" );
static const Slot innerDifSlot =
	initDifShellCinfo()->getSlot( "innerDif.source" );
static const Slot outerDifSlot =
	initDifShellCinfo()->getSlot( "outerDif.source" );

////////////////////////////////////////////////////////////////////////////////
// Class functions
////////////////////////////////////////////////////////////////////////////////

/// Faraday's constant (Coulomb / Mole)
const double DifShell::F_ = 0.0;

DifShell::DifShell() :
	dCbyDt_( 0.0 ),
	C_( 0.0 ),
	Ceq_( 0.0 ),
	D_( 0.0 ),
	valence_( 0.0 ),
	leak_( 0.0 ),
	shapeMode_( 0 ),
	length_( 0.0 ),
	diameter_( 0.0 ),
	thickness_( 0.0 ),
	volume_( 0.0 ),
	outerArea_( 0.0 ),
	innerArea_( 0.0 )
{ ; }

////////////////////////////////////////////////////////////////////////////////
// Field access functions
////////////////////////////////////////////////////////////////////////////////
/// C is a read-only field
double DifShell::getC( Eref e )
{
	return static_cast< DifShell* >( e.data() )->C_;
}

void DifShell::setCeq( const Conn* c, double Ceq )
{
	if ( Ceq < 0.0 ) {
		cerr << "Error: DifShell: Ceq cannot be negative!\n";
		return;
	}
	
	static_cast< DifShell* >( c->data() )->Ceq_ = Ceq;
}

double DifShell::getCeq( Eref e )
{
	return static_cast< DifShell* >( e.data() )->Ceq_;
}

void DifShell::setD( const Conn* c, double D )
{
	if ( D < 0.0 ) {
		cerr << "Error: DifShell: D cannot be negative!\n";
		return;
	}
	
	static_cast< DifShell* >( c->data() )->D_ = D;
}

double DifShell::getD( Eref e )
{
	return static_cast< DifShell* >( e.data() )->D_;
}

void DifShell::setValence( const Conn* c, double valence )
{
	if ( valence < 0.0 ) {
		cerr << "Error: DifShell: valence cannot be negative!\n";
		return;
	}
	
	static_cast< DifShell* >( c->data() )->valence_ = valence;
}

double DifShell::getValence( Eref e )
{
	return static_cast< DifShell* >( e.data() )->valence_;
}

void DifShell::setLeak( const Conn* c, double leak )
{
	static_cast< DifShell* >( c->data() )->leak_ = leak;
}

double DifShell::getLeak( Eref e )
{
	return static_cast< DifShell* >( e.data() )->leak_;
}

void DifShell::setShapeMode( const Conn* c, unsigned int shapeMode )
{
	if ( shapeMode != 0 && shapeMode != 1 && shapeMode != 3 ) {
		cerr << "Error: DifShell: I only understand shapeModes 0, 1 and 3.\n";
		return;
	}
	static_cast< DifShell* >( c->data() )->shapeMode_ = shapeMode;
}

unsigned int DifShell::getShapeMode( Eref e )
{
	return static_cast< DifShell* >( e.data() )->shapeMode_;
}

void DifShell::setLength( const Conn* c, double length )
{
	if ( length < 0.0 ) {
		cerr << "Error: DifShell: length cannot be negative!\n";
		return;
	}
	
	static_cast< DifShell* >( c->data() )->length_ = length;
}

double DifShell::getLength( Eref e )
{
	return static_cast< DifShell* >( e.data() )->length_;
}

void DifShell::setDiameter( const Conn* c, double diameter )
{
	if ( diameter < 0.0 ) {
		cerr << "Error: DifShell: diameter cannot be negative!\n";
		return;
	}
	
	static_cast< DifShell* >( c->data() )->diameter_ = diameter;
}

double DifShell::getDiameter( Eref e )
{
	return static_cast< DifShell* >( e.data() )->diameter_;
}

void DifShell::setThickness( const Conn* c, double thickness )
{
	if ( thickness < 0.0 ) {
		cerr << "Error: DifShell: thickness cannot be negative!\n";
		return;
	}
	
	static_cast< DifShell* >( c->data() )->thickness_ = thickness;
}

double DifShell::getThickness( Eref e )
{
	return static_cast< DifShell* >( e.data() )->thickness_;
}

void DifShell::setVolume( const Conn* c, double volume )
{
	DifShell* difshell = static_cast< DifShell* >( c->data() );
	
	if ( difshell->shapeMode_ != 3 )
		cerr << "Warning: DifShell: Trying to set volume, when shapeMode is not USER-DEFINED\n";
	
	if ( volume < 0.0 ) {
		cerr << "Error: DifShell: volume cannot be negative!\n";
		return;
	}
	
	difshell->volume_ = volume;
}

double DifShell::getVolume( Eref e )
{
	return static_cast< DifShell* >( e.data() )->volume_;
}

void DifShell::setOuterArea( const Conn* c, double outerArea )
{
	DifShell* difshell = static_cast< DifShell* >( c->data() );
	
	if ( difshell->shapeMode_ != 3 )
		cerr << "Warning: DifShell: Trying to set outerArea, when shapeMode is not USER-DEFINED\n";
	
	if ( outerArea < 0.0 ) {
		cerr << "Error: DifShell: outerArea cannot be negative!\n";
		return;
	}
	
	difshell->outerArea_ = outerArea;
}

double DifShell::getOuterArea( Eref e )
{
	return static_cast< DifShell* >( e.data() )->outerArea_;
}

void DifShell::setInnerArea( const Conn* c, double innerArea )
{
	DifShell* difshell = static_cast< DifShell* >( c->data() );
	
	if ( difshell->shapeMode_ != 3 )
		cerr << "Warning: DifShell: Trying to set innerArea, when shapeMode is not USER-DEFINED\n";
	
	if ( innerArea < 0.0 ) {
		cerr << "Error: DifShell: innerArea cannot be negative!\n";
		return;
	}
	
	difshell->innerArea_ = innerArea;
}

double DifShell::getInnerArea( Eref e )
{
	return static_cast< DifShell* >( e.data() )->innerArea_;
}

////////////////////////////////////////////////////////////////////////////////
// Dest functions
////////////////////////////////////////////////////////////////////////////////

void DifShell::reinit_0( const Conn* c, ProcInfo p )
{
	static_cast< DifShell* >( c->data() )->localReinit_0( p );
}

void DifShell::process_0( const Conn* c, ProcInfo p )
{
	static_cast< DifShell* >( c->data() )->localProcess_0( c->target(), p );
}

void DifShell::process_1( const Conn* c, ProcInfo p )
{
	static_cast< DifShell* >( c->data() )->localProcess_1( p );
}

void DifShell::buffer(
	const Conn* c,
	double kf,
	double kb,
	double bFree,
	double bBound )
{
	static_cast< DifShell* >( c->data() )->localBuffer( kf, kb, bFree, bBound );
}

void DifShell::fluxFromOut(
	const Conn* c,
	double outerC,
	double outerThickness )
{
	static_cast< DifShell* >( c->data() )->
		localFluxFromOut( outerC, outerThickness );
}

void DifShell::fluxFromIn(
	const Conn* c,
	double innerC,
	double innerThickness )
{
	static_cast< DifShell* >( c->data() )->
		localFluxFromIn( innerC, innerThickness );
}

void DifShell::influx(
	const Conn* c,
	double I )
{
	static_cast< DifShell* >( c->data() )->localInflux( I );
}

void DifShell::outflux(
	const Conn* c,
	double I )
{
	static_cast< DifShell* >( c->data() )->localOutflux( I );
}

void DifShell::fInflux(
	const Conn* c,
	double I,
	double fraction )
{
	static_cast< DifShell* >( c->data() )->localFInflux( I, fraction );
}

void DifShell::fOutflux(
	const Conn* c,
	double I,
	double fraction )
{
	static_cast< DifShell* >( c->data() )->localFOutflux( I, fraction );
}

void DifShell::storeInflux(
	const Conn* c,
	double flux )
{
	static_cast< DifShell* >( c->data() )->localStoreInflux( flux );
}

void DifShell::storeOutflux(
	const Conn* c,
	double flux )
{
	static_cast< DifShell* >( c->data() )->localStoreOutflux( flux );
}

void DifShell::tauPump(
	const Conn* c,
	double kP,
	double Ceq )
{
	static_cast< DifShell* >( c->data() )->localTauPump( kP, Ceq );
}

void DifShell::eqTauPump(
	const Conn* c,
	double kP )
{
	static_cast< DifShell* >( c->data() )->localEqTauPump( kP );
}

void DifShell::mmPump(
	const Conn* c,
	double vMax,
	double Kd )
{
	static_cast< DifShell* >( c->data() )->localMMPump( vMax, Kd );
}

void DifShell::hillPump(
	const Conn* c,
	double vMax,
	double Kd,
	unsigned int hill )
{
	static_cast< DifShell* >( c->data() )->localHillPump( vMax, Kd, hill );
}

////////////////////////////////////////////////////////////////////////////////
// Local dest functions
////////////////////////////////////////////////////////////////////////////////

void DifShell::localReinit_0( ProcInfo p )
{
	dCbyDt_ = leak_;
	
	double Pi = M_PI;
	double dOut = diameter_;
	double dIn = diameter_ - thickness_;
	
	switch ( shapeMode_ )
	{
	/*
	 * Onion Shell
	 */
	case 0:
		if ( length_ == 0.0 ) { // Spherical shell
			volume_ = ( Pi / 6.0 ) * ( dOut * dOut * dOut - dIn * dIn * dIn );
			outerArea_ = Pi * dOut * dOut;
			innerArea_ = Pi * dIn * dIn;
		} else { // Cylindrical shell
			volume_ = ( Pi * length_ / 4.0 ) * ( dOut * dOut - dIn * dIn );
			outerArea_ = Pi * dOut * length_;
			innerArea_ = Pi * dIn * length_;
		}
		
		break;
	
	/*
	 * Cylindrical Slice
	 */
	case 1:
		volume_ = Pi * diameter_ * diameter_ * thickness_ / 4.0;
		outerArea_ = Pi * diameter_ * diameter_ / 4.0;
		innerArea_ = outerArea_;
		break;
	
	/*
	 * User defined
	 */
	case 3:
		// Nothing to be done here. Volume and inner-, outer areas specified by
		// user.
		break;
	
	default:
		assert( 0 );
	}
}

void DifShell::localProcess_0( Eref difshell, ProcInfo p )
{
	/**
	 * Send ion concentration and thickness to adjacent DifShells. They will
	 * then compute their incoming fluxes.
	 */
	send2< double, double >( difshell, innerDifSlot, C_, thickness_ );
	send2< double, double >( difshell, outerDifSlot, C_, thickness_ );
	
	/**
	 * Send ion concentration to ion buffers. They will send back information on
	 * the reaction (forward / backward rates ; free / bound buffer concentration)
	 * immediately, which this DifShell will use to find amount of ion captured
	 * or released in the current time-step.
	 */
	send1< double >( difshell, bufferSlot, C_ );
}

void DifShell::localProcess_1( ProcInfo p )
{
	C_ += dCbyDt_ * p->dt_;
	dCbyDt_ = leak_;
}

void DifShell::localBuffer(
	double kf,
	double kb,
	double bFree,
	double bBound )
{
	dCbyDt_ += -kf * bFree * C_ + kb * bBound;
}

void DifShell::localFluxFromOut( double outerC, double outerThickness )
{
	double dx = ( outerThickness + thickness_ ) / 2.0;
	
	/**
	 * We could pre-compute ( D / Volume ), but let us leave the optimizations
	 * for the solver.
	 */
	dCbyDt_ += ( D_ / volume_ ) * ( outerArea_ / dx ) * ( outerC - C_ );
}

void DifShell::localFluxFromIn( double innerC, double innerThickness )
{
	double dx = ( innerThickness + thickness_ ) / 2.0;
	
	dCbyDt_ += ( D_ / volume_ ) * ( innerArea_ / dx ) * ( innerC - C_ );
}

void DifShell::localInflux(	double I )
{
	/**
	 * I: Amperes
	 * F_: Faraday's constant: Coulomb / mole
	 * valence_: charge on ion: dimensionless
	 */
	dCbyDt_ += I / ( F_ * valence_ * volume_ );
}

/**
 * Same as influx, except subtracting.
 */
void DifShell::localOutflux( double I )
{
	dCbyDt_ -= I / ( F_ * valence_ * volume_ );
}

void DifShell::localFInflux( double I, double fraction )
{
	dCbyDt_ += fraction * I / ( F_ * valence_ * volume_ );
}

void DifShell::localFOutflux( double I, double fraction )
{
	dCbyDt_ -= fraction * I / ( F_ * valence_ * volume_ );
}

void DifShell::localStoreInflux( double flux )
{
	dCbyDt_ += flux / volume_;
}

void DifShell::localStoreOutflux( double flux )
{
	dCbyDt_ -= flux / volume_;
}

void DifShell::localTauPump( double kP, double Ceq )
{
	dCbyDt_ += -kP * ( C_ - Ceq );
}

/**
 * Same as tauPump, except that we use the local value of Ceq.
 */
void DifShell::localEqTauPump( double kP )
{
	dCbyDt_ += -kP * ( C_ - Ceq_ );
}

void DifShell::localMMPump( double vMax, double Kd )
{
	dCbyDt_ += -( vMax / volume_ ) * ( C_ / ( C_ + Kd ) );
}

void DifShell::localHillPump( double vMax, double Kd, unsigned int hill )
{
	double ch;
	switch( hill )
	{
	case 0:
		ch = 1.0;
		break;
	case 1:
		ch = C_;
		break;
	case 2:
		ch = C_ * C_;
		break;
	case 3:
		ch = C_ * C_ * C_;
		break;
	case 4:
		ch = C_ * C_;
		ch = ch * ch;
		break;
	default:
		ch = pow( C_, static_cast< double >( hill ) );
	};
	
	dCbyDt_ += -( vMax / volume_ ) * ( ch / ( ch + Kd ) );
}

////////////////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include "../element/Neutral.h"

void testDifShell()
{
}
#endif
