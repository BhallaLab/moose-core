/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "moose.h"
#include "setgetLookup.h"
#include "../element/Neutral.h"
#include <math.h>
#include "KineticManager.h"

map< string, MethodInfo > KineticManager::methodMap_;

const Cinfo* initKineticManagerCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &KineticManager::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &KineticManager::reinitFunc ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ) );

	static Finfo* kineticManagerFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "auto", 
			ValueFtype1< bool >::global(),
			GFCAST( &KineticManager::getAuto ), 
			RFCAST( &KineticManager::setAuto ) 
		),
		new ValueFinfo( "stochastic", 
			ValueFtype1< bool >::global(),
			GFCAST( &KineticManager::getStochastic ), 
			RFCAST( &KineticManager::setStochastic )
		),
		new ValueFinfo( "spatial", 
			ValueFtype1< bool >::global(),
			GFCAST( &KineticManager::getSpatial ), 
			RFCAST( &KineticManager::setSpatial )
		),
		new ValueFinfo( "method", 
			ValueFtype1< string >::global(),
			GFCAST( &KineticManager::getMethod ), 
			RFCAST( &KineticManager::setMethod )
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		process,
	};

	// Schedule it to tick 0 stage 0
	static SchedInfo schedInfo[] = { { process, 0, 0 } };
	
	static Cinfo kineticManagerCinfo(
		"KineticManager",
		"Upinder S. Bhalla, 2007, NCBS",
		"Kinetic Manager: Handles integration methods for kinetic simulations. If in 'auto' mode then it picks a method depending on the stochastic and spatial flags. If you set a method, then the 'auto' flag goes off and all the other options are set directly by your choice.",
		initNeutralCinfo(),
		kineticManagerFinfos,
		sizeof( kineticManagerFinfos )/sizeof(Finfo *),
		ValueFtype1< KineticManager >::global(),
			schedInfo, 1
	);

 // static void addMethod( name, description,
 // 					isStochastic,isSpatial, 
 // 					isVariableDt, isImplicit,
 //						isSingleParticle, isMultiscale );

	KineticManager::addMethod( "ee", 
		"GENESIS Exponential Euler method.",
		0, 0, 0, 0, 0, 0 );
	KineticManager::addMethod( "rk2", 
		"Runge Kutta 2, 3 from GSL",
		0, 0, 0, 0, 0, 0 );
	KineticManager::addMethod( "rk4", 
		"Runge Kutta 4th order (classical) from GSL",
		0, 0, 0, 0, 0, 0 );
	KineticManager::addMethod( "rk5", 
		"Embedded Runge-Kutta-Fehlberg (4, 5) method, variable timestep. Default kinetic method from GSL",
		0, 0, 1, 0, 0, 0 );
	KineticManager::addMethod( "rkck", 
		"Embedded Runge-Kutta Cash-Karp (4, 5) from GSL",
		0, 0, 1, 0, 0, 0 );
	KineticManager::addMethod( "rk8pd", 
		"Embedded Runge-Kutta Prince-Dormand (8,9) from GSL",
		0, 0, 1, 0, 0, 0 );
	KineticManager::addMethod( "rk2imp", 
		"Implicit 2nd order Runge-Kutta at Gaussian points from GSL",
		0, 0, 1, 1, 0, 0 );
	KineticManager::addMethod( "rk4imp", 
		"Implicit 4nd order Runge-Kutta at Gaussian points from GSL",
		0, 0, 1, 1, 0, 0 );
	KineticManager::addMethod( "bsimp", 
		"Implicit Bulirsch-Stoer method of Bader and Deuflhard: Not yet implemented as it needs the Jacobian. From GSL",
		0, 0, 1, 1, 0, 0 );
	KineticManager::addMethod( "gear1", 
		"Implicit M = 1 Gear method. From GSL",
		0, 0, 1, 1, 0, 0 );
	KineticManager::addMethod( "gear2", 
		"Implicit M = 2 Gear method. From GSL",
		0, 0, 1, 1, 0, 0 );
	KineticManager::addMethod( "Gillespie1", 
		"Optimized Direct Reaction Method of Gillespie",
		1, 0, 1, 0, 0, 0 );
	KineticManager::addMethod( "Gillespie2", 
		"Optimized Next Reaction Method of Gillespie",
		1, 0, 1, 0, 0, 0 );
	KineticManager::addMethod( "Gibson-Bruck", 
		"Gibson-Bruck optimzed version of Gillespie's next reaction method",
		1, 0, 1, 0, 0, 0 );
	KineticManager::addMethod( "tau_leap", 
		"Gillespie's tau-leap method",
		1, 0, 1, 0, 0, 0 );
	KineticManager::addMethod( "adstoch", 
		"Explicit adaptive stochastic method of Vasudeva and Bhalla",
		1, 0, 1, 0, 0, 1 );
	KineticManager::addMethod( "Smoldyn", 
		"Smoldyn numerical engine for single particle stochastic calculations using Smoluchowski dynamics, from Andrews and Bray",
		1, 1, 0, 0, 1, 0 );
	KineticManager::addMethod( "Elf3d", 
		"Elf-Ehrenberg spatial Gillespie (square grid) method",
		1, 1, 0, 0, 0, 0 );
	KineticManager::addMethod( "Wils", 
		"Wils spatial Gillespie (finite element grid) method",
		1, 1, 0, 0, 0, 0 );

	return &kineticManagerCinfo;
}

static const Cinfo* kineticManagerCinfo = initKineticManagerCinfo();

static const unsigned int reacSlot =
	initKineticManagerCinfo()->getSlotIndex( "reac.n" );
static const unsigned int nSlot =
	initKineticManagerCinfo()->getSlotIndex( "nSrc" );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

KineticManager::KineticManager()
	:
	auto_( 1 ), 
	stochastic_( 0 ),
	spatial_( 0 ),
	method_( "rk5" ),
	implicit_( 0 ),
	variableDt_( 1 ),
	multiscale_( 0 ),
	singleParticle_( 0 )
{
		;
}

void KineticManager::addMethod( const string& name,
	const string& description,
	bool isStochastic,
	bool isSpatial, bool isVariableDt, bool isImplicit,
	bool isSingleParticle, bool isMultiscale )
{
	MethodInfo mi;
	mi.description = description;
	mi.isStochastic = isStochastic;
	mi.isSpatial = isSpatial;
	mi.isVariableDt = isVariableDt;
	mi.isImplicit = isImplicit;
	mi.isSingleParticle = isSingleParticle;
	mi.isMultiscale = isMultiscale;
	methodMap_[name] = mi;
}


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void KineticManager::setAuto( const Conn& c, bool value )
{
	static_cast< KineticManager* >( c.data() )->auto_ = value;
}

bool KineticManager::getAuto( const Element* e )
{
	return static_cast< KineticManager* >( e->data() )->auto_;
}

void KineticManager::setStochastic( const Conn& c, bool value )
{
	static_cast< KineticManager* >( c.data() )->stochastic_ = value;
}

bool KineticManager::getStochastic( const Element* e )
{
	return static_cast< KineticManager* >( e->data() )->stochastic_;
}

void KineticManager::setSpatial( const Conn& c, bool value )
{
	static_cast< KineticManager* >( c.data() )->spatial_ = value;
}

bool KineticManager::getSpatial( const Element* e )
{
	return static_cast< KineticManager* >( e->data() )->spatial_;
}

void KineticManager::setMethod( const Conn& c, string value )
{
	Element* e = c.targetElement();

	static_cast< KineticManager* >( e->data() )->innerSetMethod( e, value );
}

string KineticManager::getMethod( const Element* e )
{
	return static_cast< KineticManager* >( e->data() )->method_;
}

void KineticManager::innerSetMethod( Element* e, string value )
{
	map< string, MethodInfo >::iterator i = methodMap_.find( value );
	if ( i != methodMap_.end() ) {
		method_ = value;
		stochastic_ = i->second.isStochastic;
		spatial_ = i->second.isSpatial;
		variableDt_ = i->second.isVariableDt;
		multiscale_ = i->second.isMultiscale;
		singleParticle_ = i->second.isSingleParticle;
		implicit_ = i->second.isImplicit;
		auto_ = 0;
		setupSolver( e );
	} else {
		if ( stochastic_ )
			method_ = "Gillespie1";
		else
			method_ = "rk5";
		cout << "Warning: method '" << value << "' not known. Using '" <<
			method_ << "'\n";
		innerSetMethod( e, method_ );
		auto_ = 1;
	}
}

// Returns the solver set up for GSL integration, on the element e
Id gslSetup( Element* e, const string& method )
{
	Id solveId;
	if ( lookupGet< Id, string >( e, "lookupChild", solveId, "solve" ) ) {
		if ( solveId.good() ) {
			Id gslId;
			if ( lookupGet< Id, string >( 
				solveId(), "lookupChild", gslId, "GslIntegrator" ) ) {
				if ( gslId.good() )
					return solveId;
				else
					set( solveId(), "destroy" );
			} else {
				// have to clear out the old solver and make a new one.
				set( solveId(), "destroy" );
			}
		}
	}
	Element* solve = Neutral::create( "Neutral", "solve", e );
	solveId = e->id();
	assert( solveId.good() );

	Element*  ki = Neutral::create( "GslIntegrator", "integ", solve );
	assert ( ki != 0 );
	Element* ks = Neutral::create( "Stoich", "stoich", solve );
	assert( ks != 0 );
	Element* kh = Neutral::create( "KineticHub", "hub", solve );
	assert( kh != 0 );
	ks->findFinfo( "hub" )->add( ks, kh, kh->findFinfo( "hub" ) );
	ks->findFinfo( "gsl" )->add( ks, ki, ki->findFinfo( "gsl" ) );
	set< string >( ki, "method", method );
	string simpath = e->id().path() + "/##";
	set< string >( ks, "path", simpath );
	set< double >( ki, "relativeAccuracy", 1.0e-5 );
	set< double >( ki, "absoluteAccuracy", 1.0e-5 );
	return solveId;
}

void eeSetup( Element* e )
{
	cout << "doing ee setup\n";
}

Id gillespieSetup( Element* e, const string& method )
{
	cout << "doing Gillespie setup\n";
	return Id();
}

/**
 * This function figures out an appropriate dt for a fixed timestep 
 * method. It does so by estimating the permissible ( default 1%) error 
 * assuming a forward Euler advance.
 * \todo Currently just a dummy function.
 */
double KineticManager::estimateDt( Element* e ) 
{
	return 0.01;	
}

void KineticManager::setupSolver( Element* e )
{
	if ( method_ == "ee" ) {
		Id solveId;
		if ( lookupGet< Id, string >( e, "lookupChild", solveId, "solve" )){
			if ( solveId.good() ) {
				set( solveId(), "destroy" );
			}
		}
	} else if ( stochastic_ == 0 && multiscale_ == 0 ) {
		// Use a GSL deterministic method.
		Id solveId = gslSetup( e, method_ );
	} else if ( stochastic_ == 1 && multiscale_ == 0 && singleParticle_ == 0 ) {
		Id solveId = gillespieSetup( e, method_ );
	}
}

/**
 * This function sets up dts to use depending on method. Where possible
 * (i.e., variable timestep methods) we would like to use the longest
 * possible dt, which is the rate of
 * graphing. But there are complications yet to be sorted out, for the
 * case of external tables.
 */
void KineticManager::setupDt( Element* e )
{
	static char* fixedDtMethods[] = {
		"ee", 
	};
	static unsigned int numFixedDtMethods = 
		sizeof( fixedDtMethods ) / sizeof( char* );

	Id cj( "/sched/cj" );
	Id t0( "/sched/cj/t0" );
	Id t1( "/sched/cj/t1" );
	Id t2( "/sched/cj/t2" );
	assert( cj.good() );
	assert( t0.good() );
	assert( t1.good() );

	for ( unsigned int i = 0; i < numFixedDtMethods; i++ ) {
		if ( method_ == fixedDtMethods[i] ) {
			double dt = estimateDt( e );
			set< double >( t0(), "dt", dt );
			set< double >( t1(), "dt", dt );
			return;
		}
	}

	double dt = 1.0;
	if ( t2.good() ) {
		get< double >( t2(), "dt", dt );
	}
	set< double >( t0(), "dt", dt );
	set< double >( t1(), "dt", dt );
	set( cj(), "resched" );
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////


 
/**
 * Reinit Function makes sure that all child elements are scheduled,
 * either directly from the clock ticks, or through a solver.
 */

void KineticManager::reinitFunc( const Conn& c, ProcInfo info )
{
	static_cast< KineticManager* >( c.data() )->reinitFuncLocal( 
					c.targetElement() );
}

void KineticManager::reinitFuncLocal( Element* e )
{
	setupSolver( e );
	setupDt( e );
}

/**
 * processFunc doesn't do anything.
 */
void KineticManager::processFunc( const Conn& c, ProcInfo info )
{
	;
//	Element* e = c.targetElement();
//static_cast< KineticManager* >( e->data() )->processFuncLocal( e, info );

}

#ifdef DO_UNIT_TESTS
/*
#include "../element/Neutral.h"
#include "Reaction.h"

void testKineticManager()
{
	cout << "\nTesting KineticManager" << flush;

	Element* n = Neutral::create( "Neutral", "n", Element::root() );
	Element* m0 = Neutral::create( "KineticManager", "m0", n );
	ASSERT( m0 != 0, "creating kineticManager" );
	Element* m1 = Neutral::create( "KineticManager", "m1", n );
	ASSERT( m1 != 0, "creating kineticManager" );
	Element* r0 = Neutral::create( "Reaction", "r0", n );
	ASSERT( r0 != 0, "creating reaction" );

	bool ret;

	ProcInfoBase p;
	Conn cm0( m0, 0 );
	Conn cm1( m1, 0 );
	Conn cr0( r0, 0 );
	p.dt_ = 0.001;
	set< double >( m0, "concInit", 1.0 );
	set< int >( m0, "mode", 0 );
	set< double >( m1, "concInit", 0.0 );
	set< int >( m1, "mode", 0 );
	set< double >( r0, "kf", 0.1 );
	set< double >( r0, "kb", 0.1 );
	ret = m0->findFinfo( "reac" )->add( m0, r0, r0->findFinfo( "sub" ) );
	ASSERT( ret, "adding msg 0" );
	ret = m1->findFinfo( "reac" )->add( m1, r0, r0->findFinfo( "prd" ) );
	ASSERT( ret, "adding msg 1" );

	// First, test charging curve for a single compartment
	// We want our charging curve to be a nice simple exponential
	// n = 0.5 + 0.5 * exp( - t * 0.2 );
	double delta = 0.0;
	double n0 = 1.0;
	double n1 = 0.0;
	double y = 0.0;
	double y0 = 0.0;
	double y1 = 0.0;
	double tau = 5.0;
	double nMax = 0.5;
	Reaction::reinitFunc( cr0, &p );
	KineticManager::reinitFunc( cm0, &p );
	KineticManager::reinitFunc( cm1, &p );
	for ( p.currTime_ = 0.0; p.currTime_ < 20.0; p.currTime_ += p.dt_ ) 
	{
		n0 = KineticManager::getN( m0 );
		n1 = KineticManager::getN( m1 );
//		cout << p.currTime_ << "	" << n1 << endl;

		y = nMax * exp( -p.currTime_ / tau );
		y0 = 0.5 + y;
		y1 = 0.5 - y;
		delta += ( n0 - y0 ) * ( n0 - y0 );
		delta += ( n1 - y1 ) * ( n1 - y1 );
		Reaction::processFunc( cr0, &p );
		KineticManager::processFunc( cm0, &p );
		KineticManager::processFunc( cm1, &p );
	}
	ASSERT( delta < 5.0e-6, "Testing kineticManager and reacn" );

	// Get rid of all the compartments.
	set( n, "destroy" );
}
*/
#endif
