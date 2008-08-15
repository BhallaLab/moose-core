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
#include "../element/Wildcard.h"
#include <math.h>
#include "KineticManager.h"
#include "../utility/utility.h"

static map< string, KMethodInfo >& methodMap()
{
	static map< string, KMethodInfo > methodMap_;

	return methodMap_;
}


static map< string, KMethodInfo >& fillMethodMap()
{
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
	KineticManager::addMethod( "smoldyn", 
		"Smoldyn numerical engine for single particle stochastic calculations using Smoluchowski dynamics, from Andrews and Bray",
		1, 1, 0, 0, 1, 0 );
	KineticManager::addMethod( "Smoldyn", 
		"Smoldyn numerical engine for single particle stochastic calculations using Smoluchowski dynamics, from Andrews and Bray",
		1, 1, 0, 0, 1, 0 );
	KineticManager::addMethod( "Elf3d", 
		"Elf-Ehrenberg spatial Gillespie (square grid) method",
		1, 1, 0, 0, 0, 0 );
	KineticManager::addMethod( "Wils", 
		"Wils spatial Gillespie (finite element grid) method",
		1, 1, 0, 0, 0, 0 );

	return methodMap();
}

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
		new ValueFinfo( "variableDt", 
			ValueFtype1< bool >::global(),
			GFCAST( &KineticManager::getVariableDt ), 
			&dummyFunc
		),
		new ValueFinfo( "singleParticle", 
			ValueFtype1< bool >::global(),
			GFCAST( &KineticManager::getSingleParticle ), 
			&dummyFunc
		),
		new ValueFinfo( "multiscale", 
			ValueFtype1< bool >::global(),
			GFCAST( &KineticManager::getMultiscale ), 
			&dummyFunc
		),
		new ValueFinfo( "implicit", 
			ValueFtype1< bool >::global(),
			GFCAST( &KineticManager::getImplicit ), 
			&dummyFunc
		),
		new ValueFinfo( "description", 
			ValueFtype1< string >::global(),
			GFCAST( &KineticManager::getDescription ), 
			&dummyFunc
		),
		new ValueFinfo( "recommendedDt", 
			ValueFtype1< double >::global(),
			GFCAST( &KineticManager::getRecommendedDt ), 
			&dummyFunc
		),
		new ValueFinfo( "eulerError", 
			ValueFtype1< double >::global(),
			GFCAST( &KineticManager::getEulerError ), 
			RFCAST( &KineticManager::setEulerError )
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "resched", Ftype0::global(),
			RFCAST( &KineticManager::reschedFunc )
		),
	
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		process,
	};

	// Schedule it to tick 1 stage 0
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

	// methodMap.size(); // dummy function to keep compiler happy.

	return &kineticManagerCinfo;
}

static map< string, KMethodInfo >& extMethodMap = fillMethodMap();

static const Cinfo* kineticManagerCinfo = initKineticManagerCinfo();

static const Slot reacSlot = initKineticManagerCinfo()->getSlot( "reac.n" );
static const Slot nSlot = initKineticManagerCinfo()->getSlot( "nSrc" );

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
	singleParticle_( 0 ),
	recommendedDt_( 0.001 ),
	eulerError_( 0.01 )
{
		;
}

void KineticManager::addMethod( const char* name,
	const char* description,
	bool isStochastic,
	bool isSpatial, bool isVariableDt, bool isImplicit,
	bool isSingleParticle, bool isMultiscale )
{
	KMethodInfo mi( isStochastic, isSpatial, isVariableDt, isImplicit,
		isSingleParticle, isMultiscale, string( description ) );
	/*
	mi.isStochastic = isStochastic;
	mi.isSpatial = isSpatial;
	mi.isVariableDt = isVariableDt;
	mi.isImplicit = isImplicit;
	mi.isSingleParticle = isSingleParticle;
	mi.isMultiscale = isMultiscale;
	mi.description = description;
	*/
	string temp( name );
	methodMap()[temp] = mi;
}


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void KineticManager::setAuto( const Conn* c, bool value )
{
	static_cast< KineticManager* >( c->data() )->auto_ = value;
}

bool KineticManager::getAuto( Eref e )
{
	return static_cast< KineticManager* >( e.data() )->auto_;
}

void KineticManager::setStochastic( const Conn* c, bool value )
{
	static_cast< KineticManager* >( c->data() )->stochastic_ = value;
}

bool KineticManager::getStochastic( Eref e )
{
	return static_cast< KineticManager* >( e.data() )->stochastic_;
}

void KineticManager::setSpatial( const Conn* c, bool value )
{
	static_cast< KineticManager* >( c->data() )->spatial_ = value;
}

bool KineticManager::getSpatial( Eref e )
{
	return static_cast< KineticManager* >( e.data() )->spatial_;
}

void KineticManager::setMethod( const Conn* c, string value )
{
	static_cast< KineticManager* >( c->data() )->
		innerSetMethod( c->target(), value );
}

string KineticManager::getMethod( Eref e )
{
	return static_cast< KineticManager* >( e.data() )->method_;
}

bool KineticManager::getVariableDt( Eref e )
{
	return static_cast< KineticManager* >( e.data() )->variableDt_;
}

bool KineticManager::getSingleParticle( Eref e )
{
	return static_cast< KineticManager* >( e.data() )->singleParticle_;
}

bool KineticManager::getMultiscale( Eref e )
{
	return static_cast< KineticManager* >( e.data() )->multiscale_;
}

bool KineticManager::getImplicit( Eref e )
{
	return static_cast< KineticManager* >( e.data() )->implicit_;
}

string KineticManager::getDescription( Eref e )
{
	return static_cast< KineticManager* >( e.data() )->description_;
}

double KineticManager::getRecommendedDt( Eref e )
{
	return static_cast< KineticManager* >( e.data() )->recommendedDt_;
}

void KineticManager::setEulerError( const Conn* c, double value )
{
	static_cast< KineticManager* >( c->data() )->eulerError_ = value;
}

double KineticManager::getEulerError( Eref e )
{
	return static_cast< KineticManager* >( e.data() )->eulerError_;
}

//////////////////////////////////////////////////////////////////
// Here we set up some of the messier inner functions.
//////////////////////////////////////////////////////////////////

void KineticManager::innerSetMethod( Eref e, string value )
{
	map< string, KMethodInfo >::iterator i = methodMap().find( value );
	if ( i != methodMap().end() ) {
		method_ = value;
		stochastic_ = i->second.isStochastic;
		spatial_ = i->second.isSpatial;
		variableDt_ = i->second.isVariableDt;
		multiscale_ = i->second.isMultiscale;
		singleParticle_ = i->second.isSingleParticle;
		implicit_ = i->second.isImplicit;
		description_ = i->second.description;
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
Id gslSetup( Eref e, const string& method )
{
	if ( Cinfo::find( "GslIntegrator" ) == 0 ) // No GSL defined
		return Id();
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
	Element* solve = Neutral::create( "Neutral", "solve", e.id(),
		Id::scratchId() );
	solveId = solve->id();
	assert( solveId.good() );

	Element*  ki = Neutral::create( "GslIntegrator", "integ", solve->id(),
		Id::scratchId() );
	assert ( ki != 0 );
	Element* ks = Neutral::create( "Stoich", "stoich", solve->id(),
		Id::scratchId() );
	assert( ks != 0 );
	Element* kh = Neutral::create( "KineticHub", "hub", solve->id(),
		Id::scratchId() );
	assert( kh != 0 );

	// ks->findFinfo( "hub" )->add( ks, kh, kh->findFinfo( "hub" ) );
	// ks->findFinfo( "gsl" )->add( ks, ki, ki->findFinfo( "gsl" ) );
	Eref( ks ).add( "hub", kh, "hub" );
	Eref( ks ).add( "gsl", ki, "gsl" );

	set< string >( ki, "method", method );
	string simpath = e.id().path() + "/##";
	set< string >( ks, "path", simpath );
	set< double >( ki, "relativeAccuracy", 1.0e-5 );
	set< double >( ki, "absoluteAccuracy", 1.0e-5 );
	return solveId;
}

void eeSetup( Eref e )
{
	cout << "doing ee setup\n";
}

Id gillespieSetup( Eref e, const string& method )
{
	cout << "doing Gillespie setup\n";
	return Id();
}

// Returns the solver set up for GSL integration, on the element e
Id smoldynSetup( Eref e, const string& method, double recommendedDt )
{
	if ( Cinfo::find( "SmoldynHub" ) == 0 ) // No Smoldyn defined
		return Id();
	Id solveId;
	if ( lookupGet< Id, string >( e, "lookupChild", solveId, "solve" ) ) {
		if ( solveId.good() ) {
			Id smoldynId;
			if ( lookupGet< Id, string >( 
				solveId(), "lookupChild", smoldynId, "SmoldynHub" ) ) {
				if ( smoldynId.good() )
					return solveId;
				else
					set( solveId(), "destroy" );
			} else {
				// have to clear out the old solver and make a new one.
				set( solveId(), "destroy" );
			}
		}
	}
	Element* solve = Neutral::create( "Neutral", "solve", e.id(),
		Id::scratchId() );
	solveId = e.id();
	assert( solveId.good() );

	Element* ks = Neutral::create( "Stoich", "stoich", solve->id(),
		Id::scratchId() );
	assert( ks != 0 );
	Element*  sh = Neutral::create( "SmoldynHub", "SmoldynHub", solve->id(),
		Id::scratchId() );
	assert ( sh != 0 );
	set< double >( sh, "dt", recommendedDt );
	set< bool >( ks, "useOneWayReacs", 1 );
	Eref( ks ).add( "hub", sh, "hub" );
	// ks->findFinfo( "hub" )->add( ks, sh, sh->findFinfo( "hub" ) );
	string simpath = e.id().path() + "/##";
	set< string >( ks, "path", simpath );

	// This sets up additional things like the geometry information.
	set< string >( sh, "path", simpath );
	return solveId;
}

void KineticManager::setupSolver( Eref e )
{

	Id solveId;
	if ( method_ == "ee" ) { // Handle default and fallback case.
		if ( lookupGet< Id, string >( e, "lookupChild", solveId, "solve" )){
			if ( solveId.good() ) {
				set( solveId(), "destroy" );
			}
		}
		return;
	} 

	if ( stochastic_ == 0 && multiscale_ == 0 ) {
		// Use a GSL deterministic method.
		solveId = gslSetup( e, method_ );
	} else if ( stochastic_ == 1 && multiscale_ == 0 && singleParticle_ == 0 ) {
		solveId = gillespieSetup( e, method_ );
	} else if ( stochastic_ == 1 && multiscale_ == 0 && singleParticle_ == 1 ) {
		solveId = smoldynSetup( e, method_, recommendedDt_ );
	} 

	// method failed.
	if ( !solveId.good() ) {
		cout << "Warning: Unable to set up method " << method_ << 
			". Using Exp Euler (ee)\n";
		method_ = "ee";
		setupSolver( e );
	}
}

/**
 * This function sets up dts to use depending on method. Where possible
 * (i.e., variable timestep methods) we would like to use the longest
 * possible dt, which is the rate of
 * graphing. But there are complications yet to be sorted out, for the
 * case of external tables.
 */
void KineticManager::setupDt( Eref e, double dt )
{
	static const char* fixedDtMethods[] = {
		"ee", 
		"Smoldyn", 
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

	/*
	Element* elm;
	string field;
	double dt = estimateDt( e, &elm, field, EULER_ACCURACY );
	*/

	for ( unsigned int i = 0; i < numFixedDtMethods; i++ ) {
		if ( method_ == fixedDtMethods[i] ) {
			set< double >( t0(), "dt", dt );
			set< double >( t1(), "dt", dt );
			return;
		}
	}

	Id integ( e.id().path() + "/solve/integ" );
	assert( integ.good() );
	set< double >( integ(), "internalDt", dt );

	double plotDt = 1.0;
	if ( t2.good() ) {
		get< double >( t2(), "dt", plotDt );
	}
	set< double >( t0(), "dt", plotDt );
	set< double >( t1(), "dt", plotDt );
	set( cj(), "resched" );
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////


 
/**
 * Reinit Function restarts the simulation from time 0.
 */

void KineticManager::reinitFunc( const Conn* c, ProcInfo info )
{
	static_cast< KineticManager* >( c->data() )->reinitFuncLocal( 
					c->target() );
}

void KineticManager::reinitFuncLocal( Eref e )
{
	;
}
 
/**
 * Resched Function makes sure that all child elements are scheduled,
 * either directly from the clock ticks, or through a solver.
 */
void KineticManager::reschedFunc( const Conn* c )
{
	static_cast< KineticManager* >( c->data() )->reschedFuncLocal( 
					c->target() );
}

void KineticManager::reschedFuncLocal( Eref e )
{
	Id offendingElm;
	string field;
	double dt = estimateDt( e.id(), offendingElm, field, eulerError_ );
	setupSolver( e );
	setupDt( e, dt );
}

/**
 * processFunc doesn't do anything.
 */
void KineticManager::processFunc( const Conn* c, ProcInfo info )
{
	;
//	Element* e = c.targetElement();
//static_cast< KineticManager* >( e.data() )->processFuncLocal( e, info );

}

//////////////////////////////////////////////////////////////////
//
// Here we have a set of utility functions to estimate a suitable
// dt for the simulation. Needed for fixed-timestep methods, but
// also useful to set a starting timestep for variable timestep methods.
//
//////////////////////////////////////////////////////////////////

/**
 * Returns largest propensity for a reaction Element e. Direction of
 * reaction calculation is set by the isPrd flag.
 */
double KineticManager::findReacPropensity( Eref e, bool isPrd ) const
{
	static const Cinfo* rCinfo = Cinfo::find( "Reaction" );
	static const Finfo* substrateFinfo = rCinfo->findFinfo( "sub" );
	static const Finfo* productFinfo = rCinfo->findFinfo( "prd" );

	assert( e.e->cinfo()->isA( rCinfo ) );

	bool ret;
	double prop;
	if ( isPrd )
		ret = get< double >( e, "kb", prop );
	else 
		ret = get< double >( e, "kf", prop );
	assert( ret );
	double min = 1.0e10;
	double mval;

	Conn* c;
	if ( isPrd )
		c = e.e->targets( productFinfo->msg(), e.i );
	else
		c = e.e->targets( substrateFinfo->msg(), e.i );

	while ( c->good() ) {
		Element* m = c->target().e;
		assert( m->cinfo()->isA( Cinfo::find( "Molecule" ) ) );
		ret = get< double >( c->target(), "nInit", mval );
		// should really be 'n' but there is a problem with initialization
		// of the S_ array if we want to be able to do on-the-fly solver
		// rebuilding.
		assert( ret );
		prop *= mval;
		if ( min > mval )
			min = mval;

		c->increment();
	}
	delete c;


/*

	vector< Conn > list;
	vector< Conn >::iterator i;
	if ( isPrd )
		productFinfo->incomingConns( e, list );
	else
		substrateFinfo->incomingConns( e, list );

	for( i = list.begin(); i != list.end(); i++ ) {
		Element* m = i->targetElement();
		assert( m->cinfo()->isA( Cinfo::find( "Molecule" ) ) );
		ret = get< double >( m, "nInit", mval );
		// should really be 'n' but there is a problem with initialization
		// of the S_ array if we want to be able to do on-the-fly solver
		// rebuilding.
		assert( ret );
		prop *= mval;
		if ( min > mval )
			min = mval;
	}
*/
	if ( min > 0.0 )
		return prop / min;
	else
		return 0.0;
}

/**
 * Returns largest propensity for a reaction Element e. Direction of
 * reaction calculation is set by the isPrd flag.
 */
double KineticManager::findEnzSubPropensity( Eref e ) const
{
	static const Cinfo* eCinfo = Cinfo::find( "Enzyme" );
	static const Cinfo* mCinfo = Cinfo::find( "Molecule" );
	static const Finfo* substrateFinfo = eCinfo->findFinfo( "sub" );
	static const Finfo* enzymeFinfo = eCinfo->findFinfo( "enz" );
	static const Finfo* intramolFinfo = eCinfo->findFinfo( "intramol" );
	assert( e.e->cinfo()->name() == "Enzyme" );

	bool ret;
	bool mode;
	ret = get< bool >( e, "mode", mode );
	assert( ret );

	double prop;

	if ( mode ) { // An MM enzyme, implicit form.
		// Here we compute it as rate / nmin.
		double Km;
		double k3;
		ret = get< double >( e, "Km", Km );
		assert( ret );
		ret = get< double >( e, "k3", k3 );
		assert( ret );
		assert( Km > 0 );
		prop = k3 / Km;
	} else {
		ret = get< double >( e, "k1", prop );
		assert( ret );
	}
	double min = 1.0e10;
	double mval;
	Conn* sc = e.e->targets( substrateFinfo->msg(), e.i );
	if ( !sc->good() ) { // A dangling enzyme, no substrates.
		delete sc;
		return 0.0;
	}
	Conn* ec = e.e->targets( enzymeFinfo->msg(), e.i );
	assert( ec->good() );
	ret = get< double >( ec->target(), "nInit", mval );
	prop *= mval;
	assert( ret );
	min = mval;
	delete ec;

	while ( sc->good() ) {
		Eref m = sc->target();
		assert( m.e->cinfo()->isA( mCinfo ) );
		ret = get< double >( m, "nInit", mval );
		assert( ret );
		prop *= mval;
		if ( min > mval )
			min = mval;

		sc->increment();
	}
	delete sc;

	Conn* ic = e.e->targets( intramolFinfo->msg(), e.i );
	while ( ic->good() ) {
		Eref m = ic->target();
		assert( m.e->cinfo()->isA( mCinfo ) );
		ret = get< double >( m, "nInit", mval );
		assert( ret );
		if ( mval > 0.0 )
			prop /= mval;
		ic->increment();
	}
	delete ic;
	
	if ( min > 0.0 )
		return prop / min;
	else
		return 0.0;
}

/*
double KineticManager::findEnzSubPropensity( Element* e ) const
{
	static const Cinfo* eCinfo = Cinfo::find( "Enzyme" );
	static const Finfo* substrateFinfo = eCinfo->findFinfo( "sub" );
	static const Finfo* enzymeFinfo = eCinfo->findFinfo( "enz" );
	static const Finfo* intramolFinfo = eCinfo->findFinfo( "intramol" );

	assert( e->cinfo()->isA( eCinfo ) );

	bool ret;
	bool mode;
	ret = get< bool >( e, "mode", mode );
	assert( ret );

	double prop;

	if ( mode ) { // An MM enzyme, implicit form.
		// Here we compute it as rate / nmin.
		double Km;
		double k3;
		ret = get< double >( e, "Km", Km );
		assert( ret );
		ret = get< double >( e, "k3", k3 );
		assert( ret );
		assert( Km > 0 );
		prop = k3 / Km;
	} else {
		ret = get< double >( e, "k1", prop );
		assert( ret );
	}

	double min = 1.0e10;
	double mval;

	vector< Conn > list;
	vector< Conn > list2;
	vector< Conn >::iterator i;
	substrateFinfo->incomingConns( e, list );
	if ( list.size() == 0 ) // It is possible to have a dangling enzyme
		return 0.0;
	enzymeFinfo->incomingConns( e, list2 );
	assert( list2.size() == 1 );
	list.insert( list.end(), list2.begin(), list2.end() );

	// This is a little fishy for the MM case, because we don't take
	// the (Km + sub) term into account properly for the denominator.
	// The outcome of this is therefore more stringent than it needs
	// to be.
	for( i = list.begin(); i != list.end(); i++ ) {
		Element* m = i->targetElement();
		assert( m->cinfo()->isA( Cinfo::find( "Molecule" ) ) );
		ret = get< double >( m, "nInit", mval );
		assert( ret );
		prop *= mval;
		if ( min > mval )
			min = mval;
	}

	intramolFinfo->incomingConns( e, list );
	for( i = list.begin(); i != list.end(); i++ ) {
		Element* m = i->targetElement();
		assert( m->cinfo()->isA( Cinfo::find( "Molecule" ) ) );
		ret = get< double >( m, "nInit", mval );
		assert( ret );
		if ( mval > 0.0 )
			prop /= mval;
	}
	if ( min > 0.0 )
		return prop / min;
	else
		return 0.0;
}
*/

double KineticManager::findEnzPrdPropensity( Eref e ) const
{
	assert( e.e->cinfo()->isA( Cinfo::find( "Enzyme" ) ) );

	bool ret;
	bool mode;
	ret = get< bool >( e, "mode", mode );
	assert( ret );
	if ( mode ) // MM enzyme, so don't use this term at all
		return 0.0;

	double k2;
	double k3;
	ret = get< double >( e, "k2", k2 );
	assert( ret );
	ret = get< double >( e, "k3", k3 );
	assert( ret );
	return k2 + k3;
}

/**
 * This function figures out an appropriate dt for a fixed timestep 
 * method. It does so by estimating the permissible ( default 1%) error 
 * assuming a forward Euler advance.
 * Also returns the element and field that have the highest propensity,
 * that is, the shortest dt.
 * As a side-effect it assigns the KineticManager::recommendedDt_ to the
 * calculated dt value.
 */
#ifndef NDEBUG
#include <limits>
#endif

double KineticManager::estimateDt( Id mgr, 
	Id& elm, string& field, double error ) 
{
	static const Cinfo* eCinfo = Cinfo::find( "Enzyme" );
	static const Cinfo* rCinfo = Cinfo::find( "Reaction" );

	assert( error > 0.0 );
	vector< Id > elist;
	vector< Id >::iterator i;

	// int allChildren( start, insideBrace, index, ret)
	if ( allChildren( mgr, "", Id::AnyIndex, elist ) == 0 ) {
		elm = Id::badId();
		field = "";
		return 1.0;
	}

	double prop = 0.0;
	double maxProp = 0.0;

	for ( i = elist.begin(); i != elist.end(); i++ )
	{
		if ( ( *i )()->cinfo()->isA( rCinfo ) ) {
			prop = findReacPropensity( ( *i )(), 0 );
			if ( maxProp < prop ) {
				maxProp = prop;
				elm = *i;
				field = "kf";
			}
			prop = findReacPropensity( ( *i )(), 1 );
			if ( maxProp < prop ) {
				maxProp = prop;
				elm = *i;
				field = "kb";
			}
		} else if ( ( *i )()->cinfo()->isA( eCinfo ) ) {
			prop = findEnzSubPropensity( ( *i )() );
			if ( maxProp < prop ) {
				maxProp = prop;
				elm = *i;
				field = "k1";
			}
			prop = findEnzPrdPropensity( ( *i )() );
			if ( maxProp < prop ) {
				maxProp = prop;
				elm = *i;
				field = "k3";
			}
		}
	}
	assert ( error < 1.0 );
	if ( maxProp <= 0 ) 
		maxProp = 10.0;

	recommendedDt_ = sqrt( error ) / maxProp;
	assert ( !isinf( recommendedDt_ ) );

	return recommendedDt_;
}
