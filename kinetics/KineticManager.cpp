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
		new ValueFinfo( "variableDt", 
			ValueFtype1< bool >::global(),
			GFCAST( &KineticManager::getVariableDt ), 
			dummyFunc
		),
		new ValueFinfo( "singleParticle", 
			ValueFtype1< bool >::global(),
			GFCAST( &KineticManager::getSingleParticle ), 
			dummyFunc
		),
		new ValueFinfo( "multiscale", 
			ValueFtype1< bool >::global(),
			GFCAST( &KineticManager::getMultiscale ), 
			dummyFunc
		),
		new ValueFinfo( "implicit", 
			ValueFtype1< bool >::global(),
			GFCAST( &KineticManager::getImplicit ), 
			dummyFunc
		),
		new ValueFinfo( "description", 
			ValueFtype1< string >::global(),
			GFCAST( &KineticManager::getDescription ), 
			dummyFunc
		),
		new ValueFinfo( "recommendedDt", 
			ValueFtype1< double >::global(),
			GFCAST( &KineticManager::getRecommendedDt ), 
			dummyFunc
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

bool KineticManager::getVariableDt( const Element* e )
{
	return static_cast< KineticManager* >( e->data() )->variableDt_;
}

bool KineticManager::getSingleParticle( const Element* e )
{
	return static_cast< KineticManager* >( e->data() )->singleParticle_;
}

bool KineticManager::getMultiscale( const Element* e )
{
	return static_cast< KineticManager* >( e->data() )->multiscale_;
}

bool KineticManager::getImplicit( const Element* e )
{
	return static_cast< KineticManager* >( e->data() )->implicit_;
}

string KineticManager::getDescription( const Element* e )
{
	return static_cast< KineticManager* >( e->data() )->description_;
}

double KineticManager::getRecommendedDt( const Element* e )
{
	return static_cast< KineticManager* >( e->data() )->recommendedDt_;
}


//////////////////////////////////////////////////////////////////
// Here we set up some of the messier inner functions.
//////////////////////////////////////////////////////////////////

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
	static const double EULER_ACCURACY = 0.01; 
		// Actually this is pretty tight for exponential Euler

	Id cj( "/sched/cj" );
	Id t0( "/sched/cj/t0" );
	Id t1( "/sched/cj/t1" );
	Id t2( "/sched/cj/t2" );
	assert( cj.good() );
	assert( t0.good() );
	assert( t1.good() );

	Element* elm;
	string field;
	double dt = estimateDt( e, &elm, field, EULER_ACCURACY );

	for ( unsigned int i = 0; i < numFixedDtMethods; i++ ) {
		if ( method_ == fixedDtMethods[i] ) {
			set< double >( t0(), "dt", dt );
			set< double >( t1(), "dt", dt );
			return;
		}
	}

	Id integ( e->id().path() + "/solve/integ" );
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
double KineticManager::findReacPropensity( Element* e, bool isPrd ) const
{
	static const Cinfo* rCinfo = Cinfo::find( "Reaction" );
	static const Cinfo* mCinfo = Cinfo::find( "Molecule" );
	static const Finfo* substrateFinfo = rCinfo->findFinfo( "sub" );
	static const Finfo* productFinfo = rCinfo->findFinfo( "prd" );

	assert( e->cinfo()->isA( rCinfo ) );

	bool ret;
	double prop;
	if ( isPrd )
		ret = get< double >( e, "kb", prop );
	else 
		ret = get< double >( e, "kf", prop );
	assert( ret );
	double min = 1.0e10;
	double mval;

	vector< Conn > list;
	vector< Conn >::iterator i;
	if ( isPrd )
		productFinfo->incomingConns( e, list );
	else
		substrateFinfo->incomingConns( e, list );

	for( i = list.begin(); i != list.end(); i++ ) {
		Element* m = i->targetElement();
		assert( m->cinfo()->isA( mCinfo ) );
		ret = get< double >( m, "nInit", mval );
		// should really be 'n' but there is a problem with initialization
		// of the S_ array if we want to be able to do on-the-fly solver
		// rebuilding.
		assert( ret );
		prop *= mval;
		if ( min > mval )
			min = mval;
	}
	if ( min > 0.0 )
		return prop / min;
	else
		return 0.0;
}

/**
 * Returns largest propensity for a reaction Element e. Direction of
 * reaction calculation is set by the isPrd flag.
 */
double KineticManager::findEnzSubPropensity( Element* e ) const
{
	static const Cinfo* eCinfo = Cinfo::find( "Enzyme" );
	static const Cinfo* mCinfo = Cinfo::find( "Molecule" );
	static const Finfo* substrateFinfo = eCinfo->findFinfo( "sub" );
	static const Finfo* enzymeFinfo = eCinfo->findFinfo( "enz" );
	static const Finfo* intramolFinfo = eCinfo->findFinfo( "intramol" );

	assert( e->cinfo()->isA( eCinfo ) );

	bool ret;
	double prop;
	ret = get< double >( e, "k1", prop );
	assert( ret );
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

	for( i = list.begin(); i != list.end(); i++ ) {
		Element* m = i->targetElement();
		assert( m->cinfo()->isA( mCinfo ) );
		ret = get< double >( m, "nInit", mval );
		assert( ret );
		prop *= mval;
		if ( min > mval )
			min = mval;
	}

	intramolFinfo->incomingConns( e, list );
	for( i = list.begin(); i != list.end(); i++ ) {
		Element* m = i->targetElement();
		assert( m->cinfo()->isA( mCinfo ) );
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

double KineticManager::findEnzPrdPropensity( Element* e ) const
{
	static const Cinfo* eCinfo = Cinfo::find( "Enzyme" );
	assert( e->cinfo()->isA( eCinfo ) );

	bool ret;
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
 */
double KineticManager::estimateDt( Element* mgr, 
	Element** elm, string& field, double error ) 
{
	static const Cinfo* eCinfo = Cinfo::find( "Enzyme" );
	static const Cinfo* rCinfo = Cinfo::find( "Reaction" );

	assert( error > 0.0 );
	vector< Element* > elist;
	vector< Element* >::iterator i;

	if ( wildcardRelativeFind( mgr, "##", elist, 1 ) == 0 ) {
		*elm = 0;
		field = "";
		return 1.0;
	}

	double prop = 0.0;
	double maxProp = 0.0;

	for ( i = elist.begin(); i != elist.end(); i++ )
	{
		if ( ( *i )->cinfo()->isA( rCinfo ) ) {
			prop = findReacPropensity( *i, 0 );
			if ( maxProp < prop ) {
				maxProp = prop;
				*elm = *i;
				field = "kf";
			}
			prop = findReacPropensity( *i, 1 );
			if ( maxProp < prop ) {
				maxProp = prop;
				*elm = *i;
				field = "kb";
			}
		} else if ( ( *i )->cinfo()->isA( eCinfo ) ) {
			prop = findEnzSubPropensity( *i );
			if ( maxProp < prop ) {
				maxProp = prop;
				*elm = *i;
				field = "k1";
			}
			prop = findEnzPrdPropensity( *i );
			if ( maxProp < prop ) {
				maxProp = prop;
				*elm = *i;
				field = "k3";
			}
		}
	}

	recommendedDt_ = sqrt( error ) / maxProp;

	return recommendedDt_;
}
