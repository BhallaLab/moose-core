/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include "RateTerm.h"
#include "KinSparseMatrix.h"
#include "InterSolverFlux.h"
#include "Stoich.h"
#include "GslIntegrator.h"

const Cinfo* initGslIntegratorCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process",
			Ftype1< ProcInfo >::global(),
			RFCAST( &GslIntegrator::processFunc )),
		new DestFinfo( "reinit",
			Ftype1< ProcInfo >::global(),
			RFCAST( &GslIntegrator::reinitFunc )),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ) );

	static Finfo* gslShared[] =
	{
		new SrcFinfo( "reinitSrc", Ftype0::global() ),
		new DestFinfo( "assignStoich",
			Ftype1< void* >::global(),
			RFCAST( &GslIntegrator::assignStoichFunc )
			),
		new DestFinfo( "setMolN",
			Ftype2< double, unsigned int >::global(),
			RFCAST( &GslIntegrator::setMolN )
			),
		new SrcFinfo( "requestYsrc", Ftype0::global() ),
		new DestFinfo( "assignY",
			Ftype1< double* >::global(),
			RFCAST( &GslIntegrator::assignY )
			),
	};

	static Finfo* gslIntegratorFinfos[] =
	{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		new ValueFinfo( "isInitiatilized", 
			ValueFtype1< bool >::global(),
			GFCAST( &GslIntegrator::getIsInitialized ), 
			&dummyFunc
		),
		new ValueFinfo( "method", 
			ValueFtype1< string >::global(),
			GFCAST( &GslIntegrator::getMethod ), 
			RFCAST( &GslIntegrator::setMethod )
		),
		new ValueFinfo( "relativeAccuracy", 
			ValueFtype1< double >::global(),
			GFCAST( &GslIntegrator::getRelativeAccuracy ), 
			RFCAST( &GslIntegrator::setRelativeAccuracy )
		),
		new ValueFinfo( "absoluteAccuracy", 
			ValueFtype1< double >::global(),
			GFCAST( &GslIntegrator::getAbsoluteAccuracy ), 
			RFCAST( &GslIntegrator::setAbsoluteAccuracy )
		),
		new ValueFinfo( "internalDt", 
			ValueFtype1< double >::global(),
			GFCAST( &GslIntegrator::getInternalDt ), 
			RFCAST( &GslIntegrator::setInternalDt )
		),

		///////////////////////////////////////////////////////
		// DestFinfo definitions
		///////////////////////////////////////////////////////
		new DestFinfo( "assignStoich",
			Ftype1< void* >::global(),
			RFCAST( &GslIntegrator::assignStoichFunc )
			),
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		new SharedFinfo( "gsl", gslShared, 
				sizeof( gslShared )/ sizeof( Finfo* ),
					"This is connected to the Stoich object." ),
		process,
	};

	static SchedInfo schedInfo[] = { { process, 0, 0 } };

	static string doc[] =
	{
		"Name", "GslIntegrator",
		"Author", "Upinder S. Bhalla, June 2006, NCBS",
		"Description", "GslIntegrator: Integrator class for using the GSL ODE functions to do numerical "
				"integration in the Kinetic Solver set.This is currently set up to work only with "
				"the Stoich class, which represents biochemical networks.The goal is to have a "
				"standard interface so different solvers can work with different kinds of calculation.",
	};	
	
	static  Cinfo gslIntegratorCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		gslIntegratorFinfos,
		sizeof(gslIntegratorFinfos)/sizeof(Finfo *),
		ValueFtype1< GslIntegrator >::global(),
			schedInfo, 1
	);

	return &gslIntegratorCinfo;
}

static const Cinfo* gslIntegratorCinfo = initGslIntegratorCinfo();

/*
static const Slot integrateSlot =
	initGslIntegratorCinfo()->getSlot( "integrateSrc" );
	*/
static const Slot reinitSlot =
	initGslIntegratorCinfo()->getSlot( "gsl.reinitSrc" );
static const Slot requestYslot =
	initGslIntegratorCinfo()->getSlot( "gsl.requestYsrc" );


///////////////////////////////////////////////////
// Basic class function definitions
///////////////////////////////////////////////////

GslIntegrator::GslIntegrator()
{
	isInitialized_ = 0;
	method_ = "rk5";
	gslStepType_ = gsl_odeiv_step_rkf45;
	gslStep_ = 0;
	nVarMols_ = 0;
	absAccuracy_ = 1.0e-6;
	relAccuracy_ = 1.0e-6;
	internalStepSize_ = 1.0e-4;
	y_ = 0;
        gslEvolve_ = NULL;
        gslControl_ = NULL;
        
        
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

bool GslIntegrator::getIsInitialized( Eref e )
{
	return static_cast< const GslIntegrator* >( e.data() )->isInitialized_;
}

string GslIntegrator::getMethod( Eref e )
{
	return static_cast< const GslIntegrator* >( e.data() )->method_;
}
void GslIntegrator::setMethod( const Conn* c, string method )
{
	static_cast< GslIntegrator* >( c->data() )->innerSetMethod( method );
}

void GslIntegrator::innerSetMethod( const string& method )
{
	method_ = method;
	gslStepType_ = 0;
	// cout << "in void GslIntegrator::innerSetMethod( string method ) \n";
	if ( method == "rk2" ) {
		gslStepType_ = gsl_odeiv_step_rk2;
	} else if ( method == "rk4" ) {
		gslStepType_ = gsl_odeiv_step_rk4;
	} else if ( method == "rk5" ) {
		gslStepType_ = gsl_odeiv_step_rkf45;
	} else if ( method == "rkck" ) {
		gslStepType_ = gsl_odeiv_step_rkck;
	} else if ( method == "rk8pd" ) {
		gslStepType_ = gsl_odeiv_step_rk8pd;
	} else if ( method == "rk2imp" ) {
		gslStepType_ = gsl_odeiv_step_rk2imp;
	} else if ( method == "rk4imp" ) {
		gslStepType_ = gsl_odeiv_step_rk4imp;
	} else if ( method == "bsimp" ) {
		gslStepType_ = gsl_odeiv_step_rk4imp;
		cout << "Warning: implicit Bulirsch-Stoer method not yet implemented: needs Jacobian\n";
	} else if ( method == "gear1" ) {
		gslStepType_ = gsl_odeiv_step_gear1;
	} else if ( method == "gear2" ) {
		gslStepType_ = gsl_odeiv_step_gear2;
	} else {
		cout << "Warning: GslIntegrator::innerSetMethod: method '" <<
			method << "' not known, using rk5\n";
		gslStepType_ = gsl_odeiv_step_rkf45;
	}
}


double GslIntegrator::getRelativeAccuracy( Eref e )
{
	return static_cast< const GslIntegrator* >( e.data() )->relAccuracy_;
}
void GslIntegrator::setRelativeAccuracy( const Conn* c, double value )
{
	static_cast< GslIntegrator* >( c->data() )->relAccuracy_ = value;
}

double GslIntegrator::getAbsoluteAccuracy( Eref e )
{
	return static_cast< const GslIntegrator* >( e.data() )->absAccuracy_;
}
void GslIntegrator::setAbsoluteAccuracy( const Conn* c, double value )
{
	static_cast< GslIntegrator* >( c->data() )->absAccuracy_ = value;
}

double GslIntegrator::getInternalDt( Eref e )
{
	return static_cast< const GslIntegrator* >( e.data() )->internalStepSize_;
}
void GslIntegrator::setInternalDt( const Conn* c, double value )
{
	static_cast< GslIntegrator* >( c->data() )->internalStepSize_ = value;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void GslIntegrator::setMolN( const Conn* c, double y, unsigned int i )
{
	static_cast< GslIntegrator* >( c->data() )->
		innerSetMolN( y, i );
}

void GslIntegrator::innerSetMolN( double y, unsigned int i ) 
{
	if ( i >= nVarMols_ ) {
		cout << "Error: GslIntegrator::innerSetMolN: i >=nVarMols: " <<
			i << " >= " << nVarMols_ << endl;
		return;
	}
	y_[i] = y;
}

void GslIntegrator::assignStoichFunc( const Conn* c, void* stoich )
{
	static_cast< GslIntegrator* >( c->data() )->
		assignStoichFuncLocal( stoich );
}

/**
 * This function should also set up the sizes, and it should be at 
 * allocate, not reinit time.
 */
void GslIntegrator::assignStoichFuncLocal( void* stoich ) 
{
	Stoich* s = static_cast< Stoich* >( stoich );
		// memcpy( &S_[0], y, nVarMolsBytes_ );
	// y_ = s->S();
	
	nVarMols_ = s->nVarMols();
	y_ = new double[ nVarMols_ ];
	dynamicBuffers_ = s->dynamicBuffers(); // filthy, just a ptr copy

	memcpy( y_, s->Sinit(), nVarMols_ * sizeof( double ) );
	for ( unsigned int i = 0; i < nVarMols_; ++i )
		assert( !isnan( y_[ i ] ) && y_[i] >= 0.0 );

	isInitialized_ = 1;
        // Allocate GSL functions if not already allocated,
        // otherwise reset the reusable ones
        assert( gslStepType_ != 0 );
        if ( gslStep_ )
        {
            gsl_odeiv_step_free(gslStep_);
        }
        
        gslStep_ = gsl_odeiv_step_alloc( gslStepType_, nVarMols_ );
        
   	assert( gslStep_ != 0 );
        if ( !gslEvolve_ )
        {
            gslEvolve_ = gsl_odeiv_evolve_alloc(nVarMols_);
        }
        else
        {
            gsl_odeiv_evolve_reset(gslEvolve_);
        }
        assert(gslEvolve_ != 0);
        
        if ( !gslControl_ )
        {
            gslControl_ = gsl_odeiv_control_y_new( absAccuracy_, relAccuracy_ );
        }
        else 
        {
            gsl_odeiv_control_init(gslControl_,absAccuracy_, relAccuracy_, 1, 0);
        }
        assert(gslControl_!= 0);
        
        
	gslSys_.function = &Stoich::gslFunc;
	gslSys_.jacobian = 0;
	gslSys_.dimension = nVarMols_;
	gslSys_.params = stoich;
}

/**
 * Copies over the values in S to the y_ vector of the GSL.
 */
void GslIntegrator::assignY( const Conn* c, double* S )
{
	static_cast< GslIntegrator* >( c->data() )->innerAssignY( S );
}

void GslIntegrator::innerAssignY( double* S ) 
{
	 assert( nVarMols_ > 0 );
	 memcpy( y_, S, nVarMols_ * sizeof( double ) );
}

void GslIntegrator::processFunc( const Conn* c, ProcInfo info )
{
	static_cast< GslIntegrator* >( c->data() )->
		innerProcessFunc( c->target(), info );
}

/**
 * Here we want to give the integrator as long a timestep as possible,
 * or alternatively let _it_ decide the timestep. The former is done
 * by providing a long dt, typically that of the graphing process.
 * The latter is harder to manage and works best if there is only this
 * one integrator running the simulation. Here we do the former.
 */
void GslIntegrator::innerProcessFunc( Eref e, ProcInfo info )
{
	double nextt = info->currTime_ + info->dt_;
	double t = info->currTime_;
	while ( t < nextt ) {
		int status = gsl_odeiv_evolve_apply ( 
			gslEvolve_, gslControl_, gslStep_, &gslSys_, 
			&t, nextt,
			&internalStepSize_, y_);
		if ( status != GSL_SUCCESS )
			break;

		// Zero out buffered molecules. Perhaps this can be ignored
		for( vector< unsigned int >::const_iterator 
			i = dynamicBuffers_->begin(); 
			i != dynamicBuffers_->end(); ++i )
			y_[ *i ] = 0.0;
	}
}

void GslIntegrator::reinitFunc( const Conn* c, ProcInfo info )
{
    // Everything is done in assignStoichFuncLocal
	// But the init function should somehow move here.
	send0( c->target(), reinitSlot );
	// y_[] = yprime_[]
}
