/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include "GslIntegrator.h"
#include "../shell/Shell.h"

const Cinfo* GslIntegrator::initCinfo()
{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		static ReadOnlyValueFinfo< GslIntegrator, bool > isInitialized( 
			"isInitialized", 
			"True if the Stoich message has come in to set parms",
			&GslIntegrator::getIsInitialized
		);
		static ValueFinfo< GslIntegrator, string > method( "method", 
			"Numerical method to use.",
			&GslIntegrator::setMethod,
			&GslIntegrator::getMethod 
		);
		static ValueFinfo< GslIntegrator, double > relativeAccuracy( 
			"relativeAccuracy", 
			"Accuracy criterion",
			&GslIntegrator::setRelativeAccuracy,
			&GslIntegrator::getRelativeAccuracy
		);
		static ValueFinfo< GslIntegrator, double > absoluteAccuracy( 
			"absoluteAccuracy", 
			"Another accuracy criterion",
			&GslIntegrator::setAbsoluteAccuracy,
			&GslIntegrator::getAbsoluteAccuracy
		);
		static ValueFinfo< GslIntegrator, double > internalDt( 
			"internalDt", 
			"internal timestep to use.",
			&GslIntegrator::setInternalDt,
			&GslIntegrator::getInternalDt
		);

		///////////////////////////////////////////////////////
		// DestFinfo definitions
		///////////////////////////////////////////////////////
		static DestFinfo stoich( "stoich",
			"Handle data from Stoich",
			new EpFunc1< GslIntegrator, Id >( &GslIntegrator::stoich )
		);

		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< GslIntegrator >( &GslIntegrator::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< GslIntegrator >( &GslIntegrator::reinit ) );

		static DestFinfo remesh( "remesh",
			"Handle commands to remesh the pool. This may involve changing "
			"the number of pool entries, as well as changing their volumes",
			new EpFunc4< GslIntegrator, unsigned int, unsigned int, vector< unsigned int >, vector< double > >( &GslIntegrator::remesh )
		);
		
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* gslIntegratorFinfos[] =
	{
		&isInitialized,		// Value
		&method,			// Value
		&relativeAccuracy,	// Value
		&absoluteAccuracy,	// Value
		&stoich,			// DestFinfo
		&remesh,			// DestFinfo
		&proc,				// SharedFinfo
	};
	
	static  Cinfo gslIntegratorCinfo(
		"GslIntegrator",
		Neutral::initCinfo(),
		gslIntegratorFinfos,
		sizeof(gslIntegratorFinfos)/sizeof(Finfo *),
		new Dinfo< GslIntegrator >
	);

	return &gslIntegratorCinfo;
}

static const Cinfo* gslIntegratorCinfo = GslIntegrator::initCinfo();

///////////////////////////////////////////////////
// Basic class function definitions
///////////////////////////////////////////////////

GslIntegrator::GslIntegrator()
	: 
	isInitialized_( 0 ),
	method_( "rk5" ),
	absAccuracy_( 1.0e-9 ),
	relAccuracy_( 1.0e-6 ),
	internalStepSize_( 1.0 ),
	y_( 0 ),
	nVarPools_( 0 ),
	gslStepType_( 0 ), 
	gslStep_( 0 ), 
	gslControl_( 0 ), 
	gslEvolve_( 0 )
{
#ifdef USE_GSL
	gslStepType_ = gsl_odeiv_step_rkf45;
	gslStep_ = 0;
#endif // USE_GSL
}

/**
 * Needed for the Dinfo::assign function, to ensure we initialize the 
 * gsl pointers correctly. Instead of trying to guess the Element indices,
 * we zero out the pointers (not free them) so that the system has to
 * do the initialization in a separate call to GslIntegrator::stoich().
 */
GslIntegrator& GslIntegrator::operator=( const GslIntegrator& other )
{
	isInitialized_ = 0;
	method_ = "rk5";
	absAccuracy_ = 1.0e-9;
	relAccuracy_ = 1.0e-6;
	internalStepSize_ = 1.0;
	y_ = 0;
	nVarPools_ = 0;
	gslStepType_ = gsl_odeiv_step_rkf45;
	gslStep_ = 0;
	gslControl_ = 0;
	gslEvolve_ = 0;
	stoichId_ = Id();

	return *this;
}

GslIntegrator::~GslIntegrator()
{
	if ( gslEvolve_ )
		gsl_odeiv_evolve_free( gslEvolve_ );
	if ( gslControl_ )
		gsl_odeiv_control_free( gslControl_ );
	if ( gslStep_ )
		gsl_odeiv_step_free( gslStep_ );
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

bool GslIntegrator::getIsInitialized() const
{
	return isInitialized_;
}

string GslIntegrator::getMethod() const
{
	return method_;
}
void GslIntegrator::setMethod( string method )
{
#ifdef USE_GSL
	method_ = method;
	gslStepType_ = 0;
	// cout << "in void GslIntegrator::innerSetMethod( string method ) \n";
	if ( method == "rk2" ) {
		gslStepType_ = gsl_odeiv_step_rk2;
	} else if ( method == "rk4" ) {
		gslStepType_ = gsl_odeiv_step_rk4;
	} else if ( method == "rk5" || method == "gsl" ) {
		gslStepType_ = gsl_odeiv_step_rkf45;
		method_ = "rk5";
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
		method_ = "rk5";
	}
#endif // USE_GSL
}

double GslIntegrator::getRelativeAccuracy() const
{
	return relAccuracy_;
}
void GslIntegrator::setRelativeAccuracy( double value )
{
	relAccuracy_ = value;
}

double GslIntegrator::getAbsoluteAccuracy() const
{
	return absAccuracy_;
}
void GslIntegrator::setAbsoluteAccuracy( double value )
{
	absAccuracy_ = value;
}

double GslIntegrator::getInternalDt() const
{
	return internalStepSize_;
}
void GslIntegrator::setInternalDt( double value )
{
	internalStepSize_ = value;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

/**
 * This function should also set up the sizes, and it should be at 
 * allocate, not reinit time.
 */
void GslIntegrator::stoich( const Eref& e, const Qinfo* q, Id stoichId )
{
#ifdef USE_GSL
	stoichId_ = stoichId;
	Stoich* s = reinterpret_cast< Stoich* >( stoichId.eref().data() );
	nVarPools_ = s->getNumVarPools();
	unsigned int meshIndex = e.index().value();
	y_ = s->getY( meshIndex );
	if ( meshIndex == 0 )
		s->clearFlux();

	isInitialized_ = 1;
        // Allocate GSL functions if not already allocated,
        // otherwise reset the reusable ones
        assert( gslStepType_ != 0 );
        if ( gslStep_ )
        {
            gsl_odeiv_step_free(gslStep_);
        }
        
        gslStep_ = gsl_odeiv_step_alloc( gslStepType_, nVarPools_ );
        
   	assert( gslStep_ != 0 );
        if ( !gslEvolve_ )
        {
            gslEvolve_ = gsl_odeiv_evolve_alloc(nVarPools_);
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
	gslSys_.dimension = nVarPools_;

	// Use a good guess at the correct ProcInfo to set up.
	// Should be reassigned at Reinit, just to be sure.
	stoichThread_.set( s, Shell::procInfo(), meshIndex );
	gslSys_.params = static_cast< void* >( &stoichThread_ );
	// gslSys_.params = static_cast< void* >( s );
#endif // USE_GSL
}

/**
 * Here we want to give the integrator as long a timestep as possible,
 * or alternatively let _it_ decide the timestep. The former is done
 * by providing a long dt, typically that of the graphing process.
 * The latter is harder to manage and works best if there is only this
 * one integrator running the simulation. Here we do the former.
 */
void GslIntegrator::process( const Eref& e, ProcPtr info )
{
#ifdef USE_GSL
	double nextt = info->currTime + info->dt;
	double t = info->currTime;
	while ( t < nextt ) {
		int status = gsl_odeiv_evolve_apply ( 
			gslEvolve_, gslControl_, gslStep_, &gslSys_, 
			&t, nextt,
			&internalStepSize_, y_);
		if ( status != GSL_SUCCESS )
			break;

		/*
		// Zero out buffered molecules. Perhaps this can be ignored
		for( vector< unsigned int >::const_iterator 
			i = dynamicBuffers_->begin(); 
			i != dynamicBuffers_->end(); ++i )
			y_[ *i ] = 0.0;
			*/
	}
#endif // USE_GSL
	Stoich* s = reinterpret_cast< Stoich* >( stoichId_.eref().data() );
	s->clearFlux( e.index().value(), info->threadIndexInGroup );
}

void GslIntegrator::reinit( const Eref& e, ProcPtr info )
{
	Stoich* s = reinterpret_cast< Stoich* >( stoichId_.eref().data() );
	unsigned int meshIndex = e.index().value();
	stoichThread_.set( s, info, meshIndex );
	if ( meshIndex == 0 ) {
		s->clearFlux();
		s->innerReinit();
	}
	nVarPools_ = s->getNumVarPools();
	y_ = s->getY( meshIndex );
#ifdef USE_GSL
	if ( isInitialized_ ) {
        assert( gslStepType_ != 0 );
        if ( gslStep_ )
        {
            gsl_odeiv_step_free(gslStep_);
        }
        
        gslStep_ = gsl_odeiv_step_alloc( gslStepType_, nVarPools_ );
        
   	assert( gslStep_ != 0 );
        if ( !gslEvolve_ )
        {
            gslEvolve_ = gsl_odeiv_evolve_alloc(nVarPools_);
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
	
	}
#endif // USE_GSL
}

void GslIntegrator::remesh( const Eref& e, const Qinfo* q,
	unsigned int numTotalEntries, unsigned int startEntry, 
	vector< unsigned int > localIndices, vector< double > vols )
{
	if ( e.index().value() != 0 ) {
		return;
	}
	if ( q->protectedAddToStructuralQ() )
		return;
	// cout << "GslIntegrator::remesh for " << e << endl;
	assert( vols.size() > 0 );
	if ( vols.size() != e.element()->dataHandler()->localEntries() ) {
		Neutral* n = reinterpret_cast< Neutral* >( e.data() );
		Id stoichId = stoichId_;
		n->setLastDimension( e, q, vols.size() );
	// instead of setLastDimension we should use a function that sets up
	// an arbitrary mapping of indices.

	// Note that at this point the data pointer may be invalid!
	// Now we reassign everything.
		assert( e.element()->dataHandler()->localEntries() == vols.size() );
		GslIntegrator* gsldata = reinterpret_cast< GslIntegrator* >( e.data() );
		for ( unsigned int i = 0; i < vols.size(); ++i ) {
			gsldata[i].stoich( e, q, stoichId );
		}
	}
}
