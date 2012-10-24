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
#include "StoichPools.h"
#include "GslStoich.h"
#include "../shell/Shell.h"

const Cinfo* GslStoich::initCinfo()
{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		static ReadOnlyValueFinfo< GslStoich, bool > isInitialized( 
			"isInitialized", 
			"True if the Stoich message has come in to set parms",
			&GslStoich::getIsInitialized
		);
		static ValueFinfo< GslStoich, string > method( "method", 
			"Numerical method to use.",
			&GslStoich::setMethod,
			&GslStoich::getMethod 
		);
		static ValueFinfo< GslStoich, double > relativeAccuracy( 
			"relativeAccuracy", 
			"Accuracy criterion",
			&GslStoich::setRelativeAccuracy,
			&GslStoich::getRelativeAccuracy
		);
		static ValueFinfo< GslStoich, double > absoluteAccuracy( 
			"absoluteAccuracy", 
			"Another accuracy criterion",
			&GslStoich::setAbsoluteAccuracy,
			&GslStoich::getAbsoluteAccuracy
		);
		static ValueFinfo< GslStoich, double > internalDt( 
			"internalDt", 
			"internal timestep to use.",
			&GslStoich::setInternalDt,
			&GslStoich::getInternalDt
		);

		///////////////////////////////////////////////////////
		// DestFinfo definitions
		///////////////////////////////////////////////////////
		static DestFinfo stoich( "stoich",
			"Handle data from Stoich",
			new EpFunc1< GslStoich, Id >( &GslStoich::stoich )
		);

		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< GslStoich >( &GslStoich::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< GslStoich >( &GslStoich::reinit ) );

		static DestFinfo remesh( "remesh",
			"Handle commands to remesh the pool. This may involve changing "
			"the number of pool entries, as well as changing their volumes",
			new EpFunc5< GslStoich, 
			double,
			unsigned int, unsigned int, 
			vector< unsigned int >, vector< double > >( 
					&GslStoich::remesh )
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
		"GslStoich",
		Neutral::initCinfo(),
		gslIntegratorFinfos,
		sizeof(gslIntegratorFinfos)/sizeof(Finfo *),
		new Dinfo< GslStoich >
	);

	return &gslIntegratorCinfo;
}

static const Cinfo* gslIntegratorCinfo = GslStoich::initCinfo();

///////////////////////////////////////////////////
// Basic class function definitions
///////////////////////////////////////////////////

GslStoich::GslStoich()
	: 
	isInitialized_( 0 ),
	method_( "rk5" ),
	absAccuracy_( 1.0e-9 ),
	relAccuracy_( 1.0e-6 ),
	internalStepSize_( 1.0 ),
	y_( 0 ),
	stoichId_(),
	stoich_( 0 ),
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
 * do the initialization in a separate call to GslStoich::stoich().
 */
GslStoich& GslStoich::operator=( const GslStoich& other )
{
	isInitialized_ = 0;
	method_ = "rk5";
	absAccuracy_ = 1.0e-9;
	relAccuracy_ = 1.0e-6;
	internalStepSize_ = 1.0;
	y_.clear();
	gslStepType_ = gsl_odeiv_step_rkf45;
	gslStep_ = 0;
	gslControl_ = 0;
	gslEvolve_ = 0;
	stoichId_ = Id();
	stoich_ = 0;

	return *this;
}

GslStoich::~GslStoich()
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

bool GslStoich::getIsInitialized() const
{
	return isInitialized_;
}

string GslStoich::getMethod() const
{
	return method_;
}
void GslStoich::setMethod( string method )
{
#ifdef USE_GSL
	method_ = method;
	gslStepType_ = 0;
	// cout << "in void GslStoich::innerSetMethod( string method ) \n";
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
		cout << "Warning: GslStoich::innerSetMethod: method '" <<
			method << "' not known, using rk5\n";
		gslStepType_ = gsl_odeiv_step_rkf45;
		method_ = "rk5";
	}
#endif // USE_GSL
}

double GslStoich::getRelativeAccuracy() const
{
	return relAccuracy_;
}
void GslStoich::setRelativeAccuracy( double value )
{
	relAccuracy_ = value;
}

double GslStoich::getAbsoluteAccuracy() const
{
	return absAccuracy_;
}
void GslStoich::setAbsoluteAccuracy( double value )
{
	absAccuracy_ = value;
}

double GslStoich::getInternalDt() const
{
	return internalStepSize_;
}
void GslStoich::setInternalDt( double value )
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
void GslStoich::stoich( const Eref& e, const Qinfo* q, Id stoichId )
{
#ifdef USE_GSL
	stoichId_ = stoichId;
	stoich_ = reinterpret_cast< StoichCore* >( stoichId.eref().data() );
	unsigned int nVarPools = stoich_->getNumVarPools();
	// stoich_->clearFlux();

	isInitialized_ = 1;
        // Allocate GSL functions if not already allocated,
        // otherwise reset the reusable ones
        assert( gslStepType_ != 0 );
        if ( gslStep_ )
        {
            gsl_odeiv_step_free(gslStep_);
        }
        
        gslStep_ = gsl_odeiv_step_alloc( gslStepType_, nVarPools );
        
   	assert( gslStep_ != 0 );
        if ( !gslEvolve_ )
        {
            gslEvolve_ = gsl_odeiv_evolve_alloc(nVarPools);
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
	gslSys_.dimension = nVarPools;
	gslSys_.params = static_cast< void* >( this );
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
void GslStoich::process( const Eref& e, ProcPtr info )
{
#ifdef USE_GSL
	double nextt = info->currTime + info->dt;
	for ( currMeshEntry_ = 0; 
					currMeshEntry_ < numMeshEntries(); ++currMeshEntry_ ) {
		double t = info->currTime;
		while ( t < nextt ) {
			int status = gsl_odeiv_evolve_apply ( 
				gslEvolve_, gslControl_, gslStep_, &gslSys_, 
				&t, nextt,
				&internalStepSize_, &y_[currMeshEntry_][0] );
			if ( status != GSL_SUCCESS )
				break;

		}
	}
#endif // USE_GSL
	// stoich_->clearFlux( e.index().value(), info->threadIndexInGroup );
}

void GslStoich::reinit( const Eref& e, ProcPtr info )
{
	// stoich_->clearFlux();
	// stoich_->innerReinit();
	unsigned int nVarPools = stoich_->getNumVarPools();
#ifdef USE_GSL
	if ( isInitialized_ ) {
        assert( gslStepType_ != 0 );
        if ( gslStep_ )
        {
            gsl_odeiv_step_free(gslStep_);
        }
        
        gslStep_ = gsl_odeiv_step_alloc( gslStepType_, nVarPools );
        
   	assert( gslStep_ != 0 );
        if ( !gslEvolve_ )
        {
            gslEvolve_ = gsl_odeiv_evolve_alloc(nVarPools);
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

void GslStoich::remesh( const Eref& e, const Qinfo* q,
	double oldVol,
	unsigned int numTotalEntries, unsigned int startEntry, 
	vector< unsigned int > localIndices, vector< double > vols )
{
	if ( e.index().value() != 0 ) {
		return;
	}
	/*
	if ( q->addToStructuralQ() )
		return;
		*/
	// cout << "GslStoich::remesh for " << e << endl;
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
		GslStoich* gsldata = reinterpret_cast< GslStoich* >( e.data() );
		for ( unsigned int i = 0; i < vols.size(); ++i ) {
			gsldata[i].stoich( e, q, stoichId );
		}
	}
}
///////////////////////////////////////////////////
// Numerical function definitions
///////////////////////////////////////////////////

/*
// Update the v_ vector for individual reac velocities.
void GslStoich::updateV( vector< double >& v )
{
	// Some algorithm to assign the values from the computed rates
	// to the corresponding v_ vector entry
	// for_each( rates_.begin(), rates_.end(), assign);

	// v.resize( rates_.size() );
	vector< RateTerm* >::const_iterator i;
	vector< double >::iterator j = v.begin();
	const double* s = S( currMeshEntry_ );

	for ( i = rates_.begin(); i != rates_.end(); i++)
	{
		*j++ = (**i)( s );
		assert( !isnan( *( j-1 ) ) );
	}
}

void GslStoich::updateRates( vector< double>* yprime, double dt, 
	vector< double >& v )
{
	stoich_->updateV( S( currMeshEntry_ ), v );

	stoich_->updateRates( yprime, dt, v );
}

// Update the function-computed molecule terms. These are not integrated,
// but their values may be used by molecules that are.
// The molecule vector S_ has a section for FuncTerms. In this section
// there is a one-to-one match between entries in S_ and FuncTerm entries.
void GslStoich::updateFuncs( double t, unsigned int meshIndex )
{
	vector< FuncTerm* >::const_iterator i;
	vector< double >::iterator j = S_[meshIndex].begin() + numVarPools_ + numBufPools_;

	for ( i = funcs_.begin(); i != funcs_.end(); i++)
	{
		*j++ = (**i)( &( S_[meshIndex][0] ), t );
		assert( !isnan( *( j-1 ) ) );
	}
}
*/



/**
 * This is the function used by GSL to advance the simulation one step.
 * We have a design decision here: to perform the calculations 'in place'
 * on the passed in y and f arrays, or to copy the data over and use
 * the native calculations in the Stoich object. We chose the latter,
 * because memcpy is fast, and the alternative would be to do a huge
 * number of array lookups (currently it is direct pointer references).
 * Someday should benchmark to see how well it works.
 * The derivative array f is used directly by the stoich function
 * updateRates that computes these derivatives, so we do not need to
 * do any memcopies there.
 *
 * Perhaps not by accident, this same functional form is used by CVODE.
 * Should make it easier to eventually use CVODE as a solver too.
 */

// Static function passed in as the stepper for GSL
int GslStoich::gslFunc( double t, const double* y, double* yprime, void* s )
{
	GslStoich* g = static_cast< GslStoich* >( s );
	return g->innerGslFunc( t, y, yprime );
}


int GslStoich::innerGslFunc( double t, const double* y, double* yprime ) 
{
	// Copy the y array into the S_ vector.
	memcpy( varS( currMeshEntry_ ), y, 
					stoich_->getNumVarPools() * sizeof( double ) );

	stoich_->updateFuncs( varS( currMeshEntry_ ), t );

	stoich_->updateRates( S( currMeshEntry_ ), yprime );

	// updateDiffusion happens in the previous Process Tick, coordinated
	// by the MeshEntries. At this point the new values are there in the
	// flux_ matrix.

	return GSL_SUCCESS;
}
