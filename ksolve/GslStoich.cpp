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
#include "../shell/Shell.h"
#include "../mesh/MeshEntry.h"
#include "../mesh/Boundary.h"
#include "../mesh/ChemMesh.h"
#include "GslStoich.h"

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

		static ValueFinfo< GslStoich, Id > compartment( 
			"compartment",
			"This is the Id of the compartment, which must be derived from"
			"the ChemMesh baseclass. The GslStoich needs"
			"the ChemMesh Id only for diffusion, "
			" and one can pass in Id() instead if there is no diffusion,"
			" or just leave it unset.",
			&GslStoich::setCompartment,
			&GslStoich::getCompartment
		);

		///////////////////////////////////////////////////////
		// DestFinfo definitions
		///////////////////////////////////////////////////////
		static DestFinfo stoich( "stoich",
			"Assign the StoichCore and ChemMesh Ids. The GslStoich needs"
			"the StoichCore pointer in all cases, in order to perform all"
			"calculations.",
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
		&compartment,		// Value
		&stoich,			// DestFinfo
		&remesh,			// DestFinfo
		&proc,				// SharedFinfo
	};
	
	static  Cinfo gslIntegratorCinfo(
		"GslStoich",
		StoichPools::initCinfo(),
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
	compartmentId_( 0 ),
	diffusionMesh_( 0 ),
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

Id GslStoich::getCompartment() const
{
	return compartmentId_;
}
void GslStoich::setCompartment( Id value )
{
	if ( value == Id() || !value.element()->cinfo()->isA( "ChemMesh" ) )
   	{
		cout << "Warning: GslStoich::setCompartment: "
				"Value must be a ChemMesh subclass\n";
		compartmentId_ = Id();
		diffusionMesh_ = 0;
	} else {
		compartmentId_ = value;
		diffusionMesh_ = reinterpret_cast< ChemMesh* >(
				compartmentId_.eref().data() );
	}
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
	if ( stoichId_ == Id() ) {
		isInitialized_ = 0;
		return;
	}
	stoich_ = reinterpret_cast< StoichCore* >( stoichId.eref().data() );

	unsigned int nVarPools = stoich_->getNumVarPools();
	// stoich_->clearFlux();
	resizeArrays( stoich_->getNumAllPools() );
	vector< double > temp( stoich_->getNumVarPools(), 0.0 );
	y_.resize( numMeshEntries(), temp );

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
        
        
	gslSys_.function = &GslStoich::gslFunc;
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
	if ( !isInitialized_ )
			return;
#ifdef USE_GSL
			// Hack till we sort out threadData
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
	if ( !isInitialized_ )
			return;
	unsigned int nVarPools = stoich_->getNumVarPools();
	for ( unsigned int i = 0; i < numMeshEntries(); ++i ) {
		memcpy( varS( i ), Sinit( i ), nVarPools * sizeof( double ) );
		memcpy( &(y_[i][0]), Sinit( i ), nVarPools * sizeof( double ) );
		stoich_->updateFuncs( varS( i ), 0 );
	}

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
	// cout << "GslStoich::remesh for " << e << endl;
	assert( vols.size() > 0 );
	// Here we could have a change in the meshing, or in the volumes.
	// Either way we need to do scaling.
	unsigned int numPools = numPoolEntries( 0 );
	vector< double > initConcs( numPools, 0.0 );
	vector< unsigned int > localEntryList( vols.size(), 0 );
	for ( unsigned int i = 0; i < vols.size(); ++i )
			localEntryList[i] = i;

	for ( unsigned int i = 0; i < numPools; ++i ) {
		initConcs[i] = Sinit( 0 )[i] / ( NA * oldVol );
	}
	meshSplit( initConcs, vols, localEntryList );
	vector< double > temp( numPools, 0.0 );
	y_.resize( vols.size(), temp );
}
///////////////////////////////////////////////////
// Numerical function definitions
///////////////////////////////////////////////////

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

void GslStoich::updateDiffusion( double *yprime )
{
	const double *adx; 
	const unsigned int* colIndex;
	unsigned int numInRow = 
			diffusionMesh_->getStencil( currMeshEntry_, &adx, &colIndex);
	double vSelf = diffusionMesh_->getMeshEntrySize( currMeshEntry_ );
	const double* sSelf = S( currMeshEntry_ );
	for ( unsigned int i = 0; i < numInRow; ++i ) {
		double scale = adx[i] ;
		scale = 1.0 / ( scale * scale );
		unsigned int other = colIndex[i];

		// Get all concs at the other meshEntry
		const double* sOther = S( other ); 
		double vOther = diffusionMesh_->getMeshEntrySize( other );
		
		for ( unsigned int j = 0; j < stoich_->getNumVarPools(); ++j )
			yprime[j] += stoich_->getDiffConst(j) * scale * 
					( sOther[j]/vOther - sSelf[j]/vSelf );
	}
}

int GslStoich::innerGslFunc( double t, const double* y, double* yprime ) 
{
	// Copy the y array into the S_ vector.
	memcpy( varS( currMeshEntry_ ), y, 
					stoich_->getNumVarPools() * sizeof( double ) );

	stoich_->updateFuncs( varS( currMeshEntry_ ), t );

	stoich_->updateRates( S( currMeshEntry_ ), yprime );

	if ( diffusionMesh_ && diffusionMesh_->innerGetNumEntries() > 1 )
		updateDiffusion( yprime );
	
	/*
	cout << "\nTime = " << t << endl;
	for ( unsigned int i = 0; i < stoich_->getNumVarPools(); ++i )
			cout << i << "	" << S( currMeshEntry_ )[i] << 
					"	" << S( currMeshEntry_ )[i] << endl;
					*/

	// updateDiffusion happens in the previous Process Tick, coordinated
	// by the MeshEntries. At this point the new values are there in the
	// flux_ matrix.

	return GSL_SUCCESS;
}

///////////////////////////////////////////////////
// Field access functions
///////////////////////////////////////////////////

void GslStoich::setN( const Eref& e, double v )
{
	unsigned int i = e.index().value(); // Later: Handle node location.
	unsigned int j = stoich_->convertIdToPoolIndex( e.id() );
	assert( i < numMeshEntries() );
	assert( j < numPoolEntries( i ) );
	varS(i)[j] = v;
}

double GslStoich::getN( const Eref& e ) const
{
	unsigned int i = e.index().value();
	unsigned int j = stoich_->convertIdToPoolIndex( e.id() );
	assert( i < numMeshEntries() );
	assert( j < numPoolEntries( i ) );
	return S(i)[j];
}

void GslStoich::setNinit( const Eref& e, double v )
{
	unsigned int i = e.index().value();
	unsigned int j = stoich_->convertIdToPoolIndex( e.id() );
	assert( i < numMeshEntries() );
	assert( j < numPoolEntries( i ) );
	varSinit(i)[j] = v;
}

double GslStoich::getNinit( const Eref& e ) const
{
	unsigned int i = e.index().value();
	unsigned int j = stoich_->convertIdToPoolIndex( e.id() );
	assert( i < numMeshEntries() );
	assert( j < numPoolEntries( i ) );
	return Sinit(i)[j];
}

void GslStoich::setSpecies( const Eref& e, unsigned int v )
{
	unsigned int j = stoich_->convertIdToPoolIndex( e.id() );
	assert( j < numPoolEntries( e.index().value() ) );
	stoich_->setSpecies( j, v );
}

unsigned int GslStoich::getSpecies( const Eref& e )
{
	unsigned int j = stoich_->convertIdToPoolIndex( e.id() );
	assert( j < numPoolEntries( e.index().value() ) );
	return stoich_->getSpecies( j );
}

void GslStoich::setDiffConst( const Eref& e, double v )
{
	unsigned int j = stoich_->convertIdToPoolIndex( e.id() );
	assert( j < numPoolEntries( e.index().value() ) );
	stoich_->setDiffConst( j, v );
}

double GslStoich::getDiffConst( const Eref& e ) const
{
	unsigned int j = stoich_->convertIdToPoolIndex( e.id() );
	assert( j < numPoolEntries( e.index().value() ) );
	return stoich_->getDiffConst( j );
}
