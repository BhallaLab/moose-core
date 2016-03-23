/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <math.h>
#include "header.h"
#include <sys/time.h>
#ifdef USE_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_odeiv.h>
#endif

#include "OdeSystem.h"
#include "VoxelPoolsBase.h"
#include "VoxelPools.h"
#include "RateTerm.h"
#include "FuncTerm.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "../mesh/VoxelJunction.h"
#include "XferInfo.h"
#include "ZombiePoolInterface.h"
#include "Stoich.h"

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////

//Rahul - adding some random work just to increase the time spent in process function...
time_t time_taken_voxel = 0;

VoxelPools::VoxelPools()
{
#ifdef USE_GSL
		driver_ = 0;
#endif
}

VoxelPools::~VoxelPools()
{
	for ( unsigned int i = 0; i < rates_.size(); ++i )
		delete( rates_[i] );
#ifdef USE_GSL
	if ( driver_ )
		gsl_odeiv2_driver_free( driver_ );
#endif
}

//////////////////////////////////////////////////////////////
// Solver ops
//////////////////////////////////////////////////////////////
void VoxelPools::reinit( double dt )
{
	VoxelPoolsBase::reinit();
#ifdef USE_GSL
	if ( !driver_ )
		return;
	gsl_odeiv2_driver_reset( driver_ );
	gsl_odeiv2_driver_reset_hstart( driver_, dt );
#endif
}

void VoxelPools::setStoich( Stoich* s, const OdeSystem* ode )
{
	stoichPtr_ = s;
#ifdef USE_GSL
	if ( ode ) {
		sys_ = ode->gslSys;
		if ( driver_ )
			gsl_odeiv2_driver_free( driver_ );
		driver_ = gsl_odeiv2_driver_alloc_y_new(&sys_, ode->gslStep, ode->initStepSize, ode->epsAbs, ode->epsRel );
	}
#endif
	VoxelPoolsBase::reinit();
}


static int rkf45_OpenMP_apply (void *vstate, size_t dim, double t, double h, double y[], double yerr[], const double dydt_in[], double dydt_out[], const gsl_odeiv2_system * sys)
{
	    rkf45_state_t *state = (rkf45_state_t *) vstate;
	    size_t i;
	    double *const k1 = state->k1;
	    double *const k2 = state->k2;
	    double *const k3 = state->k3;
	    double *const k4 = state->k4;
	    double *const k5 = state->k5;
	    double *const k6 = state->k6;
	    double *const ytmp = state->ytmp;
	    double *const y0 = state->y0;

	    int cellsPerThread = 1;
	    int numThreads = 4;

	    memcpy (y0, y, dim);

	    /*K1 step */
	    {
		    if (dydt_in != NULL)
				  memcpy (k1, dydt_in, dim);
		    else 
		    {
				  int s = GSL_ODEIV_FN_EVAL (sys, t, y, k1);
				  if (s != GSL_SUCCESS)
						return s;
		    }
	    }

#pragma omp parallel for private(i) shared(dim,h) schedule(guided, cellsPerThread) num_threads(numThreads)
	    for (i = 0; i < dim; i++)
			  ytmp[i] = y[i] +  ah[0] * h * k1[i];
	    /*k2 step */
	    {
		    int s = GSL_ODEIV_FN_EVAL (sys, t + ah[0] * h, ytmp, k2);
		    if (s != GSL_SUCCESS)
				  return s;
	    }

#pragma omp parallel for private(i) shared(dim,h) schedule(guided, cellsPerThread) num_threads(numThreads)
	    for (i = 0; i < dim; i++)
			  ytmp[i] = y[i] + h * (b3[0] * k1[i] + b3[1] * k2[i]);

	    /*k3 step */
	    {
		     int s = GSL_ODEIV_FN_EVAL (sys, t + ah[1] * h, ytmp, k3);
			if (s != GSL_SUCCESS)
				   return s;
	    }

#pragma omp parallel for private(i) shared(dim,h) schedule(guided, cellsPerThread) num_threads(numThreads)
	    for (i = 0; i < dim; i++)
			  ytmp[i] = y[i] + h * (b4[0] * k1[i] + b4[1] * k2[i] + b4[2] * k3[i]);

		 /*k4 step*/ 
	    {
			 int s = GSL_ODEIV_FN_EVAL (sys, t + ah[2] * h, ytmp, k4);
			 if (s != GSL_SUCCESS)
				    return s;
	    }

#pragma omp parallel for private(i) shared(dim,h) schedule(guided, cellsPerThread) num_threads(numThreads)
	    for (i = 0; i < dim; i++)
			  ytmp[i] = y[i] + h * (b5[0] * k1[i] + b5[1] * k2[i] + b5[2] * k3[i] +
b5[3] * k4[i]);

		 /*k5 step */
	    {
			 int s = GSL_ODEIV_FN_EVAL (sys, t + ah[3] * h, ytmp, k5);
			 if (s != GSL_SUCCESS)
				    return s;
	    }
#pragma omp parallel for private(i) shared(dim,h) schedule(guided, cellsPerThread) num_threads(numThreads)
			for (i = 0; i < dim; i++)
				   ytmp[i] = y[i] + h * (b4[0] * k1[i] + b4[1] * k2[i] + b4[2] * k3[i]);

		 /*k6 and final sum */
	    {
			 int s = GSL_ODEIV_FN_EVAL (sys, t + ah[4] * h, ytmp, k6);
			 if (s != GSL_SUCCESS)
				    return s;
	    }
#pragma omp parallel for private(i) shared(dim,h) schedule(guided, cellsPerThread) num_threads(numThreads)
			 for (i = 0; i < dim; i++)
			 {
				    const double d_i = c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i] + c6 * k6[i];
				    y[i] += h * d_i;
			 }

			 /* Derivatives at output */
					   
			 if (dydt_out != NULL)
			 {
				    int s = GSL_ODEIV_FN_EVAL (sys, t + h, y, dydt_out);
				    if (s != GSL_SUCCESS)
				    {
						  /* Restore initial values */
						  memcpy (y, y0, dim);
						  return s;
				    }
			 }
/* difference between 4th and 5th order */
#pragma omp parallel for private(i) shared(dim,h) schedule(guided, cellsPerThread) num_threads(numThreads)
			 for (i = 0; i < dim; i++)
				    yerr[i] = h * (ec[1] * k1[i] + ec[3] * k3[i] + ec[4] * k4[i] + ec[5] * k5[i] + ec[6] * k6[i]);


			 return GSL_SUCCESS;
}


int VoxelPools::rungeKutta45(void *vstate, size_t dim, double t, double h, double y[], double yerr[], const double dydt_in[], double dydt_out[], const gsl_odeiv2_system * sys)
{

	   double tmpVar = 7.345;
	   double ttmp = 0.847; 
	   for(int j = 0; j < 90; j++)
			 for(int k = 0; k < 100; k++)
				    tmpVar += ttmp;

	struct timeval stop, start;
	gettimeofday(&start, NULL);

	    int GSLSUCCESS = 0;
	    rkf45_state_t *state = (rkf45_state_t *) vstate;
	    size_t i;
	    double *const k1 = state->k1;
	    double *const k2 = state->k2;
	    double *const k3 = state->k3;
	    double *const k4 = state->k4;
	    double *const k5 = state->k5;
	    double *const k6 = state->k6;
	    double *const ytmp = state->ytmp;
	    double *const y0 = state->y0;


	    memcpy (y0, y, dim); // memcpy in case of failure...

	    /*K1 step */
	    {
		    if (dydt_in == NULL)
		    {
				  int s = RKF45_ODEIV_FN_EVAL (sys, t, y, k1);
				  if (s != GSLSUCCESS)
						return s;
		    }
		    else
				  memcpy (k1, dydt_in, dim);
	    }
		    for (i = 0; i < dim; i++)
				  ytmp[i] = y[i] + ah[0] * h * k1[i];

	    /*k2 step */
	    {
		    int s = RKF45_ODEIV_FN_EVAL (sys, t + ah[0] * h, ytmp, k2);
		    if (s != GSLSUCCESS)
				  return s;
	    }
		    for (i = 0; i < dim; i++)
				  ytmp[i] = y[i] + h * (b3[0] * k1[i] + b3[1] * k2[i]);

	    /*k3 step */
	    {
		     int s = RKF45_ODEIV_FN_EVAL (sys, t + ah[1] * h, ytmp, k3);
			if (s != GSLSUCCESS)
				   return s;
	    }

			for (i = 0; i < dim; i++)
				   ytmp[i] = y[i] + h * (b4[0] * k1[i] + b4[1] * k2[i] + b4[2] * k3[i]);

		 /*k4 step*/ 
	    {
			 int s = RKF45_ODEIV_FN_EVAL (sys, t + ah[2] * h, ytmp, k4);
			 if (s != GSLSUCCESS)
				    return s;
	    }

			 for (i = 0; i < dim; i++)
				    ytmp[i] =  y[i] + h * (b5[0] * k1[i] + b5[1] * k2[i] + b5[2] * k3[i] + b5[3] * k4[i]);

		 /*k5 step */
	    {
			 int s = RKF45_ODEIV_FN_EVAL (sys, t + ah[3] * h, ytmp, k5);
			 if (s != GSLSUCCESS)
				    return s;
	    }

			 for (i = 0; i < dim; i++)
				    ytmp[i] = y[i] + h * (b6[0] * k1[i] + b6[1] * k2[i] + b6[2] * k3[i] + b6[3] * k4[i] + b6[4] * k5[i]);

		 /*k6 and final sum */
	    {
			 int s = RKF45_ODEIV_FN_EVAL (sys, t + ah[4] * h, ytmp, k6);
			 if (s != GSLSUCCESS)
				    return s;
	    }

			 for (i = 0; i < dim; i++)
			 {
				    const double d_i = c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i] + c6 * k6[i];
				    y[i] += h * d_i;
			 }

			 /* Derivatives at output */
					   
			 if (dydt_out != NULL)
			 {
				    int s = RKF45_ODEIV_FN_EVAL (sys, t + h, y, dydt_out);
				    if (s != GSLSUCCESS)
				    {
						  /* Restore initial values */
						  memcpy (y, y0, dim);
						  return s;
				    }
			 }
			 /* difference between 4th and 5th order */
			   for (i = 0; i < dim; i++)
					 yerr[i] = h * (ec[1] * k1[i] + ec[3] * k3[i] + ec[4] * k4[i] + ec[5] * k5[i] + ec[6] * k6[i]);


	gettimeofday(&stop, NULL);
	
	time_taken_voxel = stop.tv_usec - start.tv_usec;

//	cout << "Time Taken from rungekutta = " << time_taken_voxel << endl;
			     
			   return GSLSUCCESS;
}

void VoxelPools::advance( const ProcInfo* p )
{

#ifdef USE_GSL
	   double T = p->currTime - p->dt;

	int status = gsl_odeiv2_driver_apply( driver_, &T, p->currTime, varS());

	if ( status != GSL_SUCCESS ) {
		cout << "Error: VoxelPools::advance: GSL integration error at time "
			 << T << "\n";
		cout << "Error info: " << status << ", " << 
				gsl_strerror( status ) << endl;
		if ( status == GSL_EMAXITER ) 
			cout << "Max number of steps exceeded\n";
		else if ( status == GSL_ENOPROG ) 
			cout << "Timestep has gotten too small\n";
		else if ( status == GSL_EBADFUNC ) 
			cout << "Internal error\n";
		assert( 0 );
	}

//	   int status;
//	   int GSLSUCCESS = 0;
//
//	   const double t1 = p->currTime;
//			 
//	   const gsl_odeiv2_system* dydt = driver_->sys;
//	   double h0 = driver_->h;
//	   int final_step = 0;
//
//
//
//    /* Determine integration direction sign */
//
//	    while (t1 - T > 0.0)
//	    {
//			 const gsl_odeiv2_system* dydt = driver_->sys;
//			 double h0 = driver_->h;
//			 int final_step = 0;
//			 double dt = t1 - T;
//			
//			 if ((dt >= 0.0 && h0 > dt) || (dt < 0.0 && h0 < dt))
//			 {
//				    h0 = dt;
//				    final_step = 1;
//			 }
//			 else
//				    final_step = 0;
//					  
//			 if (driver_->s->type->can_use_dydt_in)
//				    status = rungeKutta45(driver_->s->state, driver_->s->dimension, T, h0, varS(), driver_->e->yerr, driver_->e->dydt_in, driver_->e->dydt_out, dydt);
//
//			 if (final_step)
//				    T = t1;
//			 else
//				    T = T + h0;
//					    
//			 /* Suggest step size for next time-step. Change of step size is not  suggested in the final step, because that step can be very small compared to previous step, to reach t1. */
//			 if (final_step == 0) driver_->h = h0;
//	    }
//
//	if ( status != GSLSUCCESS ) 
//	{
//		cout << "Error: VoxelPools::advance: GSL integration error at time " << T << "\n";
//		assert( 0 );
//	}
#endif
}

void VoxelPools::setInitDt( double dt )
{
#ifdef USE_GSL
	gsl_odeiv2_driver_reset_hstart( driver_, dt );
#endif
}

// static func. This is the function that goes into the Gsl solver.
int VoxelPools::gslFunc( double t, const double* y, double *dydt, void* params )
{
	VoxelPools* vp = reinterpret_cast< VoxelPools* >( params );
	double* q = const_cast< double* >( y ); // Assign the func portion.

	vp->stoichPtr_->updateFuncs( q, t );
	vp->updateRates( y, dydt );
#ifdef USE_GSL
	return GSL_SUCCESS;
#else
	return 0;
#endif
}
///////////////////////////////////////////////////////////////////////
// Here are the internal reaction rate calculation functions
///////////////////////////////////////////////////////////////////////

void VoxelPools::updateAllRateTerms( const vector< RateTerm* >& rates,
			   unsigned int numCoreRates )
{
	// Clear out old rates if any
	for ( unsigned int i = 0; i < rates_.size(); ++i )
		delete( rates_[i] );

	rates_.resize( rates.size() );
	for ( unsigned int i = 0; i < numCoreRates; ++i )
		rates_[i] = rates[i]->copyWithVolScaling( getVolume(), 1, 1 );
	for ( unsigned int i = numCoreRates; i < rates.size(); ++i ) {
		rates_[i] = rates[i]->copyWithVolScaling(  getVolume(), 
				getXreacScaleSubstrates(i - numCoreRates),
				getXreacScaleProducts(i - numCoreRates ) );
	}
}

void VoxelPools::updateRateTerms( const vector< RateTerm* >& rates,
			   unsigned int numCoreRates, unsigned int index )
{
	// During setup or expansion of the reac system, it is possible to
	// call this function before the rates_ term is assigned. Disable.
 	if ( index >= rates_.size() )
		return;
	delete( rates_[index] );
	if ( index >= numCoreRates )
		rates_[index] = rates[index]->copyWithVolScaling(
				getVolume(), 
				getXreacScaleSubstrates(index - numCoreRates),
				getXreacScaleProducts(index - numCoreRates ) );
	else
		rates_[index] = rates[index]->copyWithVolScaling(  
				getVolume(), 1.0, 1.0 );
}

void VoxelPools::updateRates( const double* s, double* yprime ) const
{
	const KinSparseMatrix& N = stoichPtr_->getStoichiometryMatrix();
	vector< double > v( N.nColumns(), 0.0 );
	vector< double >::iterator j = v.begin();
	// totVar should include proxyPools only if this voxel uses them
	unsigned int totVar = stoichPtr_->getNumVarPools() + 
			stoichPtr_->getNumProxyPools();
	// totVar should include proxyPools if this voxel does not use them
	unsigned int totInvar = stoichPtr_->getNumBufPools();
	assert( N.nColumns() == 0 || 
			N.nRows() == stoichPtr_->getNumAllPools() );
	assert( N.nColumns() == rates_.size() );

	for ( vector< RateTerm* >::const_iterator
		i = rates_.begin(); i != rates_.end(); i++) {
		*j++ = (**i)( s );
		assert( !std::isnan( *( j-1 ) ) );
	}

	for (unsigned int i = 0; i < totVar; ++i)
		*yprime++ = N.computeRowRate( i , v );
	for (unsigned int i = 0; i < totInvar ; ++i)
		*yprime++ = 0.0;
}

/**
 * updateReacVelocities computes the velocity *v* of each reaction.
 * This is a utility function for programs like SteadyState that need
 * to analyze velocity.
 */
void VoxelPools::updateReacVelocities( 
			const double* s, vector< double >& v ) const
{
	const KinSparseMatrix& N = stoichPtr_->getStoichiometryMatrix();
	assert( N.nColumns() == rates_.size() );

	vector< RateTerm* >::const_iterator i;
	v.clear();
	v.resize( rates_.size(), 0.0 );
	vector< double >::iterator j = v.begin();

	for ( i = rates_.begin(); i != rates_.end(); i++) {
		*j++ = (**i)( s );
		assert( !std::isnan( *( j-1 ) ) );
	}
}

/// For debugging: Print contents of voxel pool
void VoxelPools::print() const
{
	cout << "numAllRates = " << rates_.size() << 
			", numLocalRates= " << stoichPtr_->getNumCoreRates() << endl;
	VoxelPoolsBase::print();
}

////////////////////////////////////////////////////////////
/** 
 * Handle volume updates. Inherited Virtual func.
 */
void VoxelPools::setVolumeAndDependencies( double vol )
{
	VoxelPoolsBase::setVolumeAndDependencies( vol );
	stoichPtr_->setupCrossSolverReacVols();
	updateAllRateTerms( stoichPtr_->getRateTerms(), 
		stoichPtr_->getNumCoreRates() );
}

////////////////////////////////////////////////////////////
#if 0
/**
 * Zeroes out rate terms that are involved in cross-reactions that 
 * are not present on current voxel.
 */
void VoxelPools::filterCrossRateTerms(
		const vector< pair< Id, Id > >&  
				offSolverReacCompts  )
{
		/*
From VoxelPoolsBase:proxyPoolVoxels[comptIndex][#] we know
if specified compt has local proxies.
	Note that compt is identified by an index, and actually looks up
	the Ksolve.
From Ksolve::compartment_ we know which compartment a given ksolve belongs 
	in
From Ksolve::xfer_[otherKsolveIndex].ksolve we have the id of the other
	Ksolves.
From Stoich::offSolverReacCompts_ which is pair< Id, Id > we have the 
	ids of the _compartments_ feeding into the specified rateTerms.

Somewhere I need to make a map of compts to comptIndex.

The ordering of the xfer vector is simply by the order of the script call
for buildXfer.

This has become too ugly
Skip the proxyPoolVoxels info, or use the comptIndex here itself to
build the table.
comptIndex looks up xfer which holds the Ksolve Id. From that we can
get the compt id. All this relies on this mapping being correct.
Or I should pass in the compt when I build it.

OK, now we have VoxelPoolsBase::proxyPoolCompts_ vector to match the
comptIndex.

*/
	unsigned int numCoreRates = stoichPtr_->getNumCoreRates();
 	for ( unsigned int i = 0; i < offSolverReacCompts.size(); ++i ) {
		const pair< Id, Id >& p = offSolverReacCompts[i];
		if ( !isVoxelJunctionPresent( p.first, p.second) ) {
			unsigned int k = i + numCoreRates;
			assert( k < rates_.size() );
			if ( rates_[k] )
				delete rates_[k];
			rates_[k] = new ExternReac;
		}
	}
}
#endif
