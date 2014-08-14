/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "RateTerm.h"
#include "FuncTerm.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "ZombiePoolInterface.h"
#include "Stoich.h"
#include "GssaSystem.h"
#include "VoxelPoolsBase.h"
#include "GssaVoxelPools.h"
#include "../randnum/randnum.h"

/**
 * The SAFETY_FACTOR Protects against the total propensity exceeding
 * the cumulative
 * sum of propensities, atot. We do a lot of adding and subtracting of
 * dependency terms from atot. Roundoff error will eventually cause
 * this to drift from the true sum. To guarantee that we never lose
 * the propensity of the last reaction, this safety factor scales the
 * first calculation of atot to be slightly larger. Periodically this
 * will cause the reaction picking step to exceed the last reaction 
 * index. This is safe, we just pick another random number.  
 * This will happen rather infrequently.
 * That is also a good time to update the cumulative sum.
 * A double should have >15 digits, so cumulative error will be much
 * smaller than this.
 */
const double SAFETY_FACTOR = 1.0 + 1.0e-9;

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////

GssaVoxelPools::GssaVoxelPools()
	: 
			VoxelPoolsBase(),
			t_( 0.0 ),
			atot_( 0.0 )
{;}

GssaVoxelPools::~GssaVoxelPools()
{;}

//////////////////////////////////////////////////////////////
// Solver ops
//////////////////////////////////////////////////////////////

void GssaVoxelPools::updateDependentMathExpn( 
				const GssaSystem* g, unsigned int rindex )
{
	const vector< unsigned int >& deps = g->dependentMathExpn[ rindex ];
	unsigned int offset = g->stoich->getNumVarPools() + 
			g->stoich->getNumBufPools();
	for( vector< unsigned int >::const_iterator 
			i = deps.begin(); i != deps.end(); ++i ) {
		varS()[ *i + offset] = g->stoich->funcs( *i )->operator()( S(), t_);
	}
}

void GssaVoxelPools::updateDependentRates( 
	const vector< unsigned int >& deps, const Stoich* stoich )
{
	for ( vector< unsigned int >::const_iterator
			i = deps.begin(); i != deps.end(); ++i ) {
		atot_ -= v_[ *i ];
		// atot_ += ( v[ *i ] = ( *rates_[ *i ] )( S() );
		atot_ += ( v_[ *i ] = getReacVelocity( *i, S() ) );
	}
}

unsigned int GssaVoxelPools::pickReac() const
{
	// double r =  gsl_rng_uniform( rng ) * atot_;
	double r = mtrand() * atot_;
	double sum = 0.0;

	// This is an inefficient way to do it. Can easily get to 
	// log time or thereabouts by doing one or two levels of 
	// subsidiary tables. Too many levels causes slow-down because
	// of overhead in managing the tree. 
	// Slepoy, Thompson and Plimpton 2008
	// report a linear time version.
	for ( vector< double >::const_iterator 
			i = v_.begin(); i != v_.end(); ++i ) {
		if ( r < ( sum += *i ) )
			return static_cast< unsigned int >( i - v_.begin() );
	}
	return v_.size();
}

void GssaVoxelPools::setNumReac( unsigned int n )
{
	v_.clear();
	v_.resize( n, 0.0 );
}

void GssaVoxelPools::advance( const ProcInfo* p, const GssaSystem* g )
{
	double nextt = p->currTime;
	while ( t_ < nextt ) {
		if ( atot_ <= 0.0 ) { // reac system is stuck, will not advance.
			t_ = nextt;
			return;
		}
		unsigned int rindex = pickReac();
		if ( rindex >= g->stoich->getNumRates() ) {
			// probably cumulative roundoff error here. 
			// Recalculate atot to avoid, and redo.
			updateReacVelocities( g, S(), v_ );
			atot_ = 0;
			for ( vector< double >::const_iterator 
					i = v_.begin(); i != v_.end(); ++i )
				atot_ += *i;
			atot_ *= SAFETY_FACTOR;
			rindex = v_.size() - 1;
		}

		g->transposeN.fireReac( rindex, Svec() );
		updateDependentMathExpn( g, rindex );
		// atot_ = g->updateDependentRates( atot_, rinidex );
		updateDependentRates( g->dependency[ rindex ], g->stoich );
		// double r = gsl_rng_uniform( rng );
		double r = mtrand();
		while ( r <= 0.0 ) {
			// r = gsl_rng_uniform( rng )
			r = mtrand();
		}
		t_ -= ( 1.0 / atot_ ) * log( r );
	}
}

void GssaVoxelPools::reinit( const GssaSystem* g )
{
	VoxelPoolsBase::reinit(); // Assigns S = Sinit;
	g->stoich->updateFuncs( varS(), 0 );

	unsigned int numVarPools = g->stoich->getNumVarPools();
	double* n = varS();
	if ( g->useRandInit ) { 
		// round up or down probabilistically depending on fractional 
		// num molecules.
		for ( unsigned int i = 0; i < numVarPools; ++i ) {
			double base = floor( n[i] );
			double frac = n[i] - base;
			// if ( gsl_rng_uniform( rng ) > frac )
			if ( mtrand() > frac )
				n[i] = base;
			else
				n[i] = base + 1.0;
		}
	} else { // Just round to the nearest int.
		for ( unsigned int i = 0; i < numVarPools; ++i ) {
			n[i] = round( n[i] );
		}
	}
	t_ = 0.0;
	// vector< double > yprime( g->stoich->getNumAllPools(), 0.0 );
				// i = yprime.begin(); i != yprime.end(); ++i )
	updateReacVelocities( g, S(), v_ );
	atot_ = 0;
	for ( vector< double >::const_iterator 
		i = v_.begin(); i != v_.end(); ++i ) {
		atot_ += *i;
	}
	atot_ *= SAFETY_FACTOR;
}

/////////////////////////////////////////////////////////////////////////
// Rate computation functions
/////////////////////////////////////////////////////////////////////////

void GssaVoxelPools::updateAllRateTerms( const vector< RateTerm* >& rates,
			   unsigned int numCoreRates )
{
	for ( unsigned int i = 0; i < rates_.size(); ++i )
		delete( rates_[i] );
	rates_.resize( rates.size() );

	for ( unsigned int i = 0; i < numCoreRates; ++i )
		rates_[i] = rates[i]->copyWithVolScaling( getVolume(), 1, 1 );
	for ( unsigned int i = numCoreRates; i < rates.size(); ++i )
		rates_[i] = rates[i]->copyWithVolScaling( getVolume(), 
						getXreacScaleSubstrates(i - numCoreRates),
						getXreacScaleProducts(i - numCoreRates ) );
}

void GssaVoxelPools::updateRateTerms( const vector< RateTerm* >& rates,
			   unsigned int numCoreRates, unsigned int index 	)
{
	// During setup or expansion of the reac system, this might be called
	// before the local rates_ term is assigned. If so, ignore.
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
/**
 * updateReacVelocities computes the velocity *v* of each reaction.
 * This is a utility function for programs like SteadyState that need
 * to analyze velocity.
 */
void GssaVoxelPools::updateReacVelocities( const GssaSystem* g, 
			const double* s, vector< double >& v ) const
{
	const KinSparseMatrix& N = g->stoich->getStoichiometryMatrix();
	assert( N.nColumns() == rates_.size() );

	vector< RateTerm* >::const_iterator i;
	v.clear();
	v.resize( rates_.size(), 0.0 );
	vector< double >::iterator j = v.begin();

	for ( i = rates_.begin(); i != rates_.end(); i++) {
		*j++ = (**i)( s );
		assert( !isnan( *( j-1 ) ) );
	}
}

double GssaVoxelPools::getReacVelocity( 
				unsigned int r, const double* s ) const
{
	return rates_[r]->operator()( s );
}

void GssaVoxelPools::setStoich( const Stoich* stoichPtr )
{
	stoichPtr_ = stoichPtr;
}

// Handle volume updates. Inherited virtual func.
void GssaVoxelPools::setVolumeAndDependencies( double vol )
{
	VoxelPoolsBase::setVolumeAndDependencies( vol );
	stoichPtr_->setupCrossSolverReacVols();
	updateAllRateTerms( stoichPtr_->getRateTerms(),
		stoichPtr_->getNumCoreRates() );
}
