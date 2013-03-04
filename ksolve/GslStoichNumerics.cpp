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
#include "SolverBase.h"
#include "VoxelPools.h"
#include "OdeSystem.h"
#include "../shell/Shell.h"
#include "../mesh/MeshEntry.h"
#include "../mesh/Boundary.h"
#include "../mesh/ChemMesh.h"
#include "GslStoich.h"

typedef vector< unsigned int >::const_iterator VCI;

///////////////////////////////////////////////////////////////////////////
// Junction operations
///////////////////////////////////////////////////////////////////////////

/*
void GslStoich::updateJunctionDiffusion( unsigned int meshIndex, 
	double diffScale, const vector< unsigned int >& diffTerms, double* v,
    double dt	)
{
	unsigned int j = 0;
	for ( vector< unsigned int >::const_iterator i = diffTerms.begin();
					i != diffTerms.end(); ++i ) {
		double temp = stoich_->getDiffConst( *i ) * S( meshIndex )[*i] * 
				diffScale;
		v[j++] += temp;
		y_[meshIndex][*i] -= temp * dt;
		// varS( meshIndex )[*i] -= temp * dt;
	}
}
*/

unsigned int GslStoich::fillReactionDelta( 
		const SolverJunction* j, 
		const vector< vector< double > > & lastS, 
		double* yprime ) const
{
	double* orig = yprime;
	// For reactions across junctions, we keep within the local
	// sendMeshIndex. Send the delta from the proxy pool identified
	// by remoteReacPools.
	// unsigned int numVarPools = coreStoich()->getNumVarPools();
	for ( VCI k = 
		j->sendMeshIndex().begin(); k != j->sendMeshIndex().end(); ++k )
	{
		assert( *k < lastS.size() );
		const vector< double >& ts = lastS[ *k ];
		const vector< double >& ty = y_[ *k ];
		for ( VCI q = j->remoteReacPools().begin(); 
						q != j->remoteReacPools().end(); ++q )
	   	{
			assert( *q < ts.size() );
			assert( *q < ty.size() );
			*yprime++ = ty[ *q ] - ts[ *q ];
		}
	}
	return yprime - orig; // Number of terms computed.
}

void GslStoich::fillDiffusionDelta( 
		const SolverJunction* j, 
		const vector< vector< double > > & lastS, 
		double* yprime ) const
{
	unsigned int numVarPools = coreStoich()->getNumVarPools();
	// For the diffusion across junctions, we let the regular diffusion
	// calculations figure new mol# for the abutting voxels. We send
	// the change (delta) in mol# of these abutting voxels over to the 
	// corresponding core pools of the abutting solver.
	for ( VCI k = 
		j->abutMeshIndex().begin(); k != j->abutMeshIndex().end(); ++k )
	{
		assert( *k < lastS.size() );
		const vector< double >& ts = lastS[ *k ];
		const vector< double >& ty = y_[ *k ];
		for ( VCI q = 
			j->diffTerms().begin(); q != j->diffTerms().end(); ++q )
	   	{
			assert( *q < numVarPools );
			*yprime++ = ty[ *q ] - ts[ *q ];
		}
	}
}

/**
 * This function calculates cross-junction rates and sends out as msgs.
 * Since we need specific messages between solvers, we handle them through
 * FieldElements which are one per junction. 
 *
 * This function should only be called on the master solver.
 *
 * The message sends the Delta information to the follower solver,
 * where it is handled by
 * junctionPoolDelta/handleJunctionPoolDelta.
 */
void GslStoich::vUpdateJunction( const Eref& e, 
				const vector< vector< double > > & lastS,
				unsigned int threadNum, double dt )
{
	Id junction( e.id().value() + 1 );
	assert( junction.element()->cinfo()->isA( "SolverJunction" ) );
	for ( unsigned int i = 0; i < getNumJunctions(); ++i ) {
		SolverJunction* j = getJunction(i);
		unsigned int numReac = j->remoteReacPools().size();
		unsigned int numDiff = j->diffTerms().size();
		vector< double > v( 
			numReac * j->sendMeshIndex().size() + 
			numDiff * j->abutMeshIndex().size(), 
			0 );
		double* yprime = &v[0];

		unsigned int offset = fillReactionDelta( j, lastS, yprime );
		if ( numDiff > 0 ) {
			fillDiffusionDelta( j, lastS, yprime + offset );
		}

		Eref je( junction.element(), i );
		// Each Junction FieldElement connects up to precisely one target.
		junctionPoolDeltaFinfo()->send( je, threadNum, v );
	}
}

/**
 * Handles incoming cross-border rates. Just adds onto y_ matrix, using
 * Forward Euler.
 */
void GslStoich::vHandleJunctionPoolDelta( unsigned int fieldIndex,
	   	const vector< double >& v )
{
	assert( fieldIndex < getNumJunctions() );
	const SolverJunction* j = getJunction( fieldIndex );
	assert( j );
	j->incrementTargets( y_, v );
	// Just to check if it makes any difference
	for ( unsigned int i = 0; i < y_.size(); ++i ) {
		double* s = pools_[i].varS();
		for ( vector< double >::const_iterator 
				j = y_[i].begin(); j != y_[i].end(); ++j )
			*s++ = *j;
	}
}

void GslStoich::vHandleJunctionPoolNum( unsigned int fieldIndex,
	   	const vector< double >& v )
{
	assert( fieldIndex < getNumJunctions() );
	const SolverJunction* j = getJunction( fieldIndex );
	assert( j );
	unsigned int size = 
			j->abutPoolIndex().size() * j->abutMeshIndex().size() 
			+ j->sendMeshIndex().size() * j->remoteReacPools().size();
	assert( v.size() == size );
	vector< double >::const_iterator vptr = v.begin();
	for ( VCI k = j->sendMeshIndex().begin(); 
			k != j->sendMeshIndex().end(); ++k )
	{
		double* s = pools_[*k].varS();
		for ( VCI p = j->remoteReacPools().begin(); 
				p != j->remoteReacPools().end(); ++p )
	  	{
				y_[*k][*p] = *vptr;
				s[*p] = *vptr++;
		}
	}
	for ( VCI k = j->abutMeshIndex().begin(); 
			k != j->abutMeshIndex().end(); ++k )
	{
		double* s = pools_[*k].varS();
		for ( VCI p = j->abutPoolIndex().begin(); 
				p != j->abutPoolIndex().end(); ++p )
	  	{
				y_[*k][*p] = *vptr;
				s[*p] = *vptr++;
		}
	}
}

///////////////////////////////////////////////////
// Reinit and process.
///////////////////////////////////////////////////

void GslStoich::reinit( const Eref& e, ProcPtr info )
{
	if ( !isInitialized_ )
			return;
	/*
	if ( junctionsNotReady_ ) {
			// This is a little dangerous, as it relies on the stencils
			// having been reset.
		reconfigureAllJunctions( e, 0 );
		junctionsNotReady_ = false;
	}
	*/
	// unsigned int nPools = coreStoich()->getNumVarPools() + coreStoich()->getNumProxyPools();
	for ( unsigned int i = 0; i < pools_.size(); ++i ) {
		VoxelPools& p = pools_[i];
		p.reinit();
		unsigned int nPools = p.size();
		memcpy( &(y_[i][0]), p.Sinit(), nPools * sizeof( double ) );
		assert( y_[i].size() == p.size() );
		// assert( y_[i].size() <= nPools );
		ode_[ p.getSolver() ].stoich_->updateFuncs( p.varS(), 0 );
	}
	
	for ( vector< OdeSystem >::iterator
					i = ode_.begin(); i != ode_.end(); ++i ) {
			// Assume stoich returns includes off-solverpools if needed.
		i->reinit( this,
						GslStoich::gslFunc, 
						i->stoich_->getNumVarPools() +
						i->stoich_->getNumProxyPools(), 
						absAccuracy_, relAccuracy_ ); 
	}
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
	
	vector< vector< double > > lastS = y_;
	double nextt = info->currTime + info->dt;
	// Hack till we sort out threadData
	for ( currMeshEntry_ = 0; 
					currMeshEntry_ < localMeshEntries_.size(); ++currMeshEntry_ ) {
		double t = info->currTime;
		OdeSystem& os = ode_[ pools_[currMeshEntry_ ].getSolver() ];
		while ( t < nextt ) {
			int status = gsl_odeiv_evolve_apply (
				os.gslEvolve_, os.gslControl_, os.gslStep_, &os.gslSys_, 
				&t, nextt,
				&internalStepSize_, &y_[currMeshEntry_][0] );
			if ( status != GSL_SUCCESS )
				break;

		}
	}
	// if ( diffusionMesh_ && diffusionMesh_->innerGetNumEntries() > 1 )
	if ( diffusionMesh_ && pools_.size() > 1 )
		updateDiffusion( lastS, y_, info->dt );
	if ( getNumJunctions() > 0 )
		vUpdateJunction( e, lastS, info->threadIndexInGroup, info->dt );
#endif // USE_GSL
	// stoich_->clearFlux( e.index().value(), info->threadIndexInGroup );
}

/// Called on the init msg during reinit.
void GslStoich::initReinit( const Eref& e, ProcPtr info )
{
}

/**
 * Here we just send out a set of pools to the other solver.
 * Look up handlePools to see what happens when this arrives.
 */
void GslStoich::init( const Eref& e, ProcPtr info )
{

	if ( !isInitialized_ )
			return;
	Id junction( e.id().value() + 1 );
	assert( junction.element()->cinfo()->isA( "SolverJunction" ) );
	for ( unsigned int i = 0; i < getNumJunctions(); ++i ) {
		SolverJunction* j = getJunction(i);
		vector< double > v;
		unsigned int size = 
			( j->localReacPools().size() + j->sendPoolIndex().size() )
		   	* j->sendMeshIndex().size();
		v.reserve( size );
		// Fill in reaction pool nums.
		for ( VCI k = j->sendMeshIndex().begin(); 
				k != j->sendMeshIndex().end(); ++k )
	   	{
			const double* s = pools_[*k].S();
			// const double* s = S( *k );
			for ( VCI p = j->localReacPools().begin(); 
					p != j->localReacPools().end(); ++p ) 
				v.push_back( s[*p] );
		}
		// Fill in diffusion pool nums.
		for ( VCI k = j->sendMeshIndex().begin(); 
				k != j->sendMeshIndex().end(); ++k )
	   	{
			const double* s = pools_[*k].S();
			// const double* s = S( *k );
			for ( VCI p = j->sendPoolIndex().begin(); 
					p != j->sendPoolIndex().end(); ++p ) 
				v.push_back( s[*p] );
		}
		assert( v.size() == size );
		Eref je( junction.element(), i );
		// Each Junction FieldElement connects up to precisely one target.
		junctionPoolNumFinfo()->send( je, info->threadIndexInGroup, v );
	}
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

void GslStoich::updateDiffusion( 
	vector< vector< double > >& lastS,
	vector< vector< double > >& y,
			   	double dt )
{
	const double *adx; // each entry is diffn_XA/diffn_length
	const unsigned int* colIndex;

	assert( lastS.size() == y.size() );
	assert( lastS.size() == pools_.size() );

	// We only worry about diffusion of the core pools, not proxy pools.
	unsigned int numCorePools = coreStoich()->getNumVarPools();

	// Get value at midpoint in time.
	for ( unsigned int me = 0; me < pools_.size(); ++me ) {
		assert( lastS[me].size() >= numCorePools );
		assert( y[me].size() >= numCorePools );
		for ( unsigned int i = 0; i < numCorePools; ++i ) {
			lastS[me][i] = ( lastS[me][i] + y[me][i] ) / 2.0;
		}
	}

	// Simple forward Euler hack here. Later do a Crank Nicolson in
	// alternating dimensions, as suggested by NumRec.
	for ( unsigned int me = 0; me < pools_.size(); ++me ) {
		unsigned int numInRow = 
			diffusionMesh_->getStencil( me, &adx, &colIndex);
		double vSelf = diffusionMesh_->getMeshEntrySize( me );
		const double* sSelf = &(lastS[ me ][0]);
		StoichCore* stoich = ode_[ pools_[me].getSolver() ].stoich_;
		/* Don't use xa anymore, it is folded into the stencil.
		vector< double > xa;
		if ( me < numMeshEntries() ) // Local ones
			xa = diffusionMesh_->getDiffusionArea( me);
		else
			; // Fill from junction, later.
		assert ( xa.size() == numInRow );
			*/
		for ( unsigned int i = 0; i < numInRow; ++i ) {
			unsigned int other = colIndex[i];

			// Get all concs at the other meshEntry
			const double* sOther = &( lastS[other][0] ); 
			double vOther = diffusionMesh_->extendedMeshEntrySize( other );
			double scale = dt * adx[i] ;
			assert( vOther > 0 );
			assert( vSelf > 0 );
		
			for ( unsigned int j = 0; j < numCorePools; ++j )
				y[me][j] += stoich->getDiffConst(j) * scale * 
						( sOther[j]/vOther - sSelf[j]/vSelf );
		}
		double* s = pools_[me].varS();
		for ( unsigned int j = 0; j < numCorePools; ++j )
			s[j] = y[me][j];
	}
}

int GslStoich::innerGslFunc( double t, const double* y, double* yprime ) 
{
	double* varS = pools_[currMeshEntry_].varS();
	unsigned int totVarPools = coreStoich()->getNumVarPools() + 
			coreStoich()->getNumProxyPools();
	// unsigned int totNumPools = pools_[ currMeshEntry_ ].size();
	// Copy the y array into the S_ vector.
	memcpy( varS, y, totVarPools * sizeof( double ) );
	StoichCore* stoich = ode_[ pools_[currMeshEntry_].getSolver() ].stoich_;

	stoich->updateFuncs( varS, t );

	stoich->updateRates( varS, yprime );
	
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
