/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <fstream>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include "header.h"
#include "Shell.h"
#include "LoadModels.h"
#include "../mesh/VoxelJunction.h"
#include "../kinetics/PoolBase.h"
#include "../ksolve/SolverJunction.h"
#include "../ksolve/SolverBase.h"
#include "../ksolve/VoxelPools.h"
#include "../ksolve/RateTerm.h"
#include "../kinetics/FuncTerm.h"
#include "SparseMatrix.h"
#include "../ksolve/KinSparseMatrix.h"
#include "../ksolve/StoichCore.h"
#include "../ksolve/OdeSystem.h"
#include "../ksolve/GslStoich.h"


void rtTestMultiCompartmentReaction()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Shell::cleanSimulation();

	Id model = shell->doLoadModel( 
					"multicompt_reac.g", "/model", "multigsl" );
	// SetGet1< string >::set( model, "buildMultiCompartment", "rk5" );
	Id A( "/model/kinetics" );
	assert( A != Id() );
	double sizeA = Field< double >::get( A, "size" );

	Id B( "/model/compartment_1" );
	assert( B != Id() );
	double sizeB = Field< double >::get( B, "size" );

	Id D( "/model/compartment_2" ); // order is scrambled.
	assert( D != Id() );
	double sizeD = Field< double >::get( D, "size" );

	Id C( "/model/compartment_3" );
	assert( C != Id() );
	double sizeC = Field< double >::get( C, "size" );

	assert( doubleEq( sizeA, 1e-15 ) );
	assert( doubleEq( sizeB, 3e-15 ) );
	assert( doubleEq( sizeC, 5e-15 ) );
	assert( doubleEq( sizeD, 2e-15 ) );

	Id gsA( "/model/kinetics/stoich" );
	assert( gsA != Id() );
	GslStoich* gs = reinterpret_cast< GslStoich* >( gsA.eref().data() );
	assert( gs->pools().size() == 1 ); // No diffusion
	assert( gs->pools()[0].size() == 5 );
	assert( gs->ode().size() == 8 ); // combos: x 1 2 3 12 13 23 123
	assert( gs->coreStoich()->getNumVarPools() == 2 );
	assert( gs->coreStoich()->getNumProxyPools() == 3 );
	assert( gs->coreStoich()->getNumRates() == 4 );
	assert( gs->coreStoich()->getNumCoreRates() == 2 );
	assert( gs->ode()[0].compartmentSignature_.size() == 0 );
	assert( gs->ode()[0].stoich_->getNumVarPools() == 2 );
	assert( gs->ode()[0].stoich_->getNumProxyPools() == 0 );
	assert( gs->ode()[0].stoich_->getNumRates() == 2 );

	assert( gs->ode()[5].compartmentSignature_.size() == 2 );
	assert( gs->ode()[5].compartmentSignature_[0] == B );
	assert( gs->ode()[5].compartmentSignature_[1] == C );
	assert( gs->ode()[5].stoich_->getNumVarPools() == 2 );
	assert( gs->ode()[5].stoich_->getNumProxyPools() == 2 );
	assert( gs->ode()[5].stoich_->getNumRates() == 3 );
	/// Should set up and check diffusion stuff.


	shell->doDelete( model );
	cout << "." << flush;
}
