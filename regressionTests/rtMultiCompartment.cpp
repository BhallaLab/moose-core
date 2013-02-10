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

void checkField( const string& path, const string& field, double value )
{
	Id id( path );
	assert( id != Id() );
	double x = Field< double >::get( id, field );
	assert( doubleEq( x, value ) );
}

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

	// R3 will use this.
	assert( gs->ode()[5].compartmentSignature_.size() == 2 );
	assert( gs->ode()[5].compartmentSignature_[0] == B );
	assert( gs->ode()[5].compartmentSignature_[1] == C );
	assert( gs->ode()[5].stoich_->getNumVarPools() == 2 );
	assert( gs->ode()[5].stoich_->getNumProxyPools() == 2 );
	assert( gs->ode()[5].stoich_->getNumRates() == 3 );

	// R4 will use this.
	assert( gs->ode()[2].compartmentSignature_.size() == 1 );
	assert( gs->ode()[2].compartmentSignature_[0] == D );
	assert( gs->ode()[2].stoich_->getNumVarPools() == 2 );
	assert( gs->ode()[2].stoich_->getNumProxyPools() == 1 );
	assert( gs->ode()[2].stoich_->getNumRates() == 3 );

	////////////////////////////////////////////////////////////////
	Id gsB( "/model/compartment_1/stoich" );
	assert( gsB != Id() );
	gs = reinterpret_cast< GslStoich* >( gsB.eref().data() );
	assert( gs->pools().size() == 1 ); // No diffusion
	assert( gs->pools()[0].size() == 5 );
	assert( gs->ode().size() == 2 ); // combos: x C
	assert( gs->coreStoich()->getNumVarPools() == 4 ); // M1, M3, M6, cplx
	assert( gs->coreStoich()->getNumProxyPools() == 1 ); // M4 on C
	assert( gs->coreStoich()->getNumRates() == 3 ); // R6, R7, (R5 on C)
	assert( gs->coreStoich()->getNumCoreRates() == 2 ); // R6, R7
	assert( gs->ode()[0].compartmentSignature_.size() == 0 );
	assert( gs->ode()[0].stoich_->getNumVarPools() == 4 );
	assert( gs->ode()[0].stoich_->getNumProxyPools() == 0 );
	assert( gs->ode()[0].stoich_->getNumRates() == 2 );

	// R5 uses this.
	assert( gs->ode()[1].compartmentSignature_.size() == 1 );
	assert( gs->ode()[1].compartmentSignature_[0] == C );
	assert( gs->ode()[1].stoich_->getNumVarPools() == 4 );
	assert( gs->ode()[1].stoich_->getNumProxyPools() == 1 );
	assert( gs->ode()[1].stoich_->getNumRates() == 3 );

	////////////////////////////////////////////////////////////////
	Id gsC( "/model/compartment_3/stoich" );
	assert( gsC != Id() );
	gs = reinterpret_cast< GslStoich* >( gsC.eref().data() );
	assert( gs->pools().size() == 1 ); // No diffusion
	assert( gs->pools()[0].size() == 2 );
	assert( gs->ode().size() == 1 ); // No combos, just core reacs.
	assert( gs->coreStoich()->getNumVarPools() == 2 ); // M1, M3, M6, cplx
	assert( gs->coreStoich()->getNumProxyPools() == 0 ); // M4 on C
	assert( gs->coreStoich()->getNumRates() == 1 ); // R8
	assert( gs->coreStoich()->getNumCoreRates() == 1 ); // R8
	assert( gs->ode()[0].compartmentSignature_.size() == 0 );
	assert( gs->ode()[0].stoich_->getNumVarPools() == 2 );
	assert( gs->ode()[0].stoich_->getNumProxyPools() == 0 );
	assert( gs->ode()[0].stoich_->getNumRates() == 1 );


	////////////////////////////////////////////////////////////////
	Id gsD( "/model/compartment_2/stoich" );
	assert( gsD != Id() );
	gs = reinterpret_cast< GslStoich* >( gsD.eref().data() );
	assert( gs->pools().size() == 1 ); // No diffusion
	assert( gs->pools()[0].size() == 2 );
	assert( gs->ode().size() == 1 ); // No combos, just core reacs.
	assert( gs->coreStoich()->getNumVarPools() == 2 ); // M1, M3, M6, cplx
	assert( gs->coreStoich()->getNumProxyPools() == 0 ); // M4 on C
	assert( gs->coreStoich()->getNumRates() == 1 ); // R8
	assert( gs->coreStoich()->getNumCoreRates() == 1 ); // R8
	assert( gs->ode()[0].compartmentSignature_.size() == 0 );
	assert( gs->ode()[0].stoich_->getNumVarPools() == 2 );
	assert( gs->ode()[0].stoich_->getNumProxyPools() == 0 );
	assert( gs->ode()[0].stoich_->getNumRates() == 1 );

	////////////////////////////////////////////////////////////////
	// Check out rates
	////////////////////////////////////////////////////////////////

	checkField( "/model/kinetics/R1", "Kf", 0.1 ); 
	checkField( "/model/kinetics/R1", "Kb", 0.1 ); 
	checkField( "/model/kinetics/R2", "Kf", 0.1 ); 
	checkField( "/model/kinetics/R2", "Kb", 0.1 ); 
	checkField( "/model/kinetics/R3", "Kf", 0.1 ); 
	checkField( "/model/kinetics/R3", "kb", 1.660572e-7 );
	// checkField( "/model/kinetics/R3", "Kb", 0.1 ); 
	checkField( "/model/kinetics/R4", "Kf", 0.1 ); 
	checkField( "/model/kinetics/R4", "Kb", 0.1 ); 
	checkField( "/model/compartment_1/R5", "Kf", 0.1 ); 
	checkField( "/model/compartment_1/R5", "Kb", 0.1 ); 
	checkField( "/model/compartment_3/R8", "Kf", 0.1 ); 
	checkField( "/model/compartment_3/R8", "Kb", 0.1 ); 
	checkField( "/model/compartment_2/R9", "Kf", 0.1 ); 
	checkField( "/model/compartment_2/R9", "Kb", 0.1 ); 
	checkField( "/model/compartment_1/M3/R6and7", "kcat", 0.1 ); 
	checkField( "/model/compartment_1/M3/R6and7", "k1", 2.767587e-7 ); 
	checkField( "/model/compartment_1/M3/R6and7", "k2", 0.4 );
	checkField( "/model/compartment_1/M3/R6and7", "Km", 0.001 - 8e-9 ); 
	////////////////////////////////////////////////////////////////
	// Check out concs
	////////////////////////////////////////////////////////////////
	checkField( "/model/kinetics/M1", "concInit", 0.001 - 1.6666e-8 ); // mM
	checkField( "/model/compartment_1/M1", "concInit", 0.001 ); // mM
	checkField( "/model/compartment_2/M1", "concInit", 0.001 ); // mM
	checkField( "/model/compartment_3/M1", "concInit", 0.001 ); // mM
	
	////////////////////////////////////////////////////////////////
	/// Should set up and check diffusion stuff.
	for ( unsigned int i = 0; i < 10; ++i )
		shell->doSetClock( i, 1.0 );
	shell->doReinit();
	shell->doStart( 100.0 );

	bool ok = SetGet2< string, string >::set(
		Id( "/model/graphs/conc1/M2.Co" ), "xplot", "check.plot", "M2.plot" );
	assert( ok );
	ok = SetGet2< string, string >::set(
		Id( "/model/graphs/conc1/M3.Co" ), "xplot", "check.plot", "M3.plot" );
	assert( ok );
	ok = SetGet2< string, string >::set(
		Id( "/model/graphs/conc1/M6.Co" ), "xplot", "check.plot", "M6.plot" );
	assert( ok );
	ok = SetGet2< string, string >::set(
		Id( "/model/graphs/conc2/M4.Co" ), "xplot", "check.plot", "M4.plot" );
	assert( ok );
	ok = SetGet2< string, string >::set(
		Id( "/model/graphs/conc2/M5.Co" ), "xplot", "check.plot", "M5.plot" );
	assert( ok );
	////////////////////////////////////////////////////////////////

	shell->doDelete( model );
	cout << "." << flush;
}
