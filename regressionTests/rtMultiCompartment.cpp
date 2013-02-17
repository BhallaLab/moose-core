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

void checkJunction( const string& path, Id c1, Id c2 )
{
	ObjId id( path );
	assert( !( id == ObjId::bad() ) );
	// unsigned int nr = Field< unsigned int >::get( id, "numReacs" );
	unsigned int ndm = Field< unsigned int >::get( id, "numDiffMols" );
	unsigned int nme = Field< unsigned int >::get( id, "numMeshEntries" );
	Id myCompartment = Field< Id >::get( id, "myCompartment" );
	Id otherCompartment = Field< Id >::get( id, "otherCompartment" );

	// assert( nr == 1 );
	assert( ndm == 0 );
	assert( nme == 1 );

	assert( myCompartment == c1 );
	assert( otherCompartment == c2 );
}

static void checkGraphs()
{
	vector< string > plotnames;
	plotnames.push_back( "conc1/M2.Co" );
	plotnames.push_back( "conc1/M3.Co" );
	plotnames.push_back( "conc1/M6.Co" );
	plotnames.push_back( "conc2/M4.Co" );
	plotnames.push_back( "conc2/M5.Co" );
	const double TOLERANCE = 2e-3;

	for ( vector< string >::iterator 
					i = plotnames.begin(); i != plotnames.end(); ++i )
	{
		string fullName = "/model/graphs/" + *i;
		string molName = i->substr( 6, 8 );
		string xplotName = molName + ".Co";
		Id plotId( fullName );
		assert( plotId != Id() );
		bool ok = SetGet2< string, string >::set(
			plotId, "xplot", "check.plot", xplotName );
		assert( ok );
		// Check the output values
		ok = SetGet2< double, double >::set( 
						plotId, "linearTransform", 1000, 0);
		assert( ok );
		ok = SetGet3< string, string, string >::set( 
			plotId, "compareXplot", "multicompt.plot", 
			"/graphs/" + *i, "rmsr" );
		assert( ok );
		double val = Field< double >::get( plotId, "outputValue" );
		assert( val >= 0 && val < TOLERANCE );
	}
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
	assert( gs->pools().size() == 4 ); // No diffusion, but it does this?
	assert( gs->pools()[0].size() == 5 );
	assert( gs->ode().size() == 8 ); // combos: x 1 2 3 12 13 23 123
	assert( gs->pools()[0].getSolver() == 7 );
	// const_cast< VoxelPools& >( gs->pools()[0] ).setSolver( 7 );
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

	// The combo will use this
	assert( gs->ode()[7].compartmentSignature_.size() == 3 );
	assert( gs->ode()[7].compartmentSignature_[0] == B );
	assert( gs->ode()[7].compartmentSignature_[1] == D );
	assert( gs->ode()[7].compartmentSignature_[2] == C );

	////////////////////////////////////////////////////////////////
	Id gsB( "/model/compartment_1/stoich" );
	assert( gsB != Id() );
	gs = reinterpret_cast< GslStoich* >( gsB.eref().data() );
	assert( gs->pools().size() == 2 ); // No diffusion, but goes to A.
	assert( gs->pools()[0].size() == 5 );
	assert( gs->ode().size() == 2 ); // combos: x C
	assert( gs->pools()[0].getSolver() == 1 );
	// const_cast< VoxelPools& >( gs->pools()[0] ).setSolver( 1 );
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
	assert( gs->pools()[0].getSolver() == 0 );
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
	assert( gs->pools()[0].getSolver() == 0 );
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
	// To fix: It comes to 300.  I don't see why.
	// One target is 5x vol. Other is 3x vol. 
	// The scaling for uM to mM is 1000. That would have given 100.
	// Then we need to go 3x faster to balance the 3x vol case. I would
	// have preferred to just stick to the 
	// Or should I take kb as the starting point. 
	// 1.66e-7 is the rate in 1/#.sec. 
	// Convert to 1e-15 m^3 by multiplying by NA * vol = 6e8
	// We get 100. That should have been it.
	// Instead the vol that the system uses is 3e-15, from compt B.
	//
	checkField( "/model/kinetics/R4", "Kf", 0.1 ); 
	checkField( "/model/kinetics/R4", "Kb", 0.1 ); 
	checkField( "/model/kinetics/R4", "kf", 0.1 ); 
	checkField( "/model/kinetics/R4", "kb", 0.1 ); 
	checkField( "/model/compartment_1/R5", "Kf", 0.1 ); 
	checkField( "/model/compartment_1/R5", "Kb", 0.1 ); 
	checkField( "/model/compartment_1/R5", "kf", 0.1 ); 
	checkField( "/model/compartment_1/R5", "kb", 0.1 ); 
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
	checkField( "/model/kinetics/M1A", "concInit", 0.001 - 1.6666e-8 ); // mM
	checkField( "/model/compartment_1/M1B", "concInit", 0.001 ); // mM
	checkField( "/model/compartment_3/M1C", "concInit", 0.001 ); // mM
	checkField( "/model/compartment_2/M1D", "concInit", 0.001 ); // mM
	
	////////////////////////////////////////////////////////////////
	// Check out junctions
	////////////////////////////////////////////////////////////////
	
	unsigned int nj = 0;
	nj = Field< unsigned int >::get( gsA, "num_junction" );
	assert( nj == 3 );
	nj = Field< unsigned int >::get( gsB, "num_junction" );
	assert( nj == 2 );
	nj = Field< unsigned int >::get( gsC, "num_junction" );
	assert( nj == 2 );
	nj = Field< unsigned int >::get( gsD, "num_junction" );
	assert( nj == 1 );
	ObjId oiA0( "/model/kinetics/stoich/junction[0]" );
	ObjId oiA1( "/model/kinetics/stoich/junction[1]" );
	ObjId oiA2( "/model/kinetics/stoich/junction[2]" );

	ObjId oi0( "/model/compartment_1/stoich/junction[0]" );
	ObjId oi1( "/model/compartment_1/stoich/junction[1]" );
	assert( !( oi0 == ObjId::bad() ) );
	assert( !( oi1 == ObjId::bad() ) );
	// SolverJunction* j0 = reinterpret_cast< SolverJunction* >( oi0.data() );
	// SolverJunction* j1 = reinterpret_cast< SolverJunction* >( oi1.data() );

	checkJunction( "/model/kinetics/stoich/junction[0]", A, B );
	checkJunction( "/model/kinetics/stoich/junction[1]", A, D );
	checkJunction( "/model/kinetics/stoich/junction[2]", A, C );
	checkJunction( "/model/compartment_1/stoich/junction[0]", B, A );
	checkJunction( "/model/compartment_1/stoich/junction[1]", B, C );
	checkJunction( "/model/compartment_3/stoich/junction[0]", C, A );
	checkJunction( "/model/compartment_3/stoich/junction[1]", C, B );
	checkJunction( "/model/compartment_2/stoich/junction[0]", D, A );

	////////////////////////////////////////////////////////////////
	/// Should set up and check diffusion stuff.
	for ( unsigned int i = 0; i < 10; ++i )
		shell->doSetClock( i, 0.1 );
	shell->doSetClock( 8, 1 );
	shell->doReinit();
	shell->doStart( 100.0 );

	checkGraphs();
	////////////////////////////////////////////////////////////////

	shell->doDelete( model );
	cout << "." << flush;
}


/**
 * The simulated geometry is:
 *                    D
 *                    D
 *                 BBBA
 *                 CCCCC
 *  Here, A is at (0,0,0) to (10,10,10) microns.
 *  B is then (-30,0,0) to (0,10,10) microns
 *  C is (-30,-10,0) to (20,0,10) microns
 *  D is (0,10,0) to (10,30,10) microns.
 */
void rtTestMultiCompartmentReacDiff()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Shell::cleanSimulation();

	Id model = shell->doLoadModel( 
					"multicompt_reacdiff.g", "/model", "multigsl" );

	Id A( "/model/kinetics" );
	assert( A != Id() );
	double sizeA = Field< double >::get( A, "size" );

	Id B( "/model/compartment_1" );
	assert( B != Id() );
	vector< double > coords( 9, 0 );
	coords[0] = -30e-6;	coords[1] = 0; 		coords[2] = 0;
	coords[3] = 0; 		coords[4] = 10e-6;	coords[5] = 10e-6;
	coords[6] = 		coords[7] = 		coords[8] = 10e-6;
	Field< vector< double > >::set( B, "coords", coords );
	double sizeB = Field< double >::get( B, "size" );

	Id D( "/model/compartment_2" ); // order is scrambled.
	assert( D != Id() );
	coords[0] = 0;		coords[1] = 10e-6;	coords[2] = 0;
	coords[3] = 10e-6; 	coords[4] = 30e-6;	coords[5] = 10e-6;
	coords[6] = 		coords[7] = 		coords[8] = 10e-6;
	Field< vector< double > >::set( D, "coords", coords );
	double sizeD = Field< double >::get( D, "size" );

	Id C( "/model/compartment_3" );
	assert( C != Id() );
	coords[0] = -30e-6;	coords[1] = -10e-6;	coords[2] = 0;
	coords[3] = 20e-6;	coords[4] = 0;		coords[5] = 10e-6;
	coords[6] = 		coords[7] = 		coords[8] = 10e-6;
	Field< vector< double > >::set( C, "coords", coords );
	double sizeC = Field< double >::get( C, "size" );

	assert( doubleEq( sizeA, 1e-15 ) );
	assert( doubleEq( sizeB, 3e-15 ) );
	assert( doubleEq( sizeC, 5e-15 ) );
	assert( doubleEq( sizeD, 2e-15 ) );
	////////////////////////////////////////////////////////////////

	shell->doDelete( model );
	cout << "." << flush;
}
