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
#include "../mesh/MeshEntry.h"
#include "../mesh/Boundary.h"
#include "../mesh/Stencil.h"
#include "../mesh/ChemMesh.h"
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
#include "../shell/Wildcard.h"


static void makeReacDiffGraphs( const string& poolPath )
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );
	
	Id pool( poolPath );
	assert( pool != Id() );
	unsigned int n = Field< unsigned int >::get( pool, "linearSize" );
	dims[0] = n;
	Id conc1( "/model/graphs/conc1" );
	assert( conc1 != Id() );
	string tail = pool.element()->getName();
	string temp = poolPath.substr( poolPath.length() - 4, 1 );
	if ( temp == "s" ) tail = tail + "_A";
	else if ( temp == "1" ) tail = tail + "_B";
	else if ( temp == "3" ) tail = tail + "_C";
	else if ( temp == "2" ) tail = tail + "_D";
	else assert( false );
	Id plot = shell->doCreate( "Table", conc1, tail + ".conc", dims );
	assert( plot != Id() );
	shell->doAddMsg( "OneToOne", plot, "requestData", pool, "get_conc" );
}

static void dumpReacDiffGraphs()
{
	vector< Id > plots;
	wildcardFind( "/model/graphs/conc1/#.conc", plots );
	assert( plots.size() == 9 );
	for ( vector< Id >::iterator i = plots.begin(); i != plots.end(); ++i )
	{
		for ( unsigned int j = 0; 
			j < i->element()->dataHandler()->localEntries(); ++j ) {
			ObjId oi( *i, j );
			stringstream ss;
			ss << ( i->element()->getName() ) << "_" << j;	
			bool ok = SetGet2< string, string >::set(
				oi, "xplot", "check.plot", ss.str() );
			assert( ok );
		}
	}
}

void checkField( const string& path, const string& field, double value )
{
	Id id( path );
	assert( id != Id() );
	double x = Field< double >::get( id, field );
	assert( doubleApprox( x, value ) );
}

void checkJunction( const string& path, Id c1, Id c2, bool isDiffusive )
{
	ObjId id( path );
	assert( !( id == ObjId::bad() ) );
	// unsigned int nr = Field< unsigned int >::get( id, "numReacs" );
	unsigned int ndm = Field< unsigned int >::get( id, "numDiffMols" );
	//unsigned int nme = Field< unsigned int >::get( id, "numMeshEntries" );
	Id myCompartment = Field< Id >::get( id, "myCompartment" );
	Id otherCompartment = Field< Id >::get( id, "otherCompartment" );

	/*
	 unsigned int myNumMesh = Field< unsigned int >::get( myCompartment, "num_mesh" );
	unsigned int otherNumMesh = 
			Field< unsigned int >::get( otherCompartment, "num_mesh" );
	unsigned int refNumMesh = 
			( myNumMesh < otherNumMesh ) ? myNumMesh : otherNumMesh;
			*/

	// assert( nr == 1 );
	assert( ndm == static_cast< unsigned int >( isDiffusive ) );
	// assert( nme == refNumMesh );

	assert( myCompartment == c1 );
	assert( otherCompartment == c2 );
}

void checkAllJunctions( bool isDiffusive )
{
	Id A( "/model/kinetics" ); assert( A != Id() );
	Id B( "/model/compartment_1" ); assert( B != Id() );
	Id D( "/model/compartment_2" ); // order is scrambled.
	assert( D != Id() );
	Id C( "/model/compartment_3" ); assert( C != Id() );
	Id gsA( "/model/kinetics/stoich" ); assert( gsA != Id() );
	Id gsB( "/model/compartment_1/stoich" ); assert( gsB != Id() );
	Id gsC( "/model/compartment_3/stoich" ); assert( gsC != Id() );
	Id gsD( "/model/compartment_2/stoich" ); assert( gsD != Id() );

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

	checkJunction( "/model/kinetics/stoich/junction[0]", A, B, isDiffusive );
	checkJunction( "/model/kinetics/stoich/junction[1]", A, D, isDiffusive );
	checkJunction( "/model/kinetics/stoich/junction[2]", A, C, isDiffusive );
	checkJunction( "/model/compartment_1/stoich/junction[0]", B, A, isDiffusive );
	checkJunction( "/model/compartment_1/stoich/junction[1]", B, C, isDiffusive );
	checkJunction( "/model/compartment_3/stoich/junction[0]", C, A, isDiffusive );
	checkJunction( "/model/compartment_3/stoich/junction[1]", C, B, isDiffusive );
	checkJunction( "/model/compartment_2/stoich/junction[0]", D, A, isDiffusive );
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

	checkField( "/model/kinetics/R3", "kb", 1.660572e-7 );

	Id gsA( "/model/kinetics/stoich" );
	assert( gsA != Id() );
	GslStoich* gs = reinterpret_cast< GslStoich* >( gsA.eref().data() );
	assert( gs->pools().size() == 1 ); // No diffusion, but it does this?
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
	assert( gs->pools().size() == 1 ); // No diffusion, but goes to A.
	assert( gs->pools()[0].size() == 5 ); // 3 mols, 1 enz cplx, diff to C
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
	checkField( "/model/kinetics/R3", "Kb", 0.1 * 5.0 * 3.0 * 1000.0 ); 
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
	checkField( "/model/kinetics/R4", "kf", 0.1 ); 
	checkField( "/model/kinetics/R4", "kb", 0.1 ); 
	checkField( "/model/kinetics/R4", "Kf", 0.1 ); 
	checkField( "/model/kinetics/R4", "Kb", 0.2 ); 
	checkField( "/model/compartment_1/R5", "kf", 0.1 ); 
	checkField( "/model/compartment_1/R5", "kb", 0.1 ); 
	checkField( "/model/compartment_1/R5", "Kf", 0.1 ); 
	checkField( "/model/compartment_1/R5", "Kb", 0.1 * 5.0/3.0 ); 
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
	checkAllJunctions( false );

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

void checkReac( const string& name, 
				double Kf, double Kb, double kf, double kb )
{
	ObjId R( name );
	assert( !( R == ObjId::bad() ) );
	double Kf_ = Field< double >::get( R, "Kf" );
	double Kb_ = Field< double >::get( R, "Kb" );
	double kf_ = Field< double >::get( R, "kf" );
	double kb_ = Field< double >::get( R, "kb" );
	assert( doubleEq( Kf, Kf_ ) );
	assert( doubleEq( Kb, Kb_) );
	assert( doubleEq( kf, kf_ ) );
	assert( doubleEq( kb, kb_ ) );
}

void checkScaledRates()
{
	checkReac( "/model/kinetics/R1", 0.1, 0.1, 0.1, 0.1 );
	checkReac( "/model/kinetics/R2", 0.1, 0.1, 0.1, 0.1 );
	checkReac( "/model/kinetics/R3", 0.1, 1500.03, 0.1, 
					1500.03 / (NA * 1e-15) );
	checkReac( "/model/kinetics/R4", 0.1, 0.2, 0.1, 0.2 );
	checkReac( "/model/compartment_1/R5", 0.1, 0.1 * 5.0 / 3.0, 
					0.1, 0.1 * 5.0 / 3.0 );
	checkReac( "/model/compartment_3/R8", 0.1, 0.1, 0.1, 0.1 );
	checkReac( "/model/compartment_2/R9", 0.1, 0.1, 0.1, 0.1 );

	ObjId E( "/model/compartment_1/M3/R6and7" );
	assert( !( E == ObjId::bad() ) );
	double Km = Field< double >::get( E, "Km" );
	double kcat = Field< double >::get( E, "kcat" );
	double ratio = Field< double >::get( E, "ratio" );
	double k1 = Field< double >::get( E, "k1" );
	double k2 = Field< double >::get( E, "k2" );
	double k3 = Field< double >::get( E, "k3" );
	assert( doubleApprox( Km , 1.0e-3 ) );
	assert( doubleEq( kcat , 0.1 ) );
	assert( doubleEq( ratio , 4.0 ) );
	assert( doubleApprox( k1 , 8.333333333333e-7 * 6e23/NA) );
	assert( doubleEq( k2 , 0.4 ) );
	assert( doubleEq( k3 , 0.1 ) );
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
	double DT = 0.1;
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Shell::cleanSimulation();

	Id model = shell->doLoadModel( 
					"multicompt_reacdiff.g", "/model", "multigsl" );

	Id A( "/model/kinetics" );
	assert( A != Id() );
	double sizeA = Field< double >::get( A, "size" );
	unsigned int numA = Field< unsigned int >::get( A, "num_mesh" );

	Id B( "/model/compartment_1" );
	assert( B != Id() );
	vector< double > coords( 9, 0 );
	coords[0] = -30e-6;	coords[1] = 0; 		coords[2] = 0;
	coords[3] = 0; 		coords[4] = 10e-6;	coords[5] = 10e-6;
	coords[6] = 		coords[7] = 		coords[8] = 10e-6;
	Field< vector< double > >::set( B, "coords", coords );
	double sizeB = Field< double >::get( B, "size" );
	unsigned int numB = Field< unsigned int >::get( B, "num_mesh" );

	Id D( "/model/compartment_2" ); // order is scrambled.
	assert( D != Id() );
	coords[0] = 0;		coords[1] = 10e-6;	coords[2] = 0;
	coords[3] = 10e-6; 	coords[4] = 30e-6;	coords[5] = 10e-6;
	coords[6] = 		coords[7] = 		coords[8] = 10e-6;
	Field< vector< double > >::set( D, "coords", coords );
	double sizeD = Field< double >::get( D, "size" );
	unsigned int numD = Field< unsigned int >::get( D, "num_mesh" );

	Id C( "/model/compartment_3" );
	assert( C != Id() );
	coords[0] = -30e-6;	coords[1] = -10e-6;	coords[2] = 0;
	coords[3] = 20e-6;	coords[4] = 0;		coords[5] = 10e-6;
	coords[6] = 		coords[7] = 		coords[8] = 10e-6;
	Field< vector< double > >::set( C, "coords", coords );
	double sizeC = Field< double >::get( C, "size" );
	unsigned int numC = Field< unsigned int >::get( C, "num_mesh" );

	assert( doubleEq( sizeA, 1e-15 ) );
	assert( doubleEq( sizeB, 3e-15 ) );
	assert( doubleEq( sizeC, 5e-15 ) );
	assert( doubleEq( sizeD, 2e-15 ) );
	assert( numA == 1 );
	assert( numB == 3 );
	assert( numC == 5 );
	assert( numD == 2 );

	Field< bool >::set( A, "alwaysDiffuse", false );
	Field< bool >::set( B, "alwaysDiffuse", false );
	Field< bool >::set( C, "alwaysDiffuse", false );
	Field< bool >::set( D, "alwaysDiffuse", false );
	// This should get all the stoichs to reconfigure their junctions, 
	// which was pending since the meshes have been redone.
	SetGet0::set( model, "rebuild" );
	////////////////////////////////////////////////////////////////
	// Check out rates
	////////////////////////////////////////////////////////////////
	checkScaledRates();
	
	////////////////////////////////////////////////////////////////
	// Check out junctions
	////////////////////////////////////////////////////////////////
	checkAllJunctions( true );
	ObjId jid0( "/model/kinetics/stoich/junction[0]" );
	assert( !( jid0 == ObjId::bad() ) );
	SolverJunction* j = reinterpret_cast< SolverJunction* >( jid0.data() );
	assert( j->sendMeshIndex().size() == 1 ); // Sending data from 1 voxel
	assert( j->sendMeshIndex()[0] == 0 );

	ObjId jid1( "/model/kinetics/stoich/junction[1]" );
	assert( !( jid1 == ObjId::bad() ) );
	j = reinterpret_cast< SolverJunction* >( jid1.data() );
	assert( j->sendMeshIndex().size() == 1 ); // Sending data from 1 voxel
	assert( j->sendMeshIndex()[0] == 0 );

	ObjId jid2( "/model/kinetics/stoich/junction[2]" );
	assert( !( jid2 == ObjId::bad() ) );
	j = reinterpret_cast< SolverJunction* >( jid2.data() );
	assert( j->sendMeshIndex().size() == 1 ); // Sending data from 1 voxel
	assert( j->sendMeshIndex()[0] == 0 );
	////////////////////////////////////////////////////////////////
	Id gsA( "/model/kinetics/stoich" );
	assert( gsA != Id() );
	GslStoich* gs = reinterpret_cast< GslStoich* >( gsA.eref().data() );
	assert( gs->pools().size() == 4 ); // One for self, 3 for diffn.
	assert( gs->pools()[0].size() == 5 ); // M1, M2 and proxies M3, M4, M5
	assert( gs->pools()[1].size() == 2 ); // M1 and M2.
	assert( gs->pools()[2].size() == 2 ); // M1 and M2.
	assert( gs->pools()[3].size() == 2 ); // M1 and M2.
	assert( gs->ode().size() == 8 ); // combos: x 1 2 3 12 13 23 123
	assert( gs->pools()[0].getSolver() == 7 );
	assert( gs->pools()[1].getSolver() == 0 ); // Should not even be calling
	assert( gs->pools()[2].getSolver() == 0 ); // Should not even be calling
	assert( gs->pools()[3].getSolver() == 0 ); // Should not even be calling
	assert( gs->coreStoich()->getNumVarPools() == 2 );
	assert( gs->coreStoich()->getNumProxyPools() == 3 );
	assert( gs->coreStoich()->getNumRates() == 4 );
	assert( gs->coreStoich()->getNumCoreRates() == 2 );
	assert( gs->ode()[0].compartmentSignature_.size() == 0 );
	assert( gs->ode()[0].stoich_->getNumVarPools() == 2 );
	assert( gs->ode()[0].stoich_->getNumProxyPools() == 0 );
	assert( gs->ode()[0].stoich_->getNumRates() == 2 );
	// The combo will use this
	assert( gs->ode()[7].compartmentSignature_.size() == 3 );
	assert( gs->ode()[7].compartmentSignature_[0] == B );
	assert( gs->ode()[7].compartmentSignature_[1] == D );
	assert( gs->ode()[7].compartmentSignature_[2] == C );

	// Checking stencils.
	// The stencil here should connect up to the proxy compts for B,C,D.
	const double* entry;
	const unsigned int* colIndex;
	unsigned int numInRow;
	// const double diffConst = 1e-12;
	// DiffConst * length of voxel /area of voxel
	const double adx = 1e-5; 
	numInRow = gs->compartmentMesh()->getStencil( 0, &entry, &colIndex);
	assert( numInRow  == 3 );
	assert( colIndex[0] == 1 );
	assert( colIndex[1] == 2 );
	assert( colIndex[2] == 3 );
	assert( doubleEq( entry[0], adx ) );
	assert( doubleEq( entry[1], adx ) );
	assert( doubleEq( entry[2], adx ) );
	numInRow = gs->compartmentMesh()->getStencil( 1, &entry, &colIndex);
	assert( numInRow == 1 );
	assert( colIndex[0] == 0 );
	assert( doubleEq( entry[0], adx ) );
	numInRow = gs->compartmentMesh()->getStencil( 2, &entry, &colIndex);
	assert( numInRow == 1 );
	assert( colIndex[0] == 0 );
	assert( doubleEq( entry[0], adx ) );
	numInRow = gs->compartmentMesh()->getStencil( 3, &entry, &colIndex);
	assert( numInRow == 1 );
	assert( colIndex[0] == 0 );
	assert( doubleEq( entry[0], adx ) );

	////////////////////////////////////////////////////////////////
	Id gsB( "/model/compartment_1/stoich" );
	assert( gsB != Id() );
	gs = reinterpret_cast< GslStoich* >( gsB.eref().data() );

	// 3 diffusion compts plus 3 to connect to C. A is handling the other.
	assert( gs->pools().size() == 6 ); 
	assert( gs->pools()[0].size() == 5 ); // 4 molecules, one X reac to C
	assert( gs->pools()[1].size() == 5 ); // 4 molecules, one X reac to C
	assert( gs->pools()[2].size() == 5 ); // 4 molecules, one X reac to C
	assert( gs->pools()[3].size() == 4 ); // 1 diff mol
	assert( gs->pools()[4].size() == 4 ); // 1 diff mol
	assert( gs->pools()[5].size() == 4 ); // 1 diff mol
	assert( gs->ode().size() == 2 ); // combos: x C
	assert( gs->pools()[0].getSolver() == 1 );
	assert( gs->pools()[1].getSolver() == 1 );
	assert( gs->pools()[2].getSolver() == 1 ); // Only case of X reac.
	assert( gs->pools()[3].getSolver() == 0 );
	assert( gs->pools()[4].getSolver() == 0 );
	assert( gs->pools()[5].getSolver() == 0 );
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
	assert( gs->pools().size() == 5 ); // 5 voxels, diffn is handled by A&B.
	// because C is the follower on all the X-diffs.
	assert( gs->pools()[0].size() == 2 ); // 2 mols, X reacs are followers
	assert( gs->pools()[1].size() == 2 ); // 2 mols, X reacs are followers
	assert( gs->pools()[2].size() == 2 ); // 2 mols, X reacs are followers
	assert( gs->pools()[3].size() == 2 ); // 2 mols, X reacs are followers
	assert( gs->pools()[4].size() == 2 ); // 2 mols, X reacs are followers
	/*
	assert( gs->pools()[5].size() == 2 ); // 1 xdiff
	assert( gs->pools()[6].size() == 2 ); // 1 xdiff
	assert( gs->pools()[7].size() == 2 ); // 1 xdiff
	assert( gs->pools()[8].size() == 2 ); // 1 xdiff
	*/
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
	assert( gs->pools().size() == 2 ); // 2 voxels, X-diff follower with A.
	assert( gs->pools()[0].size() == 2 ); // 2 mols, no X-reac
	assert( gs->pools()[1].size() == 2 ); // 2 mols, no X-reac
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
	makeReacDiffGraphs( "/model/kinetics/M1" );
	makeReacDiffGraphs( "/model/kinetics/M2" );
	makeReacDiffGraphs( "/model/compartment_1/M1" );
	makeReacDiffGraphs( "/model/compartment_1/M3" );
	makeReacDiffGraphs( "/model/compartment_1/M6" );
	makeReacDiffGraphs( "/model/compartment_3/M1" );
	makeReacDiffGraphs( "/model/compartment_3/M4" );
	makeReacDiffGraphs( "/model/compartment_2/M1" );
	makeReacDiffGraphs( "/model/compartment_2/M5" );
	shell->doUseClock( "/model/graphs/conc1/#.conc", "process", 8 );

	for ( unsigned int i = 0; i < 10; ++i )
		shell->doSetClock( i, DT );
	shell->doSetClock( 8, 1 );
	shell->doReinit();
	shell->doStart( 100.0 );

	dumpReacDiffGraphs();

	shell->doDelete( model );
	cout << "." << flush;
}
