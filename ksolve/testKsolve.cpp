/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef WIN32
	#include <sys/time.h>
#else
	#include <time.h>
	#include <winsock2.h>
#endif
#include "StoichHeaders.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include "ElementValueFinfo.h"
#include "GslIntegrator.h"
#include "SolverBase.h"
#include "VoxelPools.h"
#include "OdeSystem.h"
#include "SolverJunction.h"
#include "../mesh/Boundary.h"
#include "../mesh/MeshEntry.h"
#include "../mesh/ChemMesh.h"
#include "GslStoich.h"
#include "../shell/Shell.h"
#include "ReadKkit.h"

static const double TOLERANCE = 1e-6;

// This is a regression test
void testKsolveZombify( string modelFile )
{
	ReadKkit rk;
	Id base = rk.read( modelFile, "dend", Id() );
	assert( base != Id() );
	// Id kinetics = s->doFind( "/kinetics" );

	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );
	Id stoich = s->doCreate( "Stoich", base, "stoich", dims );
	assert( stoich != Id() );
	string temp = "/dend/##";
	bool ret = Field<string>::set( stoich, "path", temp );
	assert( ret );

	/*
	rk.run();
	rk.dumpPlots( "dend.plot" );
	*/

	s->doDelete( base );
	cout << "." << flush;
}

/**
 * Benchmarks assorted models: both time and values. Returns 
 * time it takes to run the model.
 * modelName is the base name of the model
 * plotName is of the form "conc1/foo.Co"
 * simTime is the time 
 */
 double testGslIntegrator( string modelName, string plotName,
 	double plotDt, double simTime )
{
	ReadKkit rk;
	Id base = rk.read( modelName + ".g" , "model", Id() );
	assert( base != Id() );
	// Id kinetics = s->doFind( "/kinetics" );

	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );
	/*
	Id stoich = s->doCreate( "Stoich", base, "stoich", dims );
	assert( stoich != Id() );
	string temp = "/model/##";
	bool ret = Field<string>::set( stoich, "path", temp );
	assert( ret );
	*/
	Id stoich = base;

	Id gsl = s->doCreate( "GslIntegrator", base, "gsl", dims );
	/*
	MsgId mid = s->doAddMsg( "Single", 
		ObjId( stoich, 0 ), "plugin", 
		ObjId( gsl, 0 ), "stoich" );
	assert( mid != Msg::badMsg );

	const Finfo* f = Stoich::initCinfo()->findFinfo( "plugin" );
	assert( f );
	const SrcFinfo1< Stoich* >* plug = 
		dynamic_cast< const SrcFinfo1< Stoich* >* >( f );
	assert( plug );
	Stoich* stoichData = reinterpret_cast< Stoich* >( stoich.eref().data() );
	GslIntegrator* gi = reinterpret_cast< GslIntegrator* >( gsl.eref().data() );
	*/

	ProcInfo p;
	p.dt = 1.0;
	p.currTime = 0;

	// plug->send( stoich.eref(), &p, stoichData );
	bool ret = SetGet1< Id >::set( gsl, "stoich", stoich );
	assert( ret );
	ret = Field< bool >::get( gsl, "isInitialized" );
	assert( ret );

	s->doSetClock( 0, plotDt );
	s->doSetClock( 1, plotDt );
	s->doSetClock( 2, plotDt );
	string gslpath = rk.getBasePath() + "/gsl";
	string  plotpath = rk.getBasePath() + "/graphs/##[TYPE=Table]," +
		rk.getBasePath() + "/moregraphs/##[TYPE=Table]";
	s->doUseClock( gslpath, "process", 0 );
	s->doUseClock( plotpath, "process", 2 );
	struct timeval tv0;
	struct timeval tv1;
	gettimeofday( &tv0, 0 );
	s->doReinit();
	s->doStart( simTime );
	gettimeofday( &tv1, 0 );

	if ( plotName.length() > 0 ) {
		Id plotId( string( "/model/graphs/" ) + plotName );
		assert( plotId != Id() );
		bool ok = SetGet::strSet( plotId, "compareXplot",
			modelName + ".plot,/graphs/" + plotName + ",rmsr" );
		assert( ok );
		double rmsr = Field< double >::get( plotId, "outputValue" );
		assert( rmsr < TOLERANCE );
	}

	s->doDelete( base );
	cout << "." << flush;
	double sret = tv1.tv_sec - tv0.tv_sec;
	double uret = tv1.tv_usec;
	uret -= tv0.tv_usec;
	return sret + 1e-6 * uret;
	
}

void testGsolver(string modelName, string plotName, double plotDt, double simTime, double volume )
{
	ReadKkit rk;
	Id base = rk.read( modelName + ".g" , "model", Id(), "Gssa" );
	assert( base != Id() );
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	Id kinetics( "/model/kinetics" );
	assert( kinetics != Id() );
	SetGet2< double, unsigned int >::set( kinetics, "buildDefaultMesh",
		volume, 1 );

	vector< unsigned int > dims( 1, 1 );
	Id stoich = base;

	for ( unsigned int i = 0; i < 10; ++i )
		s->doSetClock( i, plotDt );

	// string  plotpath = rk.getBasePath() + "/graphs/##[TYPE=Table]," +
		// rk.getBasePath() + "/moregraphs/##[TYPE=Table]";
	// s->doUseClock( base.path(), "process", 0 );
	// s->doUseClock( plotpath, "process", 2 );
	s->doReinit();
	s->doStart( simTime );


	string plotfile = modelName + ".out";
	Id tempId( "/model/graphs/conc1" );
	vector< Id > kids = Field< vector< Id > >::get( tempId, "children" );
	for ( unsigned int i = 0 ; i < kids.size(); ++i ) {
		string str = Field< string >::get( kids[i], "name" );
		SetGet2< string, string>::set( kids[i], "xplot", plotfile, str);
	}
	// This is to pick up if the graph is on another window.
	tempId = Id( "/model/graphs/conc2" );
	kids = Field< vector< Id > >::get( tempId, "children" );
	for ( unsigned int i = 0 ; i < kids.size(); ++i ) {
		string str = Field< string >::get( kids[i], "name" );
		SetGet2< string, string>::set( kids[i], "xplot", plotfile, str);
	}

	/*
	if ( plotName.length() > 0 ) {
		Id plotId( string( "/model/graphs/conc1/" ) + plotName );
		assert( plotId != Id() );
		SetGet2< string, string>::set( plotId, "xplot", plotfile, plotName);
		bool ok = SetGet::strSet( plotId, "compareXplot",
			modelName + ".plot,/graphs/" + plotName + ",rmsr" );
		assert( ok );
		double rmsr = Field< double >::get( plotId, "outputValue" );
		assert( rmsr < TOLERANCE );
	}
	*/
	s->doDelete( base );
	cout << "." << flush;
}

////////////////////////////////////////////////////////////////////////
// Proper unit tests below here. Don't need whole MOOSE infrastructure.
////////////////////////////////////////////////////////////////////////

/**
 * This makes *meshA* and *meshB*, and puts *poolA* and *poolB* in it.
 * There is
 * a reversible conversion reaction *reac* also in meshA.
 */
Id makeInterMeshReac( Shell* s )
{
	vector< int > dims( 1, 1 );
	Id model = s->doCreate( "Neutral", Id(), "model", dims );
	Id meshA = s->doCreate( "CubeMesh", model, "meshA", dims );
	Id meshEntryA = Neutral::child( meshA.eref(), "mesh" );
	Id meshB = s->doCreate( "CubeMesh", model, "meshB", dims );
	Id meshEntryB = Neutral::child( meshB.eref(), "mesh" );
	Id poolA = s->doCreate( "Pool", meshA, "A", dims );
	Id poolB = s->doCreate( "Pool", meshB, "B", dims );
	Id reac = s->doCreate( "Reac", meshA, "reac", dims );
	Field< double >::set( reac, "Kf", 0.2 );
	Field< double >::set( reac, "Kb", 0.1 );

	Field< double >::set( poolA, "nInit", 100 );

	MsgId mid = s->doAddMsg( "OneToOne",
		poolA, "mesh", meshEntryA, "mesh" );
	assert( mid != Msg::bad );
	mid = s->doAddMsg( "OneToOne", poolB, "mesh", meshEntryB, "mesh" );
	assert( mid != Msg::bad );
	mid = s->doAddMsg( "Single", meshEntryA, "remeshReacs", reac, "remesh");
	assert( mid != Msg::bad );

	mid = s->doAddMsg( "Single", reac, "sub", poolA, "reac" );
	assert( mid != Msg::bad );
	mid = s->doAddMsg( "Single", reac, "prd", poolB, "reac" );
	assert( mid != Msg::bad );

	vector< double > coords( 9, 0 );
	coords[0] = 1; // x0 position
	coords[3] = 2; // x1 position
	coords[4] = 1; // y1 position
	coords[5] = 1; // z1 position
	coords[6] = 1; // dx
	coords[7] = 1; // dy
	coords[8] = 1; // dz
	Field< vector< double > >::set( meshB, "coords", coords );
	Field< double >::set( meshA, "size", 1 ); 
	// Field< double >::set( meshB, "size", 8 );
	double ret = Field< double >::get( poolA, "size" );
	assert( doubleEq( ret, 1 ) );
	ret = Field< double >::get( poolB, "size" );
	assert( doubleEq( ret, 1 ) );
	ret = Field< double >::get( meshA, "dx" );
	assert( doubleEq( ret, 1 ) );
	ret = Field< double >::get( meshB, "dx" );
	assert( doubleEq( ret, 1 ) );
	unsigned int n = Field< unsigned int >::get( meshA, "nx" );
	assert( n == 1 );
	n = Field< unsigned int >::get( meshA, "nx" );
	assert( n == 1 );
	n = Field< unsigned int >::get( meshA, "num_mesh" );
	assert( n == 1 );
	n = Field< unsigned int >::get( meshB, "num_mesh" );
	assert( n == 1 );

	return model;
}

void testInterMeshReac()
{
	const double DT = 0.01;
	const double RUNTIME = 50.0;
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	s->doSetClock( 4, DT );
	s->doSetClock( 5, DT );
	s->doSetClock( 6, DT );
	vector< int > dims( 1, 1 );
	Id model = makeInterMeshReac( s );
	// Create solvers for meshA and meshB.
	Id comptA( "/model/meshA" );
	Id comptB( "/model/meshB" );
	Id meshA( "/model/meshA/mesh" );
	Id meshB( "/model/meshB/mesh" );
	Id stoichA = s->doCreate( "GslStoich", comptA, "stoichA", dims );
	assert ( stoichA != Id() );
	Field< string >::set( stoichA, "path", "/model/meshA/##" );
	Field< Id >::set( stoichA, "compartment", Id( "/model/meshA" ) );
	Field< string >::set( stoichA, "method", "rk5" );

	Id stoichB = s->doCreate( "GslStoich", comptB, "stoichB", dims );
	assert ( stoichB != Id() );
	Field< string >::set( stoichB, "path", "/model/meshB/##" );
	Field< Id >::set( stoichB, "compartment", Id( "/model/meshB" ) );
	Field< string >::set( stoichB, "method", "rk5" );

	MsgId mid = s->doAddMsg( "Single", meshA, "remesh", stoichA, 
					"remesh" );
	assert( mid != Msg::bad );
	mid = s->doAddMsg( "Single", meshB, "remesh", stoichB, "remesh" );
	assert( mid != Msg::bad );

	GslStoich* gsA = 
			reinterpret_cast< GslStoich* >( stoichA.eref().data() );
	assert( gsA->pools().size() == 1 );
	assert( gsA->pools()[0].size() == 2 );
	assert( gsA->ode().size() == 2 ); // One blank, one with a single jn.

	GslStoich* gsB = 
			reinterpret_cast< GslStoich* >( stoichB.eref().data() );
	assert( gsB->pools().size() == 1 );
	assert( gsB->pools()[0].size() == 1 );
	assert( gsB->ode().size() == 1 );

	SetGet1<Id>::set( stoichA, "addJunction", stoichB );

	//assert( gsA->pools().size() == 1 );
	assert( gsA->pools()[0].size() == 2 );
	assert( gsA->ode().size() == 2 );
	assert( gsB->pools().size() == 1 );
	assert( gsB->pools()[0].size() == 1 );
	assert( gsB->ode().size() == 1 );

	assert( gsA->ode()[0].stoich_->getNumRates() == 0 );
	assert( gsA->ode()[1].stoich_->getNumRates() == 1 );
	vector< unsigned int > molIndex;
	unsigned int numSub = 
			gsA->ode()[1].stoich_->rates(0)->getReactants( molIndex );
	assert( numSub == 1 );
	assert( molIndex.size() == 2 ); // 1 sub and 1 prd. 
	assert( molIndex[0] == 0 );
	assert( molIndex[1] == 1 );

	s->doUseClock( "/model/meshA/stoichA,/model/meshB/stoichB", "init",  4 );
	s->doUseClock( "/model/meshA/stoichA,/model/meshB/stoichB", "process",  5 );


	Id poolA( "/model/meshA/A" );
	Id poolB( "/model/meshB/B" );
	assert( poolA != Id() );
	assert( poolB != Id() );
	Field< double >::set( ObjId( poolA, 0 ), "nInit", 1.0 );
	s->doReinit();
	s->doStart( RUNTIME );

	double a = gsA->pools()[0].S()[0];
	double b = gsA->pools()[0].S()[1];
	assert( a < 1.0 );
	assert( 0.0 < b && b < 1.0 );
	assert( doubleEq( 2 * a, b ) );
	a = Field< double >::get( ObjId( poolA, 0 ), "n" );
	b = Field< double >::get( ObjId( poolB, 0 ), "n" );
	assert( a < 1.0 );
	assert( 0.0 < b && b < 1.0 );
	assert( doubleEq( 2 * a, b ) );

	s->doDelete( model );
	cout << "." << flush;
}

Id makeSimpleReac( Shell* s )
{
	vector< int > dims( 1, 1 );
	Id model = s->doCreate( "Neutral", Id(), "model", dims );
	Id meshA = s->doCreate( "CubeMesh", model, "meshA", dims );
	Id meshEntryA = Neutral::child( meshA.eref(), "mesh" );
	Id poolA = s->doCreate( "Pool", meshA, "A", dims );
	Id poolB = s->doCreate( "Pool", meshA, "B", dims );
	Id reac = s->doCreate( "Reac", meshA, "reac", dims );

	Field< double >::set( poolA, "nInit", 100 );
	Field< double >::set( reac, "Kf", 0.1 );
	Field< double >::set( reac, "Kb", 0.1 );

	MsgId mid = s->doAddMsg( "OneToOne", poolA, "mesh", meshEntryA, "mesh");
	assert( mid != Msg::bad );
	mid = s->doAddMsg( "OneToOne", poolB, "mesh", meshEntryA, "mesh" );
	assert( mid != Msg::bad );
	mid = s->doAddMsg( "Single", meshEntryA, "remeshReacs", reac, "remesh");
	assert( mid != Msg::bad );

	mid = s->doAddMsg( "Single", reac, "sub", poolA, "reac" );
	assert( mid != Msg::bad );
	mid = s->doAddMsg( "Single", reac, "prd", poolB, "reac" );
	assert( mid != Msg::bad );

	return model;
}

void testGslStoich()
{
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );
	Id model = makeSimpleReac( s );
	// Create solver 
	Id meshA( "/model/meshA" );
	assert ( meshA != Id() );
	// This sequence is somewhat messy. We must create the core on the
	// GslStoich. We then must assign the path to the StoichCore, 
	// use the converted model to allocate the arrays in the GslStoich,
	// and when this is done we can complete zombification.
	Id solver = s->doCreate( "GslStoich", model, "solver", dims );
	assert ( solver != Id() );
	Field< string >::set( solver, "path", "/model/meshA/##" );
	unsigned int nVarPools = 
			Field< unsigned int >::get( solver, "numVarPools" );
	assert( nVarPools == 2 );
	GslStoich* gs = reinterpret_cast< GslStoich* >( solver.eref().data() );
	// gs->coreStoich()->print();
	assert( gs->coreStoich()->getNumRates() == 1 );
	assert( gs->coreStoich()->getNumCoreRates() == 1 );
	assert( gs->ode().size() == 1 );
	assert( gs->ode()[0].stoich_->getNumRates() == 1 );
	assert( gs->ode()[0].stoich_->getNumCoreRates() == 1 );
	// gs->ode()[0].stoich_->print();

	ProcInfo p;
	p.dt = 1;
	p.currTime = 0;

	Id poolA( "/model/meshA/A" );
	assert( poolA != Id() );
	double x = Field< double >::get( poolA, "nInit" );
	assert( doubleEq( x, 100 ) );
	x = Field< double >::get( poolA, "n" );
	assert( doubleEq( x, 0 ) );

	gs->reinit( solver.eref(), &p );

	x = Field< double >::get( poolA, "nInit" );
	assert( doubleEq( x, 100 ) );
	x = Field< double >::get( poolA, "n" );
	assert( doubleEq( x, 100 ) );

	for ( p.currTime = 0.0; p.currTime < 100.5; p.currTime += p.dt ) {
		double n = Field< double >::get( poolA, "n" );
		// cout << p.currTime << "	" << n << "	" << 50 + 50 * exp( -p.currTime * 0.2 ) << endl;
		assert( doubleEq( n, 50 + 50 * exp( -p.currTime * 0.2 ) ) );
		gs->process( solver.eref(), &p );
	}

	//MsgId mid = s->doAddMsg( "Single", meshA, "meshSplit", solver, "remesh" );
	// assert( mid != Msg::bad );

	s->doDelete( model );
	cout << "." << flush;
}

void testJunctionSetup()
{
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );
	Id model = makeInterMeshReac( s );
	// Create solvers for meshA and meshB.
	Id meshA( "/model/meshA/mesh" );
	assert ( meshA != Id() );
	Id meshB( "/model/meshB/mesh" );
	assert ( meshB != Id() );
	Id comptA( "/model/meshA" );
	Id comptB( "/model/meshB" );
	Id stoichA = s->doCreate( "GslStoich", comptA, "stoichA", dims );
	assert ( stoichA != Id() );
	Field< string >::set( stoichA, "path", "/model/meshA/##" );
	Field< Id >::set( stoichA, "compartment", Id( "/model/meshA" ) );
	Field< string >::set( stoichA, "method", "rk5" );

	Id stoichB = s->doCreate( "GslStoich", comptB, "stoichB", dims );
	assert ( stoichB != Id() );
	Field< string >::set( stoichB, "path", "/model/meshB/##" );
	Field< Id >::set( stoichB, "compartment", Id( "/model/meshB" ) );
	Field< string >::set( stoichB, "method", "rk5" );

	MsgId mid = s->doAddMsg( "Single", meshA, "remesh", stoichA, 
					"remesh" );
	assert( mid != Msg::bad );
	mid = s->doAddMsg( "Single", meshB, "remesh", stoichB, "remesh" );
	assert( mid != Msg::bad );

	assert( Field< unsigned int >::get( stoichA, "num_junction" ) == 0 );
	assert( Field< unsigned int >::get( stoichB, "num_junction" ) == 0 );
	SetGet1<Id>::set( stoichA, "addJunction", stoichB );
	assert( Field< unsigned int >::get( stoichA, "num_junction" ) == 1 );
	assert( Field< unsigned int >::get( stoichB, "num_junction" ) == 1 );

	Id junctionA( "/model/meshA/stoichA/junction" );
	Id junctionB( "/model/meshB/stoichB/junction" );
	assert ( junctionA != Id() );
	assert ( junctionB != Id() );

	vector< Id > tgts;
	junctionA.element()->getNeighbours( tgts, junctionPoolDeltaFinfo() );
	assert( tgts.size() == 1 );
	assert( tgts[0] == junctionB );

	////////////////////////////////////////////////////////////////////
	//
	// OK, we've made the junction. This is what it should have:
	// JunctionA (on A): 
	// 		reacTerms_: deprecated
	// 		diffTerms_: empty
	// 		diffScale_: empty
	// 		targetMeshIndices_, aka meshMap(): {0} ?
	// 			as far as I can tell it isn't used except possibly at setup
	//		localReacPools_: empty
	//		remoteReacPools_: {1}
	//		sendPoolIndex_: empty, this is only for diffusion
	//		sendMeshIndex_: {0}. Specifies meshIndex involved in 
	//				cross-reac as well as diffusion.
	//		abutPoolIndex_: empty, used only for diffusion
	//		abutMeshIndex_: empty, used only for diffusion
	//
	// JunctionB (on B): The reac-related ones are:
	// 		targetMeshIndices_, aka meshMap(): {0} ?
	// 			as far as I can tell it isn't used except possibly at setup
	//		localReacPools_: {0}
	//		remoteReacPools_: empty
	//		sendMeshIndex_: {0}. Specifies meshIndex involved in 
	//				cross-reac as well as diffusion.
	//
	////////////////////////////////////////////////////////////////////
	const SolverJunction* jA = reinterpret_cast< const SolverJunction* >(
			junctionA.eref().data() );
	const SolverJunction* jB = reinterpret_cast< const SolverJunction* >(
			junctionB.eref().data() );
	assert( jA->localReacPools().size() == 0 );
	assert( jA->remoteReacPools().size() == 1 );
	assert( jA->remoteReacPools()[0] == 1 );
	assert( jA->sendMeshIndex().size() == 1 );
	assert( jA->sendMeshIndex()[0] == 0 );

	assert( jB->localReacPools().size() == 1 );
	assert( jB->localReacPools()[0] == 0 );
	assert( jB->remoteReacPools().size() == 0 );
	assert( jB->sendMeshIndex().size() == 1 );
	assert( jB->sendMeshIndex()[0] == 0 );

	s->doDelete( model );
	cout << "." << flush;
}

typedef pair< unsigned int, unsigned int > PII;

/**
 * Generates the entries for the following mesh junction:
 *
 *                   meshB
 *         0 1 2 3 4
 *         ---
 *         0 1|5 6 7
 *            |
 *         2 3|8 9 10
 *             ---
 *         4 5 6 7
 * meshA
 */
void testMatchMeshEntries()
{
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );
	Id meshA = s->doCreate( "CubeMesh", Id(), "A", dims );
	Id meshB = s->doCreate( "CubeMesh", Id(), "B", dims );
	GslStoich self;
	GslStoich other;

	vector< double > coords( 9, 0.0 );
	coords[3] = 4; // x1
	coords[4] = 3; // y1
	coords[5] = 1; // z1
	coords[6] = 1; // dx
	coords[7] = 1; // dy
	coords[8] = 1; // dz

	
	Field< bool >::set( meshA, "preserveNumEntries", false );
	Field< vector< double > >::set( meshA, "coords", coords );
	vector< unsigned int > s2m( 12, ~0 );
	s2m[0] = 0;
	s2m[1] = 1;
	s2m[4] = 2;
	s2m[5] = 3;
	s2m[8] = 4;
	s2m[9] = 5;
	s2m[10] = 6;
	s2m[11] = 7;
	Field< vector< unsigned int > >::set( meshA, "spaceToMesh", s2m );

	vector< unsigned int > m2s( 8, 0 );
	m2s[0] = 0;
	m2s[1] = 1;
	m2s[2] = 4;
	m2s[3] = 5;
	m2s[4] = 8;
	m2s[5] = 9;
	m2s[6] = 10;
	m2s[7] = 11;
	Field< vector< unsigned int > >::set( meshA, "meshToSpace", m2s );

	//start with only the relevant. Later see how it handles the whole.
	vector< unsigned int > surface( 5, 0 ); 
	surface[0] = 0;
	surface[1] = 1;
	surface[2] = 5;
	surface[3] = 10;
	surface[4] = 11;
	Field< vector< unsigned int > >::set( meshA, "surface", surface );
	self.setCompartment( meshA );


	coords[1] = -1; // y0. Using the upside-down y convention for this.
	coords[3] = 5; // x1
	coords[4] = 2; // y1
	Field< bool >::set( meshB, "preserveNumEntries", false );
	Field< vector< double > >::set( meshB, "coords", coords );
	s2m.clear();
	s2m.resize( 15, ~0 );
	s2m[0] = 0;
	s2m[1] = 1;
	s2m[2] = 2;
	s2m[3] = 3;
	s2m[4] = 4;
	s2m[7] = 5;
	s2m[8] = 6;
	s2m[9] = 7;
	s2m[12] = 8;
	s2m[13] = 9;
	s2m[14] = 10;
	Field< vector< unsigned int > >::set( meshB, "spaceToMesh", s2m );
	m2s.clear();
	m2s.resize( 11, 0 );
	m2s[0] = 0;
	m2s[1] = 1;
	m2s[2] = 2;
	m2s[3] = 3;
	m2s[4] = 4;
	m2s[5] = 7;
	m2s[6] = 8;
	m2s[7] = 9;
	m2s[8] = 12;
	m2s[9] = 13;
	m2s[10] = 14;
	Field< vector< unsigned int > >::set( meshB, "meshToSpace", m2s );

	surface[0] = 0;
	surface[1] = 1;
	surface[2] = 7;
	surface[3] = 12;
	surface[4] = 13;
	Field< vector< unsigned int > >::set( meshB, "surface", surface );
	other.setCompartment( meshB );

	vector< unsigned int > selfMeshIndex;
	vector< unsigned int > otherMeshIndex;
	vector< VoxelJunction > selfMeshMap;
	vector< VoxelJunction > otherMeshMap;

	self.matchMeshEntries( &other, 
					selfMeshIndex, selfMeshMap,
					otherMeshIndex, otherMeshMap );

	assert( selfMeshIndex.size() == 5 );
	assert( selfMeshIndex[0] == 0 );
	assert( selfMeshIndex[1] == 1 );
	assert( selfMeshIndex[2] == 3 );
	assert( selfMeshIndex[3] == 6 );
	assert( selfMeshIndex[4] == 7 );

	assert( otherMeshIndex.size() == 5 );
	assert( otherMeshIndex[0] == 0 );
	assert( otherMeshIndex[1] == 1 );
	assert( otherMeshIndex[2] == 5 );
	assert( otherMeshIndex[3] == 8 );
	assert( otherMeshIndex[4] == 9 );

	assert ( selfMeshMap.size() ==  6);
	vector< VoxelJunction >::iterator i = selfMeshMap.begin();
	assert( i->first == 0 && i->second == 0 ); ++i;
	assert( i->first == 1 && i->second == 1 ); ++i;
	assert( i->first == 2 && i->second == 1 ); ++i;
	assert( i->first == 3 && i->second == 3 ); ++i;
	assert( i->first == 3 && i->second == 6 ); ++i;
	assert( i->first == 4 && i->second == 7 ); ++i;

	assert ( otherMeshMap.size() ==  6);
	i = otherMeshMap.begin();
	assert( i->first == 0 && i->second == 0 ); ++i;
	assert( i->first == 1 && i->second == 1 ); ++i;
	assert( i->first == 1 && i->second == 5 ); ++i;
	assert( i->first == 2 && i->second == 8 ); ++i;
	assert( i->first == 3 && i->second == 8 ); ++i;
	assert( i->first == 4 && i->second == 9 ); ++i;

	cout << "." << flush;
}

// Returns the Ids for the compt and the responsible stoich object.
pair< Id, Id > makeComptForDiffusion( Shell* s, Id model, unsigned int i )
{
	const double SIDE = 10e-6;
	const double DIFFCONST = 1e-12;
	vector< int > dims( 1, 1 );
	stringstream ss;
	ss << "compt_" << i;
	string comptName = ss.str();

	Id compt = s->doCreate( "CubeMesh", model, comptName, dims );
	Id comptMeshEntry = Neutral::child( compt.eref(), "mesh" );
	Id poolA = s->doCreate( "Pool", compt, "A", dims );
	Field< double >::set( poolA, "nInit", 0.0 );
	Field< double >::set( poolA, "diffConst", DIFFCONST );
	MsgId mid = 
		s->doAddMsg( "OneToOne", poolA, "mesh", comptMeshEntry, "mesh");
	assert( mid != Msg::bad );
	Id stoichA = s->doCreate( "GslStoich", compt, "stoichA", dims );
	assert ( stoichA != Id() );
	string path = "/model/" + comptName + "/##";
	Field< string >::set( stoichA, "path", path );
	Field< Id >::set( stoichA, "compartment", compt );
	Field< string >::set( stoichA, "method", "rk5" );

	mid = s->doAddMsg( "Single", comptMeshEntry, "remesh", stoichA, "remesh" );

	vector< double > coords( 9, 0.0 );
	coords[0] = SIDE * ( 5 * (i % 3) );
	coords[1] = SIDE * ( 5 * (i / 3) );
	coords[2] = 0;
	coords[3] = coords[0] + SIDE * 5;
	coords[4] = coords[1] + SIDE * 5;
	coords[5] = coords[2] + SIDE;
	coords[6] = coords[7] = coords[8] = SIDE;
	Field< bool >::set( compt, "preserveNumEntries", false );
	Field< vector< double > >::set( compt, "coords", coords );

	s->doUseClock( "/model/" + comptName + "/stoichA", "init",  4 );
	s->doUseClock( "/model/" + comptName + "/stoichA", "process",  5 );

	assert( mid != Msg::bad );

	return pair< Id, Id >( compt, stoichA );
}

// Returns the Ids for the compt and the responsible stoich object.
pair< Id, Id > makeOneDimComptForDiffusion( Shell* s, Id model, unsigned int i )
{
	const double SIDE = 10e-6;
	const double DIFFCONST = 1e-12;
	vector< int > dims( 1, 1 );
	stringstream ss;
	ss << "compt_" << i;
	string comptName = ss.str();

	Id compt = s->doCreate( "CubeMesh", model, comptName, dims );
	Id comptMeshEntry = Neutral::child( compt.eref(), "mesh" );
	Id poolA = s->doCreate( "Pool", compt, "A", dims );
	Field< double >::set( poolA, "nInit", 0.0 );
	Field< double >::set( poolA, "diffConst", DIFFCONST );
	MsgId mid = 
		s->doAddMsg( "OneToOne", poolA, "mesh", comptMeshEntry, "mesh");
	assert( mid != Msg::bad );
	Id stoichA = s->doCreate( "GslStoich", compt, "stoichA", dims );
	assert ( stoichA != Id() );
	string path = "/model/" + comptName + "/##";
	Field< string >::set( stoichA, "path", path );
	Field< Id >::set( stoichA, "compartment", compt );
	Field< string >::set( stoichA, "method", "rk5" );

	mid = s->doAddMsg( "Single", comptMeshEntry, "remesh", stoichA, "remesh" );

	vector< double > coords( 9, 0.0 );
	coords[0] = SIDE * 5 * i;
	coords[1] = 0;
	coords[2] = 0;
	coords[3] = coords[0] + SIDE * 5;
	coords[4] = coords[1] + SIDE;
	coords[5] = coords[2] + SIDE;
	coords[6] = coords[7] = coords[8] = SIDE;
	Field< bool >::set( compt, "preserveNumEntries", false );
	Field< vector< double > >::set( compt, "coords", coords );

	s->doUseClock( "/model/" + comptName + "/stoichA", "init",  4 );
	s->doUseClock( "/model/" + comptName + "/stoichA", "process",  5 );

	assert( mid != Msg::bad );

	return pair< Id, Id >( compt, stoichA );
}

// Called from testMolTransferAcrossJunctions
// Expects two 5x5 meshes, adjoining on the left-right side.
//
// 20 21 22 23 24 | 29
// 15 16 17 18 19 | 27
// 10 11 12 13 14 | 27
//  5  6  7  8  9 | 26
//  0  1  2  3  4 | 25
//
//
void checkDiffusionStencil( const ChemMesh* cm0, const ChemMesh* cm1 )
{
	// const double SIDE = 10e-6;
	const double *entry;
	const unsigned int *colIndex;
	unsigned int num;

	num = cm0->getStencil( 4, &entry, &colIndex );
	assert( num == 3 ); 
	assert( colIndex[0] == 3 );
	assert( colIndex[1] == 9 );
	assert( colIndex[2] == 25 );

	num = cm0->getStencil( 9, &entry, &colIndex );
	assert( num == 4 ); 
	assert( colIndex[0] == 4 );
	assert( colIndex[1] == 8 );
	assert( colIndex[2] == 14 );
	assert( colIndex[3] == 26 );

	num = cm0->getStencil( 14, &entry, &colIndex );
	assert( num == 4 ); 
	assert( colIndex[0] == 9 );
	assert( colIndex[1] == 13 );
	assert( colIndex[2] == 19 );
	assert( colIndex[3] == 27 );
	num = cm0->getStencil( 19, &entry, &colIndex );
	assert( num == 4 ); 
	assert( colIndex[0] == 14 );
	assert( colIndex[1] == 18 );
	assert( colIndex[2] == 24 );
	assert( colIndex[3] == 28 );

	num = cm0->getStencil( 24, &entry, &colIndex );
	assert( num == 3 ); 
	assert( colIndex[0] == 19 );
	assert( colIndex[1] == 23 );
	assert( colIndex[2] == 29 );

	num = cm0->getStencil( 25, &entry, &colIndex );
	assert( num == 1 );
	// assert( doubleEq( entry[0], SIDE ) );
	// assert( colIndex[0] == 4 );
	num = cm0->getStencil( 26, &entry, &colIndex );
	// assert( num == 1 ); assert( colIndex[0] == 9 );
	num = cm0->getStencil( 27, &entry, &colIndex );
	// assert( num == 1 ); assert( colIndex[0] == 14 );
	num = cm0->getStencil( 28, &entry, &colIndex );
	// assert( num == 1 ); assert( colIndex[0] == 19 );
	num = cm0->getStencil( 29, &entry, &colIndex );
	// assert( num == 1 ); assert( colIndex[0] == 24 );
}

void testMolTransferAcrossJunctions()
{
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );

	Id model = s->doCreate( "Neutral", Id(), "model", dims );
	pair< Id, Id > compt0 = makeComptForDiffusion( s, model, 0 );
	pair< Id, Id > compt1 = makeComptForDiffusion( s, model, 1 );
	SetGet1<Id>::set( compt0.second, "addJunction", compt1.second );
	unsigned int nj;
	nj = Field< unsigned int >::get( compt0.second, "num_junction" );
	assert( nj == 1 );
	nj = Field< unsigned int >::get( compt1.second, "num_junction" );
	assert( nj == 1 );
	Id jn0( "/model/compt_0/stoichA/junction" );
	assert( jn0 != Id() );
	Id jn1( "/model/compt_1/stoichA/junction" );
	assert( jn1 != Id() );
	unsigned int m = Field< unsigned int >::get( jn0, "numMeshEntries" );
	assert( m == 5 );
	m = Field< unsigned int >::get( jn1, "numMeshEntries" );
	assert( m == 5 );

	// Look up mol transfer here.
	vector< double > nInit( 25, -1.0 );
	nInit[4]=0; nInit[9]=1; nInit[14]=2; nInit[19]=3; nInit[24]=4;
	Id pool0( "/model/compt_0/A" );
	Field< double >::setVec( pool0, "nInit", nInit );
	nInit.clear();
	nInit.resize( 25, -2.0 );
	nInit[0] = 9; nInit[5] = 8; nInit[10] = 7; nInit[15] = 6; nInit[20] = 5;
	Id pool1( "/model/compt_1/A" );
	Field< double >::setVec( pool1, "nInit", nInit );

	s->doUseClock( "/model/compt_#/stoichA", "init",  3 );
	s->doSetClock( 3, 1.0 );
	s->doSetClock( 4, 1.0 );
	s->doReinit();
	s->doStart( 1.0 );
	GslStoich* gs0 = 
			reinterpret_cast< GslStoich* >( compt0.second.eref().data() );
	GslStoich* gs1 = 
			reinterpret_cast< GslStoich* >( compt1.second.eref().data() );
	assert( gs0->getNumLocalVoxels() == 25 );
	assert( gs0->getNumAllVoxels() == 30 );
	assert( gs0->getNumVarPools() == 1 );
	assert( gs0->getNumAllPools() == 1 );
	assert( gs1->getNumLocalVoxels() == 25 );
	assert( gs1->getNumAllVoxels() == 25 );
	assert( gs1->getNumVarPools() == 1 );
	assert( gs1->getNumAllPools() == 1 );

	for ( unsigned int i = 0 ; i < 5; ++i ) {
		assert( doubleEq( gs0->S(i * 5 + 4)[0], i ) ); // original0
		assert( doubleEq( gs1->S(i * 5 )[0], 9 - i ) ); // original1
	}

	for ( unsigned int i = 0 ; i < 5; ++i ) {
		// after checkin 4348, the diffusion coupling is asymmetric and
		// the second StoichPool does not receive pool# info.
		// assert( doubleEq( sp1->S(i + 25)[0], i ) ); // diffused original0
		assert( doubleEq( gs0->S(i + 25)[0], 9 - i ) ); //diffused original1
	}

	// Look up stencil here
	ChemMesh* cm0 = 
			reinterpret_cast< ChemMesh* >( compt0.first.eref().data() );
	ChemMesh* cm1 = 
			reinterpret_cast< ChemMesh* >( compt1.first.eref().data() );

	checkDiffusionStencil( cm0, cm1 );

	s->doDelete( model );
	cout << "." << flush;
}


// Defined in regressionTests/rtReacDiff.cpp
extern double checkNdimDiff( const vector< double >& conc, 
				double D, double t,
				double dx, double n, unsigned int cubeSide, bool doPrint );

// Builds a 15x15 matrix and sets off diffusion in one corner. The
// matrix is tiled as nine (3x3) GslStoich objects, each handling 5x5.
void testDiffusionAcrossJunctions()
{
	const double DIFFCONST = 1e-12;
	const double SIDE = 10e-6;
	const double DT = 10;
	const double RUNTIME = 500.0;
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	s->doSetClock( 4, DT );
	s->doSetClock( 5, DT );
	s->doSetClock( 6, DT );

	vector< int > dims( 1, 1 );

	Id model = s->doCreate( "Neutral", Id(), "model", dims );
	vector< pair< Id, Id > > compt( 9 );
	for ( unsigned int i = 0; i < 9; ++i )
		compt[i] = makeComptForDiffusion( s, model, i );

	for ( unsigned int k = 0; k < 9; ++k ) {
		int ix = k % 3;
		int iy = k / 3;
		if ( ix < 2 )
			SetGet1<Id>::set( compt[k].second, "addJunction", 
							compt[k+1].second );
		if ( iy < 2 )
			SetGet1<Id>::set( compt[k].second, "addJunction", 
							compt[k+3].second );
	}
	unsigned int nj;
	nj = Field< unsigned int >::get( compt[2].second, "num_junction" );
	assert( nj == 2 );
	nj = Field< unsigned int >::get( compt[3].second, "num_junction" );
	assert( nj == 3 );
	nj = Field< unsigned int >::get( compt[4].second, "num_junction" );
	assert( nj == 4 );
	nj = Field< unsigned int >::get( compt[5].second, "num_junction" );
	assert( nj == 3 );
	nj = Field< unsigned int >::get( compt[6].second, "num_junction" );
	assert( nj == 2 );

	for ( unsigned int i = 0; i < 9; ++i ) {
		stringstream ss;
		ss << "/model/compt_" << i << "/stoichA/junction";
		string jName = ss.str();
		Id jn( jName );
		assert( jn != Id() );
		unsigned int m = Field< unsigned int >::get( jn, "numMeshEntries" );
		assert( m == 5 );
		SolverJunction* sj = 
				reinterpret_cast< SolverJunction* >( jn.eref().data() );
		assert( sj->sendMeshIndex().size() == 5 );
		assert( sj->meshMap().size() == 5 );
		assert( doubleEq( sj->meshMap()[0].diffScale, SIDE ) );
	}
	/////////////////////////////////////////////////////////////
	// Check stencils
	GslStoich* gs0 = 
		reinterpret_cast< GslStoich* >( compt[0].second.eref().data() );
	const double* entry;
	const unsigned int* colIndex;
	unsigned int ns;
	assert( gs0->compartmentMesh()->getNumEntries() == 25 );
	ns = gs0->compartmentMesh()->getStencil( 0, &entry, &colIndex );
	assert( ns == 2 );
	assert( colIndex[0] == 1 );
	assert( colIndex[1] == 5 );
	ns = gs0->compartmentMesh()->getStencil( 4, &entry, &colIndex );
	assert( ns == 3 );
	assert( colIndex[0] == 3 );
	assert( colIndex[1] == 9 );
	assert( colIndex[2] == 25 );
	ns = gs0->compartmentMesh()->getStencil( 9, &entry, &colIndex );
	assert( ns == 4 );
	assert( colIndex[0] == 4 );
	assert( colIndex[1] == 8 );
	assert( colIndex[2] == 14 );
	assert( colIndex[3] == 26 );
	ns = gs0->compartmentMesh()->getStencil( 24, &entry, &colIndex );
	assert( ns == 4 );
	assert( colIndex[0] == 19 );
	assert( colIndex[1] == 23 );
	assert( colIndex[2] == 29 );
	assert( colIndex[3] == 34 );

	for ( unsigned int i = 25; i < 35; ++i ) {
		ns = gs0->compartmentMesh()->getStencil( i, &entry, &colIndex );
		assert( ns == 1 );
		assert( doubleEq( entry[0], SIDE ) );
	}

	ns = gs0->compartmentMesh()->getStencil( 25, &entry, &colIndex );
	assert( colIndex[0] == 4 );
	ns = gs0->compartmentMesh()->getStencil( 26, &entry, &colIndex );
	assert( colIndex[0] == 9 );
	ns = gs0->compartmentMesh()->getStencil( 27, &entry, &colIndex );
	assert( colIndex[0] == 14 );
	ns = gs0->compartmentMesh()->getStencil( 28, &entry, &colIndex );
	assert( colIndex[0] == 19 );
	ns = gs0->compartmentMesh()->getStencil( 29, &entry, &colIndex );
	assert( colIndex[0] == 24 );
	ns = gs0->compartmentMesh()->getStencil( 30, &entry, &colIndex );
	assert( colIndex[0] == 20 );
	ns = gs0->compartmentMesh()->getStencil( 31, &entry, &colIndex );
	assert( colIndex[0] == 21 );
	ns = gs0->compartmentMesh()->getStencil( 32, &entry, &colIndex );
	assert( colIndex[0] == 22 );
	ns = gs0->compartmentMesh()->getStencil( 33, &entry, &colIndex );
	assert( colIndex[0] == 23 );
	ns = gs0->compartmentMesh()->getStencil( 34, &entry, &colIndex );
	assert( colIndex[0] == 24 );
	
	/////////////////////////////////////////////////////////////
	// Check the solverJunction for gs0 and gs1.
	GslStoich* gs1 = 
		reinterpret_cast< GslStoich* >( compt[1].second.eref().data() );

	assert( gs0->getNumJunctions() == 2 );
	SolverJunction* sj = gs0->getJunction( 0 );
	assert ( sj->abutMeshIndex().size() == 5 );
	assert ( sj->abutMeshIndex()[0] == 25 );
	assert ( sj->abutMeshIndex()[1] == 26 );
	assert ( sj->abutMeshIndex()[2] == 27 );
	assert ( sj->abutMeshIndex()[3] == 28 );
	assert ( sj->abutMeshIndex()[4] == 29 );
	assert ( sj->sendMeshIndex().size() == 5 );
	assert ( sj->sendMeshIndex()[0] == 4 );
	assert ( sj->sendMeshIndex()[1] == 9 );
	assert ( sj->sendMeshIndex()[2] == 14 );
	assert ( sj->sendMeshIndex()[3] == 19 );
	assert ( sj->sendMeshIndex()[4] == 24 );

	sj = gs0->getJunction( 1 );
	assert ( sj->abutMeshIndex().size() == 5 );
	assert ( sj->abutMeshIndex()[0] == 30 );
	assert ( sj->abutMeshIndex()[1] == 31 );
	assert ( sj->abutMeshIndex()[2] == 32 );
	assert ( sj->abutMeshIndex()[3] == 33 );
	assert ( sj->abutMeshIndex()[4] == 34 );
	assert ( sj->sendMeshIndex().size() == 5 );
	assert ( sj->sendMeshIndex()[0] == 20 );
	assert ( sj->sendMeshIndex()[1] == 21 );
	assert ( sj->sendMeshIndex()[2] == 22 );
	assert ( sj->sendMeshIndex()[3] == 23 );
	assert ( sj->sendMeshIndex()[4] == 24 );

	assert( gs1->getNumJunctions() == 3 );
	sj = gs1->getJunction( 0 );
	assert ( sj->abutMeshIndex().size() == 0 );
	assert ( sj->sendMeshIndex().size() == 5 );
	assert ( sj->sendMeshIndex()[0] == 0 );
	assert ( sj->sendMeshIndex()[1] == 5 );
	assert ( sj->sendMeshIndex()[2] == 10 );
	assert ( sj->sendMeshIndex()[3] == 15 );
	assert ( sj->sendMeshIndex()[4] == 20 );

	/////////////////////////////////////////////////////////////
	// Put in some reference starting values to see if data is 
	// going across correctly.
	Id pool0( "/model/compt_0/A" );
	vector< double > temp( 25 );
	for ( unsigned int i = 0; i < 25; ++i )
		temp[i] = i + 100;
	Field< double >::setVec( pool0, "nInit", temp );

	Id pool1( "/model/compt_1/A" );
	for ( unsigned int i = 0; i < 25; ++i )
		temp[i] = i + 200;
	Field< double >::setVec( pool1, "nInit", temp );

	Id pool3( "/model/compt_3/A" );
	for ( unsigned int i = 0; i < 25; ++i )
		temp[i] = i + 300;
	Field< double >::setVec( pool3, "nInit", temp );

	assert( gs0->getNumAllVoxels() == 35 );
	vector< double > svec( 35, 0 );

	///////////////////////////////////////////////////////////////////
	// Call the 'init' method to be sure that all the pool concs are being
	// sent over to the expected locations. Messy, disable the 'process'
	// clock.

	s->doReinit();
	s->doSetClock( 5, 0 );
	s->doSetClock( 6, 0 );
	for ( unsigned int i = 0; i < 25; ++i ) {
		svec[i] = gs0->S(i)[0];
		assert( doubleEq( svec[i], i + 100 ) );
	}
	s->doStart( DT * 0.5 );
	
	// ProcInfo info;
	/*
	for ( unsigned int i = 0; i < 9; ++i ) {
		Eref e = compt[i].second.eref();
		GslStoich* gs = reinterpret_cast< GslStoich* >( e.data());
		gs->init( e, s->procInfo() );
	}
	Qinfo::clearQ( 0 );
	Qinfo::clearQ( 0 );
	*/
	for ( unsigned int i = 0; i < 35; ++i )
		svec[i] = gs0->S(i)[0];
	assert( doubleEq( svec[0], 100 ) );
	assert( doubleEq( svec[1], 101 ) );
	assert( doubleEq( svec[2], 102 ) );
	assert( doubleEq( svec[3], 103 ) );
	assert( doubleEq( svec[4], 104 ) );
	assert( doubleEq( svec[25], 200 ) );
	assert( doubleEq( svec[26], 205 ) );
	assert( doubleEq( svec[27], 210 ) );
	assert( doubleEq( svec[28], 215 ) );
	assert( doubleEq( svec[29], 220 ) );
	assert( doubleEq( svec[30], 300 ) );
	assert( doubleEq( svec[31], 301 ) );
	assert( doubleEq( svec[32], 302 ) );
	assert( doubleEq( svec[33], 303 ) );
	assert( doubleEq( svec[34], 304 ) );

	for ( unsigned int i = 0; i < 35; ++i )
		svec[i] = gs1->S(i)[0];
	assert( doubleEq( svec[25], 0 ) ); // Base conc for next compt is 0.
	assert( doubleEq( svec[26], 0 ) );
	assert( doubleEq( svec[27], 0 ) );
	assert( doubleEq( svec[28], 0 ) );
	assert( doubleEq( svec[29], 0 ) );

	GslStoich* gs3 = 
		reinterpret_cast< GslStoich* >( compt[3].second.eref().data() );
	for ( unsigned int i = 0; i < 35; ++i )
		svec[i] = gs3->S(i)[0];
	assert( doubleEq( svec[25], 0 ) );
	assert( doubleEq( svec[26], 0 ) );
	assert( doubleEq( svec[27], 0 ) );
	assert( doubleEq( svec[28], 0 ) );
	assert( doubleEq( svec[29], 0 ) );

	///////////////////////////////////////////////////////////////////
	// Run it for 1 timestep to see that the calculations are going right.
	
	s->doSetClock( 5, DT );
	s->doSetClock( 6, DT );
	s->doReinit();
	for ( unsigned int i = 0; i < 25; ++i ) {
		svec[i] = gs0->S(i)[0];
		assert( doubleEq( svec[i], i + 100 ) );
	}

	s->doStart( DT * 0.5 );
	for ( unsigned int i = 0; i < 35; ++i )
		svec[i] = gs0->S(i)[0];

	assert( doubleEq( svec[0], 100 + 0.1 + 0.5 ) );
	assert( doubleEq( svec[1], 101 + 0.5 ) );
	assert( doubleEq( svec[2], 102 + 0.5 ) );
	assert( doubleEq( svec[3], 103 + 0.5 ) );
	assert( doubleEq( svec[4], 104 - 0.1 + 0.5 + 0.1 * ( 200 - 104) ) );
	assert( doubleEq( svec[25], 200 - 0.1 * ( 200 - 104 ) ) );

	assert( doubleEq( svec[5], 105 + 0.1 ) );
	assert( doubleEq( svec[6], 106 ) );
	assert( doubleEq( svec[7], 107 ) );
	assert( doubleEq( svec[8], 108 ) );
	assert( doubleEq( svec[9], 109 - 0.1 + 0.1 *( 205 - 109 ) ) );
	assert( doubleEq( svec[26], 205 - 0.1 * ( 205 - 109 ) ) );

	assert( doubleEq( svec[20], 120 + 0.1 - 0.5 + 0.1 * ( 300 - 120 ) ) );
	assert( doubleEq( svec[21], 121 - 0.5 + 0.1 * ( 301 - 121  ) ) );
	assert( doubleEq( svec[22], 122 - 0.5 + 0.1 * ( 302 - 122 ) ) );
	assert( doubleEq( svec[23], 123 - 0.5 + 0.1 * ( 303 - 123 ) ) );
	assert( doubleEq( svec[24], 
					124 - 0.5 - 0.1 + 
					0.1 * ( 220 - 124 ) +
				    0.1 * ( 304 - 124 ) ) );
	assert( doubleEq( svec[27], 210 - 0.1 * ( 210 - 114 ) ) );
	assert( doubleEq( svec[28], 215 - 0.1 * ( 215 - 119 ) ) );
	assert( doubleEq( svec[29], 220 - 0.1 * ( 220 - 124 ) ) );
	assert( doubleEq( svec[34], 304 - 0.1 * ( 304 - 124 ) ) );
	

	/////////////////////////////////////////////////////////////
	// Now do regular diffusion with a point source.
	/////////////////////////////////////////////////////////////

	temp.clear();
	temp.resize( 25, 0.0 );
	Field< double >::setVec( pool0, "nInit", temp );
	Field< double >::setVec( pool1, "nInit", temp );
	Field< double >::setVec( pool3, "nInit", temp );
	Field< double >::set( ObjId( pool0, 0 ), "nInit", 1.0 );

	s->doReinit();
	s->doStart( RUNTIME );

	// Now do the comparison with the reference.
	vector< Id > pools;
	vector< double > val( 15 * 15 , -1.0 );
	double tot = 0;
	for ( unsigned int i = 0; i < 9; ++i ) {
		stringstream ss;
		ss << "/model/compt_" << i << "/A";
		Id p( ss.str() );
		assert( p != Id() );
		pools.push_back( p );
		vector< double > n;
		Field< double >::getVec( p, "n", n );
		unsigned int ix = ( i % 3 ) * 5;
		unsigned int iy = ( i / 3 ) * 5;
		for ( unsigned int jy = 0; jy < 5; ++jy ) {
			for ( unsigned int jx = 0; jx < 5; ++jx ) {
				unsigned int j = jx + jy * 5;
				assert( j < 25 );
				unsigned int k = ix + jx + 15 * ( iy + jy );
				assert( k < 225 );
				assert( val[k] < -0.99); // Should not have been filled yet.
				val[k] = n[j];
				tot += n[j];
			}
		}
	}
	assert( doubleApprox( tot, 1.0 ) );
	// cout << "testDiffusionAcrossJunctions(): tot = " << tot << endl;

	// The last arg is a flag to say if output should be printed.
	double err = checkNdimDiff( val, DIFFCONST, RUNTIME, SIDE, 2, 15, 
					false);
	assert( err < 0.005 );

	s->doDelete( model );
	cout << "." << flush;
}

void fillVal( vector< double >& val, double expectedTot )
{
	// Now do the comparison with the reference.
	vector< Id > pools;
	val.clear();
	val.resize( 10, -1.0 );
	double tot = 0;
	for ( unsigned int i = 0; i < 2; ++i ) {
		stringstream ss;
		ss << "/model/compt_" << i << "/A";
		Id p( ss.str() );
		assert( p != Id() );
		pools.push_back( p );
		vector< double > n;
		Field< double >::getVec( p, "n", n );
		for ( unsigned int j = 0; j < 5; ++j ) {
			unsigned int k = i * 5 + j;
			assert( val[k] < -0.99); // Should not have been filled yet.
			val[k] = n[j];
			tot += n[j];
		}
	}
	assert( doubleApprox( tot, expectedTot ) );
}

// Builds a 1-Dim matrix with 10 elements total, split between two
// GslStoich objects, each handling 5. 
void testOneDimDiffusionAcrossJunctions()
{
	const double DIFFCONST = 1e-12;
	const double SIDE = 10e-6;
	const double DT = 10;
	const double RUNTIME = 500.0;
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	s->doSetClock( 4, DT );
	s->doSetClock( 5, DT );
	s->doSetClock( 6, DT );

	vector< int > dims( 1, 1 );

	Id model = s->doCreate( "Neutral", Id(), "model", dims );
	vector< pair< Id, Id > > compt( 9 );
	compt[0] = makeOneDimComptForDiffusion( s, model, 0 );
	compt[1] = makeOneDimComptForDiffusion( s, model, 1 );

	SetGet1<Id>::set( compt[0].second, "addJunction", compt[1].second );
	unsigned int nj;
	nj = Field< unsigned int >::get( compt[0].second, "num_junction" );
	assert( nj == 1 );
	nj = Field< unsigned int >::get( compt[1].second, "num_junction" );
	assert( nj == 1 );

	Id jn0( "/model/compt_0/stoichA/junction" );
	Id jn1( "/model/compt_1/stoichA/junction" );
	unsigned int m = Field< unsigned int >::get( jn0, "numMeshEntries" );
	assert( m == 1 );
	SolverJunction* sj = 
				reinterpret_cast< SolverJunction* >( jn0.eref().data() );
	assert( sj->sendMeshIndex().size() == 1 );
	assert( sj->meshMap().size() == 1 );
	assert( doubleEq( sj->meshMap()[0].first, 0 ) );
	assert( doubleEq( sj->meshMap()[0].second, 4 ) );
	assert( doubleEq( sj->meshMap()[0].diffScale, SIDE ) );

	sj = reinterpret_cast< SolverJunction* >( jn1.eref().data() );
	assert( sj->sendMeshIndex().size() == 1 );
	assert( sj->meshMap().size() == 1 );
	assert( doubleEq( sj->meshMap()[0].first, 0 ) );
	assert( doubleEq( sj->meshMap()[0].second, 0 ) );
	assert( doubleEq( sj->meshMap()[0].diffScale, SIDE ) );

	///////////////////////////////////////////////////////////////////
	Id pool0( "/model/compt_0/A" );
	vector< double > temp( 5 );
	for ( unsigned int i = 0; i < 5; ++i )
		temp[i] = i + 100;
	Field< double >::setVec( pool0, "nInit", temp );

	Id pool1( "/model/compt_1/A" );
	for ( unsigned int i = 0; i < 5; ++i )
		temp[i] = i + 105;
	Field< double >::setVec( pool1, "nInit", temp );

	GslStoich* gs0 = 
		reinterpret_cast< GslStoich* >( compt[0].second.eref().data() );
	assert( gs0->getNumAllVoxels() == 6 );
	vector< double > svec( 6, 0 );

	s->doReinit();
	for ( unsigned int i = 0; i < 6; ++i ) {
		svec[i] = gs0->S(i)[0];
		if ( i != 5 )
			assert( doubleEq( svec[i], i + 100 ) );
	}

	s->doStart( DT * 0.5 );
	// s->doStart( RUNTIME );
	//
	for ( unsigned int i = 0; i < 6; ++i )
		svec[i] = gs0->S(i)[0];

	assert( doubleEq( svec[0], 100 + 0.1 ) );
	assert( doubleEq( svec[1], 101 ) );
	assert( doubleEq( svec[2], 102 ) );
	assert( doubleEq( svec[3], 103 ) );
	assert( doubleEq( svec[4], 104 ) );
	assert( doubleEq( svec[5], 105 - 0.1 ) );

	GslStoich* gs1 = 
		reinterpret_cast< GslStoich* >( compt[1].second.eref().data() );
	for ( unsigned int i = 0; i < 5; ++i )
		svec[i] = gs1->S(i)[0];
	assert( doubleEq( svec[0], 105 ) );
	assert( doubleEq( svec[1], 106 ) );
	assert( doubleEq( svec[2], 107 ) );
	assert( doubleEq( svec[3], 108 ) );
	assert( doubleEq( svec[4], 109 - 0.1 ) );

	vector< double > val;
	fillVal( val, 1045.0 );
	///////////////////////////////////////////////////////////////////
	// Check the stencils
	const double* entry;
	const unsigned int* colIndex;
	unsigned int ns;
	assert( gs0->compartmentMesh()->getNumEntries() == 5 );
	ns = gs0->compartmentMesh()->getStencil( 0, &entry, &colIndex );
	assert( ns == 1 );
	assert( colIndex[0] == 1 );
	ns = gs0->compartmentMesh()->getStencil( 1, &entry, &colIndex );
	assert( ns == 2 );
	assert( colIndex[0] == 0 );
	assert( colIndex[1] == 2 );
	ns = gs0->compartmentMesh()->getStencil( 2, &entry, &colIndex );
	assert( ns == 2 );
	assert( colIndex[0] == 1 );
	assert( colIndex[1] == 3 );
	ns = gs0->compartmentMesh()->getStencil( 3, &entry, &colIndex );
	assert( ns == 2 );
	assert( colIndex[0] == 2 );
	assert( colIndex[1] == 4 );
	ns = gs0->compartmentMesh()->getStencil( 4, &entry, &colIndex );
	assert( ns == 2 );
	assert( colIndex[0] == 3 );
	assert( colIndex[1] == 5 );
	ns = gs0->compartmentMesh()->getStencil( 5, &entry, &colIndex );
	assert( ns == 1 );
	assert( colIndex[0] == 4 );

	assert( gs1->getNumAllVoxels() == 5 );
	assert( gs1->compartmentMesh()->getNumEntries() == 5 );
	ns = gs1->compartmentMesh()->getStencil( 0, &entry, &colIndex );
	assert( ns == 1 );
	assert( colIndex[0] == 1 );
	ns = gs1->compartmentMesh()->getStencil( 1, &entry, &colIndex );
	assert( ns == 2 );
	assert( colIndex[0] == 0 );
	assert( colIndex[1] == 2 );
	ns = gs1->compartmentMesh()->getStencil( 2, &entry, &colIndex );
	assert( ns == 2 );
	assert( colIndex[0] == 1 );
	assert( colIndex[1] == 3 );

	///////////////////////////////////////////////////////////////////
	// Now redo with proper diffusion.
	temp.clear();
	temp.resize( 5, 0 );
	Field< double >::setVec( pool0, "nInit", temp );
	Field< double >::setVec( pool1, "nInit", temp );
	Field< double >::set( ObjId( pool0, 0 ), "nInit", 1.0 );
	s->doReinit();
	s->doStart( RUNTIME );
	fillVal( val, 1.0 );

	// The last arg is a flag to say if output should be printed.
	double err = checkNdimDiff( val, DIFFCONST, RUNTIME, SIDE, 1, 10, 
					false);
	assert( err < 0.005 );

	s->doDelete( model );
	cout << "." << flush;
}

/**
 * Tests the assignment of pools and reacs to local and non-loncal
 * compartments during setup. Also checks that an off-solver pool is
 * correctly tracked down.
 */
void testStoichCoreCompartmentAssignment()
{
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );
	Id model = makeInterMeshReac( s );
	// Create solvers for meshA and meshB.
	Id meshA( "/model/meshA" );
	assert ( meshA != Id() );
	Id stoichA = s->doCreate( "GslStoich", meshA, "stoichA", dims );
	assert ( stoichA != Id() );
	string path = "/model/meshA/##";
	Field< string >::set( stoichA, "path", path );

	GslStoich* gs = 
			reinterpret_cast< GslStoich* >( stoichA.eref().data() );

	const StoichCore* sc = gs->coreStoich();
	const vector< Id >& pools =  sc->getOffSolverPools();
	assert( pools.size() == 1 );
	Id poolA( "/model/meshA/A" );
	assert( poolA != Id() );
	Id poolB( "/model/meshB/B" );
	assert( poolB != Id() );
	assert( poolB == pools[0] );
	assert( sc->convertIdToPoolIndex( poolA ) == 0 );
	assert( sc->convertIdToPoolIndex( poolB ) == 1 );

	/// Need to check for reacs here.


	s->doDelete( model );
	cout << "." << flush;
}

void testExtractCompts()
{
	pair< Id, Id > extractCompts( const vector< Id >& compts );

	vector< Id >test;
	pair< Id, Id > ret = extractCompts( test );
	assert( ret.first == Id() && ret.second == Id() );

	test.push_back( Id( 1 ) );
	ret = extractCompts( test );
	assert( ret.first == Id( 1 ) && ret.second == Id() );

	test.push_back( Id( 1 ) );
	ret = extractCompts( test );
	assert( ret.first == Id( 1 ) && ret.second == Id() );

	test.push_back( Id( 1 ) );
	ret = extractCompts( test );
	assert( ret.first == Id( 1 ) && ret.second == Id() );

	test.push_back( Id( 2 ) );
	ret = extractCompts( test );
	assert( ret.first == Id( 1 ) && ret.second == Id( 2 ) );

	test.push_back( Id( 1 ) );
	ret = extractCompts( test );
	assert( ret.first == Id( 1 ) && ret.second == Id( 2 ) );

	cout << "." << flush;
}

void testKineticSolvers()
{
	testExtractCompts();
	testGslStoich();
	testStoichCoreCompartmentAssignment();
	testJunctionSetup();
	testMatchMeshEntries();
}

void testKineticSolversProcess()
{
	testMolTransferAcrossJunctions();
	testOneDimDiffusionAcrossJunctions();
	testDiffusionAcrossJunctions();
	testInterMeshReac();
}
