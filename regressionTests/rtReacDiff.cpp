/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <fstream>
#include <iostream>
#include <iomanip>
#include "header.h"
#include "Shell.h"
#include "../kinetics/ReadCspace.h"
#include "../manager/SimManager.h"

/**
 * This regression test takes a simple model and replicates it over a
 * 2x2x2 cube mesh. Tests the initial creation and replication of model,
 * and that the duplicates all run properly.
 * Also tests ReadCspace.
 */
static void rtReplicateModels()
{
	static double CONCSCALE = 1e-3; // convert from uM to mM.
	// These values are obtained by running the model explicitly in a
	// single compartment, with the specified initial conditions.
	static double expectedValueAtOneSec[] =
		{ 0.7908, 1.275, 1.628, 1.912, 2.154, 2.369, 2.564, 2.744 };
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );

	ReadCspace rc;
	Id modelId = rc.readModelString( "|AabX|Jacb| 1 1 1 0.01 0.1 0.1 1",
		"model", Id(), "gsl" );

	Id a( "/model/kinetics/a" );
	Id b( "/model/kinetics/b" );
	Id c( "/model/kinetics/c" );
	Id cplx( "/model/kinetics/c/Jacb/Jacb_cplx" );
	Id mesh( "/model/kinetics/mesh" );
	assert( a != Id() );
	assert( b != Id() );
	assert( c != Id() );
	assert( cplx != Id() );
	assert( mesh != Id() );
	assert( a.element()->dataHandler()->localEntries() == 1 );
	bool ret = Field< double >::set( a, "diffConst", 0 );
	assert( ret );
	ret = Field< double >::set( b, "diffConst", 0 );
	assert( ret );
	ret = Field< double >::set( c, "diffConst", 0 );
	assert( ret );
	ret = Field< double >::set( cplx, "diffConst", 0 );
	assert( ret );

	Id J( "/model/kinetics/c/Jacb" );
	assert( J != Id() );
	double temp = Field< double >::get( J, "Km" );
	assert( doubleEq( temp, 1 * CONCSCALE ) );
	temp = Field< double >::get( J, "kcat" );
	assert( doubleEq( temp, 0.1 ) );
	temp = Field< double >::get( J, "ratio" );
	assert( doubleEq( temp, 4 ) );
	temp = Field< double >::get( J, "concK1" );
	assert( doubleEq( temp, 0.5 / CONCSCALE ) );
	temp = Field< double >::get( J, "k2" );
	assert( doubleEq( temp, 0.4 ) );
	temp = Field< double >::get( J, "k3" );
	assert( doubleEq( temp, 0.1 ) );

	assert( mesh.element()->dataHandler()->localEntries() == 1 );
	assert( a.element()->dataHandler()->localEntries() == 1 );
	vector< double > checkInit;
	Field< double >::getVec( mesh, "size", checkInit );
	assert( checkInit.size() == 1 );
	for ( unsigned int i = 0; i < checkInit.size(); ++i )
		assert( doubleEq( checkInit[i], 1e-18 ) );

	// Assign the CubeMesh dims as a 2x2x2 cube with mesh size of 1 um
	// to match the default in the ReadCspace.
	Id compt( "/model/kinetics" );
	assert( compt != Id() );
	vector< double > coords( 9, 0 );
	coords[0] = coords[1] = coords[2] = 0;
	coords[3] = coords[4] = coords[5] = 4e-6;
	coords[6] = coords[7] = coords[8] = 2e-6;
	
	ret = Field< bool >::set( compt, "preserveNumEntries", false );
	assert( ret );
	ret = Field< vector< double > >::set( compt, "coords", coords );
	assert( ret );
	Qinfo::waitProcCycles( 2 );
	assert( mesh.element()->dataHandler()->localEntries() == 8 );
	assert( a.element()->dataHandler()->localEntries() == 8 );

	Field< double >::getVec( mesh, "size", checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], 8e-18 ) );

	Field< double >::getVec( a, "size", checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], 8e-18 ) );

	Field< double >::getVec( a, "nInit", checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], 8 * CONCSCALE * NA * 1e-18 ) );

	Field< double >::getVec( a, "concInit", checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], 1 * CONCSCALE ) );

	Field< double >::getVec( a, "diffConst", checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], 0 ) );

	Field< double >::getVec( cplx, "concInit", checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], 0 ) );

	// Here we create the stoich
	Id stoich( "/model/stoich/stoichCore" );
	unsigned int numVarMols = Field< unsigned int >::get( 
		stoich, "nVarPools" );
	assert ( numVarMols == 4 ); // 2 mols + 2 enz

	Id gsl( "/model/stoich" );
	assert( gsl != Id() );
	assert( gsl.element()->dataHandler()->localEntries() == 1 );

	checkInit.resize( 0 );
	Field< double >::getVec( Id( "/model/kinetics/b" ), "concInit",
		checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], 1 * CONCSCALE ) );

	Field< double >::getVec( Id( "/model/kinetics/c" ), "concInit", checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], 1 * CONCSCALE ) );

	// Set up novel init conditions
	vector< double > init( 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		init[i] = ( i + 1 ) * CONCSCALE;
	ret = Field< double >::setVec( a, "concInit", init );
	ret = Field< double >::setVec( b, "concInit", init );
	ret = Field< double >::setVec( c, "concInit", init );
	assert( ret);

	Field< double >::getVec( a, "concInit", checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], init[i] ) );

	// Create the plots.
	vector< int > dims( 1, 8 );
	Id plots = shell->doCreate( "Table", modelId, "table", dims );
	assert( plots != Id() );
	MsgId mid = shell->doAddMsg( "OneToOne", plots, "requestData", a, "get_conc" );
	assert( mid != Msg::bad );

	// Set up scheduling.
	for ( unsigned int i = 0; i < 10; ++i )
		shell->doSetClock( i, 0.1 );
	// shell->doUseClock( "/kinetics/stoich/gsl", "process", 0 );
	shell->doUseClock( "/model/table", "process", 8 );
	shell->doReinit();
	shell->doStart( 10 );

	unsigned int size = Field< unsigned int >::get( plots, "size" );
	// cout << "size = " << size << endl;
	assert( size == 101 ); // Note that dt was 1.

	for ( unsigned int i = 0; i < 8; ++i ) {
		ObjId oid( plots, i );
		string name( "mesh" );
		name += '0' + i;
		ret = SetGet2< string, string >::set( oid, "xplot", "check.plot", 
			name );
		assert( ret );

		//Look up step # 10, starting from 0.
		double y = LookupField< unsigned int, double >::get( oid, "y", 10); 
		assert( doubleApprox( expectedValueAtOneSec[i] * CONCSCALE, y ) );
	}

	/////////////////////////////////////////////////////////////////////
	shell->doDelete( modelId );
	cout << "." << flush;
}


/**
 * The analytical solution for 1-D diffusion is:
 * 		c(x,t) = ( c0 / 2 / sqrt(PI.D.t) ).exp(-x^2/(4Dt)
 * c0 is the total amount of stuff.
 * In these sims the total amount is 1 uM conc in 1 compartment if input
 * is in middle.
 * The total amount is 1 uM conc in 2 compartments if input is at a corner
 * because the edge acts like a mirror.
 */
double checkDiff( const vector< double >& conc, 
	double D, double t, double dx)
{
	// static const double scaleFactor = sqrt( PI * D );
	// const double scaleFactor = 0.5 * dx; // Case for input in middle
	// int mid = conc.size() / 2; // Case for input in middle

	const double scaleFactor = dx; 
	int mid = 0; // Case for input at end.
	double err = 0;

	double tot1 = 0;
	double tot2 = 0;

	for ( unsigned int j = 0; j < conc.size(); ++j ) {
		int i = static_cast< int >( j );
		double x = ( i - mid ) * dx + dx /2;
		double y = scaleFactor * 
			( 1.0 / sqrt( PI * D * t ) ) * exp( -x * x / ( 4 * D * t ) );
		//assert( doubleApprox( conc[j], y ) );
		//cout << endl << t << "	" << j << ":	" << y << "	" << conc[j];
		err += ( y - conc[j] ) * ( y - conc[j] );
		tot1 += y;
		tot2 += conc[j];
	}
	// cout << t << "	" << tot1 << "	" << tot2 << endl;
	return sqrt( err );
}

/**
 * Very similar to the testSimManager.cpp::testRemeshing()
 * But it uses CylMesh.
 */
static void testDiff1D()
{
	// Diffusion length in mesh entries
	static const unsigned int diffLength = 20; 
	static const double dt = 0.05;
	static const double dx = 0.5e-6;
	static const double D = 1e-12;

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );

	vector< double > coords( 9, 0 );
	coords[3] = diffLength * dx;
	coords[6] = coords[7] = coords[8] = dx; // r0, r1, lambda

	Id mgr = shell->doCreate( "SimManager", Id(), "diff", dims );
	assert( mgr != Id() );
	SimManager* sm = reinterpret_cast< SimManager* >( mgr.eref().data() );
	sm->setPlotDt( dt );
	for ( unsigned int i = 0; i < 10; ++i )
			shell->doSetClock( i, dt );
	sm->makeStandardElements( mgr.eref(), 0, "CylMesh" );
	Id kinetics( "/diff/kinetics" );
	assert( kinetics != Id() );
	Id a = shell->doCreate( "Pool", kinetics, "a", dims );
	assert( a != Id() );
	bool ret = Field< double >::set( a, "diffConst", D );
	assert( ret );

	Id meshEntry( "/diff/kinetics/mesh" );
	assert( meshEntry != Id() );
	assert( meshEntry.element()->dataHandler()->localEntries() == 1 );
	MsgId mid = shell->doAddMsg( "OneToOne", a, "mesh", meshEntry, "mesh" );
	assert( mid != Msg::bad );

	Qinfo q;
	q.setThreadNum( ScriptThreadNum );
	sm->build( mgr.eref(), &q, "gsl" );

	ret = Field< vector< double > >::set( kinetics, "coords", coords );
	assert( ret );
	Qinfo::waitProcCycles( 2 );

	// This should assign the same init conc to the new pool objects.
	assert( a.element()->dataHandler()->localEntries() == diffLength );
	assert( meshEntry.element()->dataHandler()->localEntries() == diffLength );

	Field< double >::setRepeat( a, "concInit", 0 );
	Field< double >::set( ObjId( a, 0 ), "concInit", 1 );

	Id gsl( "/diff/stoich" );
	assert( gsl != Id() );
	Id stoich( "/diff/stoich/stoichCore" );
	assert( stoich != Id() );

    shell->doReinit();

	for ( unsigned int i = 0; i < 10; ++i ) {
		shell->doStart( 1 );
		vector< double > conc;
		Field< double >::getVec( a, "conc", conc );
		assert( conc.size() == diffLength );
		double ret = checkDiff( conc, D, i + 1, dx );
//		cout << "root sqr Error on t = " << i + 1 << " = " << ret << endl;
		assert ( ret < 0.01 );
	}

	shell->doDelete( mgr );

	cout << "." << flush;
}

/**
 * Checks calculations in n-dimensions. Uses point at corner as input.
 * Assumes cube.
 * 		c(x,t) = ( c0 / 2 / sqrt(PI.D.t) ).exp(-x^2/(4Dt)
 */
double checkNdimDiff( const vector< double >& conc, double D, double t, 
		double dx, double n, unsigned int cubeSide, bool doPrint = false )
{
	const double scaleFactor = pow( dx, n); 
	double err = 0;
	unsigned int dimY = 1;
	unsigned int dimZ = 1;
	if ( n >= 2 ) dimY = cubeSide;
	if ( n == 3 ) dimZ = cubeSide;

	double tot1 = 0.0;
	double tot2 = 0.0;
	if ( doPrint )
		cout << endl << "time	(i,j,k)	r	analytic	sim\n";
	for ( unsigned int i = 0; i < dimZ; ++i ) {
		double z = 0.0; 
		if ( n > 2.5 )
			z = i * dx + dx * 0.5;
		for ( unsigned int j = 0; j < dimY; ++j ) {
			double y = 0.0; 
			if ( n > 1.5 )
				y = j * dx + dx * 0.5;
			for ( unsigned int k = 0; k < cubeSide; ++k ) {
				double x = k * dx + dx * 0.5;
				double rsq = x * x + y * y + z * z;
				unsigned int index = ( i * cubeSide + j ) * cubeSide + k;
				double c = scaleFactor * pow( PI * D * t, -n/2 ) * 
					exp( -rsq / ( 4 * D * t ) );
				if ( doPrint )
					cout << endl << t << "	(" << 
							i << "," << j << "," << k  << 
							"), r= " << rsq << "	" << 
							c << "	" << conc[index];
				err += ( c - conc[index] ) * ( c - conc[index] );
				tot1 += c;
				tot2 += conc[index];
			}
		}
	}
	assert ( tot1 < 1.001 && tot1 > 0.97 );
	assert ( doubleApprox( tot2, 1.0 ) );
	// cout << "t = " << t << ";	tot1 = " << tot1 << ";	tot2 = " << tot2 << endl;

	return sqrt( err );
}


static void testDiffNd( unsigned int n )
{
	// Diffusion length in mesh entries
	static const unsigned int cubeSide = 15; 
	static const double dt = 0.02;
	static const double dx = 0.5e-6;
	static const double D = 1e-12;

	unsigned int vol = cubeSide;

	if ( n == 2 ) 
		vol *= cubeSide;

	if ( n == 3 ) 
		vol *= cubeSide * cubeSide;

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );

	Id mgr = shell->doCreate( "SimManager", Id(), "diff", dims );
	assert( mgr != Id() );
	SimManager* sm = reinterpret_cast< SimManager* >( mgr.eref().data() );
	sm->setPlotDt( dt );
	sm->makeStandardElements( mgr.eref(), 0, "CubeMesh" );
	Id kinetics( "/diff/kinetics" );
	assert( kinetics != Id() );
	Id a = shell->doCreate( "Pool", kinetics, "a", dims );
	assert( a != Id() );
	bool ret = Field< double >::set( a, "diffConst", D );
	assert( ret );

	Id meshEntry( "/diff/kinetics/mesh" );
	assert( meshEntry != Id() );
	assert( meshEntry.element()->dataHandler()->localEntries() == 1 );
	MsgId mid = shell->doAddMsg( "OneToOne", a, "mesh", meshEntry, "mesh" );
	assert( mid != Msg::bad );

	Qinfo q;
	q.setThreadNum( ScriptThreadNum );
	sm->build( mgr.eref(), &q, "gsl" );

	ret = Field< bool >::set( kinetics, "preserveNumEntries", false );
	vector< double > coords( 9, dx );
	coords[0] = coords[1] = coords[2] = 0;
	coords[3] = cubeSide * dx;
	if ( n >= 2 )
		coords[4] = cubeSide * dx;
	if ( n == 3 )
		coords[5] = cubeSide * dx;

	ret = Field< vector< double > >::set( kinetics, "coords", coords );
	assert( ret );
	Qinfo::waitProcCycles( 2 );

	// This should assign the same init conc to the new pool objects.
	assert( a.element()->dataHandler()->localEntries() == vol );
	assert( meshEntry.element()->dataHandler()->localEntries() == vol );
	dims[0] = vol;

	Field< double >::setRepeat( a, "concInit", 0 );
	Field< double >::set( ObjId( a, 0 ), "concInit", 1 );

    shell->doSetClock( 0, dt );
    shell->doSetClock( 1, dt );
    shell->doSetClock( 2, dt );
    shell->doSetClock( 3, 0 ); 

    shell->doReinit();

	for ( unsigned int i = 0; i < 4; ++i ) {
		shell->doStart( 1 );
		vector< double > conc;
		Field< double >::getVec( a, "conc", conc );
		assert( conc.size() == vol );
		double ret = checkNdimDiff( conc, D, i + 1, dx, n, cubeSide );
		// cout << n << "dimensions: root sqr Error on t = " << i + 1 << " = " << ret << endl;
		assert ( ret < 0.005 );
	}

	shell->doDelete( mgr );

	cout << "." << flush;
}

// Still to complete. The idea is to simply test for conservation.
static void testReacDiffNd( unsigned int n )
{
	static const bool doPrint = 0;
	// Diffusion length in mesh entries
	static const unsigned int cubeSide = 10; 
	static const double dt = 0.01;
	static const double dx = 0.5e-6;
	static const double D1 = 1e-12;
	static const double D2 = 1e-13;
	static const double D3 = 1e-13;

	unsigned int vol = cubeSide;

	if ( n == 2 ) 
		vol *= cubeSide;

	if ( n == 3 ) 
		vol *= cubeSide * cubeSide;

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );

	ReadCspace rc;
	Id mgr = rc.readModelString( "|AabX|Jacb| 1 1 1 0.01 0.1 1 1",
		"diff", Id(), "gsl" );

	// Id kinetics = shell->doCreate( "Neutral", Id(), "kinetics", dims );
	// Id a = shell->doCreate( "Pool", kinetics, "a", dims );
	Id a( "/diff/kinetics/a" );
	assert( a != Id() );
	Id b( "/diff/kinetics/b" );
	assert( b != Id() );
	Id c( "/diff/kinetics/c" );
	assert( c != Id() );
	bool ret = Field< double >::set( a, "diffConst", D1 );
	assert( ret );
	ret = Field< double >::set( b, "diffConst", D2 );
	assert( ret );
	ret = Field< double >::set( c, "diffConst", D3 );
	assert( ret );

	Id compt( "/diff/kinetics" );
	// Set it to cubeSide mesh divisions in each dimension, dx in each.
	vector< double > coords( 9, dx );
	coords[0] = coords[1] = coords[2] = 0;
	coords[3] = cubeSide * dx;
	if ( n >= 2 )
		coords[4] = cubeSide * dx;
	if ( n == 3 )
		coords[5] = cubeSide * dx;

	ret = Field< bool >::set( compt, "preserveNumEntries", false );
	assert( ret );
	ret = Field< vector< double > >::set( compt, "coords", coords );
	assert( ret );
	Qinfo::waitProcCycles( 2 );

	Id mesh( "/diff/kinetics/mesh" );
	assert( mesh != Id() );
	assert( mesh.element()->dataHandler()->localEntries() == vol );
	// This should assign the same init conc to the new pool objects.
	assert( a.element()->dataHandler()->localEntries() == vol );

	Field< double >::setRepeat( a, "concInit", 0 );
	Field< double >::setRepeat( b, "concInit", 0 );
	Field< double >::setRepeat( c, "concInit", 0 );

	Field< double >::set( ObjId( a, 0 ), "concInit", 50 );

	Field< double >::set( ObjId( c, (cubeSide + 1 ) * cubeSide / 2 ), "concInit", 10 );

	// Field< double >::set( ObjId( b, cubeSide - 1 ), "concInit", 1 );

    shell->doSetClock( 0, dt );
    shell->doSetClock( 1, dt );
    shell->doSetClock( 2, dt );
    shell->doSetClock( 3, 0 ); 

    shell->doReinit();

	if ( doPrint )
		cout << setprecision( 3 ) << setiosflags( ios::fixed ) << endl;
	for ( unsigned int i = 0; i < 4; ++i ) {
		shell->doStart( 1 );
		vector< double > conc;
		Field< double >::getVec( b, "conc", conc );
		assert( conc.size() == vol );
		if ( doPrint ) {
			for ( unsigned int j = 0; j < cubeSide; ++j ) {
				for ( unsigned int k = 0; k < cubeSide; ++k )
					cout << conc[ j * cubeSide + k ] << " ";
				cout << endl;
			}
			cout << endl;
		}
	}
	if ( doPrint )
		cout << setprecision( 6 );

	shell->doDelete( mgr );

	cout << "." << flush;
}

void rtReacDiff()
{
		/* This is currently not handled by the new GslStoich class. Soon.*/
	rtReplicateModels();
	testDiff1D();
	testDiffNd( 1 );
	testDiffNd( 2 );
	testDiffNd( 3 );

	testReacDiffNd( 2 );
}
