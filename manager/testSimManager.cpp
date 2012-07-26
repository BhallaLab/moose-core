/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "../shell/Shell.h"
#include "SimManager.h"
#include "../kinetics/ReadCspace.h"

void testBuildFromBareKineticTree()
{
}

void testBuildFromKkitTree()
{
}

void testMakeStandardElements()
{
}

void testRemeshing()
{
	static const double DiffConst = 1e-12; // m^2/sec

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );
	vector< double > coords( 9, 0 );
	coords[3] = 1.000000001e-4;
	coords[4] = coords[5] = 1e-6;
	coords[6] = 1e-6;
	coords[7] = coords[8] = 1e-6;
	unsigned int numVox = coords[3] / coords[6];
	double tau = 4 * coords[6] * coords[6] / ( DiffConst * PI );
	double runtime = 100;
	double DT = 0.1;
	assert( tau > 10 * DT );
	assert( runtime > 10 * tau );

	Id mgr = shell->doCreate( "SimManager", Id(), "meshTest", dims );
	assert( mgr != Id() );
	SimManager* sm = reinterpret_cast< SimManager* >( mgr.eref().data() );
	sm->setPlotDt( DT );
	sm->makeStandardElements( mgr.eref(), 0, "CubeMesh" );
	Id kinetics( "/meshTest/kinetics" );
	assert( kinetics != Id() );
	Id pool = shell->doCreate( "Pool", kinetics, "pool", dims );
	assert( pool != Id() );
	Field< double >::set( pool, "diffConst", DiffConst );
	Id meshEntry( "/meshTest/kinetics/mesh" );
	assert( meshEntry != Id() );
	MsgId mid = 
		shell->doAddMsg( "OneToOne", pool, "mesh", meshEntry, "mesh" );
	assert( mid != Msg::bad );

	Qinfo q;
	q.setThreadNum( ScriptThreadNum );
	sm->build( mgr.eref(), &q, "gsl" );
	// SetGet1< string >::set( mgr, "build", "gsl" );

	shell->doReinit();

	shell->doStart( runtime );

	// Make a very long, thin, cuboid: 100um x 1um x 1um, in 1 um segments.

	Field< bool >::set( kinetics, "preserveNumEntries", 0 );
	Field< vector< double > >::set( kinetics, "coords", coords );
	// Qinfo::waitProcCycles( 2 );
	unsigned int n = Field< unsigned int >::get( kinetics, "nx" );
	assert( n == numVox );

	n = pool()->dataHandler()->localEntries();
	assert( n == numVox );
	Field< double >::setRepeat( pool, "concInit", 0 );
	Field< double >::set( ObjId( pool, 0 ), "concInit", 1 );

	Id stoich( "/meshTest/stoich" );
	assert( stoich != Id() );
	Id gsl( "/meshTest/stoich/gsl" );
	assert( gsl != Id() );
	n = gsl()->dataHandler()->localEntries();
	assert( n == numVox );
	shell->doReinit();
	shell->doStart( runtime );
	vector< double > conc;
	Field< double >::getVec( pool, "conc", conc );
	assert( conc.size() == numVox );
	double dx = coords[6];
	double err = 0;
	for ( unsigned int i = 0; i < numVox; ++i ) {
		double x = i * dx;
		double y = dx *  // This part represents the init conc of 1 in dx
		( 0.5 / sqrt( PI * DiffConst * runtime ) ) * exp( -x * x / ( 4 * DiffConst * runtime ) ); // This part is the solution as a func of x,t.
		err += ( y - conc[i] ) * ( y - conc[i] );
	}
	assert( doubleApprox( err, 0 ) );


	// Another long, thin, cuboid: 100um x 1um x 1um, in 0.5 um segments.
	runtime = 50;
	coords[6] = 5.000000001e-7;
	numVox = coords[3] / coords[6];
	Field< vector< double > >::set( kinetics, "coords", coords );
	Qinfo::waitProcCycles( 2 );
	n = Field< unsigned int >::get( kinetics, "nx" );
	assert( n == numVox );
	n = pool()->dataHandler()->localEntries();
	assert( n == numVox );
	Field< double >::setRepeat( pool, "concInit", 0 );
	Field< double >::set( ObjId( pool, 0 ), "concInit", 2 );
	n = gsl()->dataHandler()->localEntries();
	assert( n == numVox );
	shell->doReinit();
	shell->doStart( runtime );
	dx = coords[6];
	err = 0;
	Field< double >::getVec( pool, "conc", conc );
	assert( conc.size() == numVox );
	for ( unsigned int i = 0; i < numVox; ++i ) {
		double x = i * dx;
		double y = 2 * dx * // This part represents the init conc of 2 in dx
		( 0.5 / sqrt( PI * DiffConst * runtime ) ) * exp( -x * x / ( 4 * DiffConst * runtime ) ); // This part is the solution as a func of x,t.
		err += ( y - conc[i] ) * ( y - conc[i] );
	}
	assert( doubleApprox( err/5, 0 ) );

	shell->doDelete( mgr );
	cout << "." << flush;
}

void verifyZombieTurnoverTypes( bool isZombie ) {
	if ( isZombie ) {
		assert( Id( "/model/kinetics/a" ).element()->cinfo()->isA( 
			"ZombiePool" ) );
		assert( Id( "/model/kinetics/b" ).element()->cinfo()->isA( 
			"ZombiePool" ) );
		assert( Id( "/model/kinetics/k/Jjkl/Jjkl_cplx" ).element()->cinfo()->isA( "ZombiePool" ) );
		assert( Id( "/model/kinetics/AabX" ).element()->cinfo()->isA( 
			"ZombieReac" ) );
		assert( Id( "/model/kinetics/BbcX" ).element()->cinfo()->isA( 
			"ZombieReac" ) );
		assert( Id( "/model/kinetics/k/Jjkl" ).element()->cinfo()->isA( 
			"ZombieEnz" ) );
	} else {
		assert( Id( "/model/kinetics/a" ).element()->cinfo()->isA( 
			"Pool" ) );
		assert( Id( "/model/kinetics/b" ).element()->cinfo()->isA( 
			"Pool" ) );
		assert( Id( "/model/kinetics/k/Jjkl/Jjkl_cplx" ).element()->cinfo()->isA( "Pool" ) );
		assert( Id( "/model/kinetics/AabX" ).element()->cinfo()->isA( 
			"Reac" ) );
		assert( Id( "/model/kinetics/BbcX" ).element()->cinfo()->isA( 
			"Reac" ) );
		assert( Id( "/model/kinetics/k/Jjkl" ).element()->cinfo()->isA( 
			"Enz" ) );
	}
}

void testZombieTurnover()
{
	// Do some round trips with a chem model, converting between different
	// numerical methods. This requires zombifying and unzombifying the
	// model. 
	// This test checks four things: 
	// the classes of all objects in the sim
	// Adding new objects and incorporating them with a refreshPath
	// Deleting objects and reconciling their absence
	// Memory leaks when we round-trip.

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	ReadCspace rc;
	Id mgr = rc.readModelString( "|AabX|BbcX|CcdX|DdeX|Eefg|Ffgh|Gghi|Hhij|Iijk|Jjkl|Kklm|Llmn| 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 101 102 201 202 301 302 401 402 501 502 601 602 701 702 801 802 901 902 1001 1002 1101 1102 1201 1202",
		"model", Id(), "gsl" );
	//  I would really like to be able to define MMenz as well here.
	verifyZombieTurnoverTypes( 1 );

	/*
	Field< string >::set( mgr, "method", "ee" );
	verifyZombieTurnoverTypes( 0 );
	Field< string >::set( mgr, "method", "gsl" );
	verifyZombieTurnoverTypes( 1 );
	*/

	shell->doDelete( mgr );
}


void testSimManager()
{
	testBuildFromBareKineticTree();
	testBuildFromKkitTree();
	testMakeStandardElements();
	testRemeshing();
	testZombieTurnover(); 
}
