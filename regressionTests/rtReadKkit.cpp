/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <fstream>
#include "header.h"
#include "Shell.h"
#include "LoadModels.h"

extern ModelType findModelType( string filename, ifstream& fin, 
	string& line );

void rtFindModelType()
{
	string line;

	ifstream dotpfin( "ca1.p" );
	assert( dotpfin );
	assert( findModelType( "ca1.p", dotpfin, line ) == DOTP );

	ifstream kkitfin( "Kholodenko.g" );
	assert( kkitfin );
	assert( findModelType( "Kholodenko.g", kkitfin, line ) == KKIT );

	ifstream sbmlfin( "Kholodenko.xml" );
	assert( sbmlfin );
	assert( findModelType( "Kholodenko.xml", sbmlfin, line ) == UNKNOWN );

	ifstream cspacefin( "M101.cspace" );
	assert( cspacefin );
	assert( findModelType( "M101.cspace", cspacefin, line ) == CSPACE );
	cout << "." << flush;
}

void rtReadKkit()
{
	const double TOLERANCE = 2e-3;

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Shell::cleanSimulation();

	Id kineticId = shell->doLoadModel( "Kholodenko.g", "/rkktest", "gsl" );
	assert( kineticId != Id() );
	unsigned int numVarMols = Field< unsigned int >::get( 
		kineticId, "nVarPools" );
	assert ( numVarMols == 15 );

	/*
	Id gsl = shell->doCreate( "GslIntegrator", kineticId, "gsl", dims );
	bool ret = SetGet1< Id >::set( gsl, "stoich", kineticId );
	assert( ret );
	ret = Field< bool >::get( gsl, "isInitialized" );
	assert( ret );

	shell->doSetClock( 0, 10 );
	shell->doSetClock( 1, 10 );
	shell->doSetClock( 2, 10 );
	shell->doSetClock( 3, 0 );
	shell->doSetClock( 4, 0 );
	shell->doSetClock( 5, 0 );
	shell->doUseClock( "/rkktest/gsl", "process", 0 );
	shell->doUseClock( "/rkktest/graphs/##[TYPE=Table],/rkktest/moregraphs/##[TYPE=Table]", "process", 2 );

	*/
	shell->doSetClock( 0, 10 );
	shell->doSetClock( 1, 10 );
	shell->doSetClock( 2, 10 );
	// cout << "Before Reinit\n"; Qinfo::reportQ();
	shell->doReinit();
	// cout << "After Reinit\n"; Qinfo::reportQ();
	shell->doStart( 5001.0 );

	Id plotId( "/rkktest/graphs/conc1/MAPK-PP.Co" );
	assert( plotId != Id() );
	unsigned int size = Field< unsigned int >::get( plotId, "size" );
	// cout << "size = " << size << endl;
	assert( size == 501 ); // Note that dt was 10.
	
	/*
	bool ok = SetGet::strSet( 
		plotId.eref(), "compareXplot", "Kholodenko.plot,/graphs/conc1/MAPK-PP.Co,rmsr" );
		*/
	bool ok = SetGet3< string, string, string >::set(
		plotId, "compareXplot", "Kholodenko.plot", 
		"/graphs/conc1/MAPK-PP.Co", "rmsr" );
	assert( ok );

	ok = SetGet2< string, string >::set(
		plotId, "xplot", "check.plot", "MAPK-PP.plot" );
	assert( ok );

	// Returns -1 on failure, otherwise the (positive) rms ratio.
	double val = Field< double >::get( plotId, "outputValue" );
	assert( val >= 0 && val < TOLERANCE );

	/////////////////////////////////////////////////////////////////////
	// shell->doDelete( kineticId );
	cout << "." << flush;
}

void rtReadCspace()
{
	const double TOLERANCE = 2e-3;

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	// Shell::cleanSimulation();

	Id kineticId = shell->doLoadModel( "Osc.cspace", "/osc", "gsl" );
	assert( kineticId != Id() );
	unsigned int numVarMols = Field< unsigned int >::get( 
		kineticId, "nVarPools" );
	assert ( numVarMols == 10 ); // 6 mols + 4 enz

	shell->doSetClock( 0, 10 );
	shell->doSetClock( 1, 10 );
	shell->doSetClock( 2, 10 );
	shell->doSetClock( 3, 10 );
	// cout << "Before Reinit\n"; Qinfo::reportQ();
	shell->doReinit();

	Id temp( "/osc/a" );
	assert( temp != Id() );
	double conc = Field< double >::get( Id( "/osc/a" ), "concInit" );
	assert( doubleEq( conc, 3.5 ) );
	conc = Field< double >::get( Id( "/osc/b" ), "conc" );
	assert( doubleEq( conc, 0 ) );
	conc = Field< double >::get( Id( "/osc/c" ), "conc" );
	assert( doubleEq( conc, 0.1 ) );
	conc = Field< double >::get( Id( "/osc/d" ), "conc" );
	assert( doubleEq( conc, 0 ) );
	conc = Field< double >::get( Id( "/osc/e" ), "conc" );
	assert( doubleEq( conc, 0 ) );
	conc = Field< double >::get( Id( "/osc/f" ), "conc" );
	assert( doubleEq( conc, 0.1 ) );

	double rate = Field< double >::get( Id( "/osc/AabX" ), "Kf" );
	assert( doubleEq( rate, 0.01 ) );
	rate = Field< double >::get( Id( "/osc/AabX" ), "Kb" );
	assert( doubleEq( rate, 0.0 ) );

	rate = Field< double >::get( Id( "/osc/b/DabX" ), "k3" );
	assert( doubleEq( rate, 0.5 ) );
	rate = Field< double >::get( Id( "/osc/b/DabX" ), "Km" );
	assert( doubleEq( rate, 1 ) );

	rate = Field< double >::get( Id( "/osc/c/Jbca" ), "k3" );
	assert( doubleEq( rate, 1 ) );
	rate = Field< double >::get( Id( "/osc/c/Jbca" ), "Km" );
	assert( doubleEq( rate, 0.05 ) );

	rate = Field< double >::get( Id( "/osc/AdeX" ), "Kf" );
	assert( doubleEq( rate, 0.01 ) );
	rate = Field< double >::get( Id( "/osc/AdeX" ), "Kb" );
	assert( doubleEq( rate, 0.0 ) );

	rate = Field< double >::get( Id( "/osc/e/DdeX" ), "k3" );
	assert( doubleEq( rate, 0.5 ) );
	rate = Field< double >::get( Id( "/osc/e/DdeX" ), "Km" );
	assert( doubleEq( rate, 1 ) );

	rate = Field< double >::get( Id( "/osc/f/Jefd" ), "k3" );
	assert( doubleEq( rate, 1 ) );
	rate = Field< double >::get( Id( "/osc/f/Jefd" ), "Km" );
	assert( doubleEq( rate, 0.05 ) );

	rate = Field< double >::get( Id( "/osc/AadX" ), "Kf" );
	assert( doubleEq( rate, 0.002 ) );
	rate = Field< double >::get( Id( "/osc/AadX" ), "Kb" );
	assert( doubleEq( rate, 0.0 ) );

	rate = Field< double >::get( Id( "/osc/AbeX" ), "Kf" );
	assert( doubleEq( rate, 0.0 ) );
	rate = Field< double >::get( Id( "/osc/AbeX" ), "Kb" );
	assert( doubleEq( rate, 0.001 ) );

	// cout << "After Reinit\n"; Qinfo::reportQ();
	shell->doStart( 15001.0 );

	Id plotId( "/osc/plotd" );
	assert( plotId != Id() );
	unsigned int size = Field< unsigned int >::get( plotId, "size" );
	// cout << "size = " << size << endl;
	assert( size == 1501 ); // Note that dt was 10.
	
	/*
	bool ok = SetGet::strSet( 
		plotId.eref(), "compareXplot", "Kholodenko.plot,/graphs/conc1/MAPK-PP.Co,rmsr" );
		*/
	bool ok = SetGet3< string, string, string >::set(
		plotId, "compareXplot", "Osc_cspace_ref_model.plot", 
		"plotd", "rmsr" );
	assert( ok );

	ok = SetGet2< string, string >::set(
		plotId, "xplot", "check.plot", "cspace_osc.plot" );
	assert( ok );

	Id plota( "/osc/plota" );
	Id plotb( "/osc/plota" );
	Id plotc( "/osc/plota" );
	Id plote( "/osc/plota" );
	Id plotf( "/osc/plota" );
	SetGet2< string, string >::set( plota, "xplot", "check.plot", "a.plot");
	SetGet2< string, string >::set( plotb, "xplot", "check.plot", "b.plot");
	SetGet2< string, string >::set( plotc, "xplot", "check.plot", "c.plot");
	SetGet2< string, string >::set( plote, "xplot", "check.plot", "e.plot");
	SetGet2< string, string >::set( plotf, "xplot", "check.plot", "f.plot");

	// Returns -1 on failure, otherwise the (positive) rms ratio.
	double val = Field< double >::get( plotId, "outputValue" );
	assert( val >= 0 && val < TOLERANCE );

	/////////////////////////////////////////////////////////////////////
	// shell->doDelete( kineticId );
	cout << "." << flush;
}
