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
	const double NA_RATIO = 6e23 / NA;

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Shell::cleanSimulation();

	Id kineticId = shell->doLoadModel( "Kholodenko.g", "/rkktest", "Neutral" );
	assert( kineticId != Id() );

	double n;
	assert( Id( "/rkktest/kinetics/MAPK/MKKK" ) != Id() );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK" ), "n" );
	assert( doubleEq( n, 0.1 / NA_RATIO ) );

	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKK" ), "n" );
	assert( doubleEq( n, 0.3 / NA_RATIO ) );

	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MAPK" ), "n" );
	assert( doubleEq( n, 0.3 / NA_RATIO ) );

	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Ras-MKKKK" ), "n" );
	assert( doubleEq( n, 0.001 / NA_RATIO ) );

	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int1" ), "n" );
	assert( doubleEq( n, 0.001 / NA_RATIO ) );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int2" ), "n" );
	assert( doubleEq( n, 0.001 / NA_RATIO ) );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int3" ), "n" );
	assert( doubleEq( n, 0.001 / NA_RATIO ) );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int4" ), "n" );
	assert( doubleEq( n, 0.001 / NA_RATIO ) );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int5" ), "n" );
	assert( doubleEq( n, 0.001 / NA_RATIO ) );

	double conc;
	const double CONC_RATIO = 1.0e-3; // The original units were micromolar.
	// This is to convert to millimolar.

	conc = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK" ), "conc" );
	assert( doubleEq( conc, 0.1 * CONC_RATIO ) );

	conc = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKK" ), "conc" );
	assert( doubleEq( conc, 0.3 * CONC_RATIO ) );

	conc = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MAPK" ), "conc" );
	assert( doubleEq( conc, 0.3 * CONC_RATIO ) );

	conc = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Ras-MKKKK" ), "conc" );
	assert( doubleEq( conc, 0.001 * CONC_RATIO ) );

	conc = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int1" ), "conc" );
	assert( doubleEq( conc, 0.001 * CONC_RATIO ) );
	conc = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int2" ), "conc" );
	assert( doubleEq( conc, 0.001 * CONC_RATIO ) );
	conc = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int3" ), "conc" );
	assert( doubleEq( conc, 0.001 * CONC_RATIO ) );
	conc = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int4" ), "conc" );
	assert( doubleEq( conc, 0.001 * CONC_RATIO ) );
	conc = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int5" ), "conc" );
	assert( doubleEq( conc, 0.001 * CONC_RATIO ) );

	double rate;

	////////////////////////////////////////////////////////////////////
	// Reac
	////////////////////////////////////////////////////////////////////
	// NumRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Neg_feedback" ), "kf" );
	assert( doubleEq( rate, 1.0 * NA_RATIO ) );
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Neg_feedback" ), "kb" );
	assert( doubleEq( rate, 0.009 ) );

	// conc rates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Neg_feedback" ), "Kf" );
	assert( doubleEq( rate, 1000.0 ) ); // In 1/mM/sec, which is 1000x 1/uM/sec
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Neg_feedback" ), "Kb" );
	assert( doubleEq( rate, 0.009 ) );

	////////////////////////////////////////////////////////////////////
	// MMEnz
	////////////////////////////////////////////////////////////////////
	// NumRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Ras-MKKKK/1" ), "numKm" );
	assert( doubleEq( rate, 0.01 / NA_RATIO ) );

	// ConcRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Ras-MKKKK/1" ), "Km" );
	assert( doubleEq( rate, 1e-5 ) );

	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Ras-MKKKK/1" ), "kcat");
	assert( doubleEq( rate, 2.5 ) );

	// NumRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int1/2" ), "numKm" );
	assert( doubleEq( rate, 0.008 / NA_RATIO ) );

	// ConcRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int1/2" ), "Km" );
	assert( doubleEq( rate, 0.008 * 1e-3 ) ); // to get millimolar
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int1/2" ), "kcat" );
	assert( doubleEq( rate, 0.25 ) );

	// NumRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK-P/3" ), "numKm" );
	assert( doubleEq( rate, ( ( 0.1 + 0.025 ) / 8.3333 ) / NA_RATIO ) );

	// ConcRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK-P/3" ), "Km" );
	assert( doubleEq( rate, ( ( 0.1 + 0.025 ) / 8.3333 ) * 1e-3 ) );
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK-P/3" ), "kcat" );
	assert( doubleEq( rate, 0.025 ) );
	
	/////////////////////////////////////////////////////////////////////
	shell->doDelete( kineticId );
	cout << "." << flush;
}

void rtRunKkit()
{
	const double TOLERANCE = 2e-3;
	const double NA_RATIO = 6e23 / NA;

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Shell::cleanSimulation();

	Id modelId = shell->doLoadModel( "Kholodenko.g", "/rkktest", "rk5" );
	assert( modelId != Id() );
	Id stoichId( "/rkktest/stoich" );
	assert( stoichId != Id() );
	Id comptId( "/rkktest/kinetics" );
	assert( comptId != Id() );
	unsigned int numVarMols = Field< unsigned int >::get( 
		stoichId, "nVarPools" );
	assert ( numVarMols == 15 );

	double n;
	assert( Id( "/rkktest/kinetics/MAPK/MKKK" ) != Id() );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK" ), "n" );
	assert( doubleEq( n, 0.1 / NA_RATIO ) );

	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKK" ), "n" );
	assert( doubleEq( n, 0.3 / NA_RATIO ) );

	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MAPK" ), "n" );
	assert( doubleEq( n, 0.3 / NA_RATIO ) );

	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Ras-MKKKK" ), "n" );
	assert( doubleEq( n, 0.001 / NA_RATIO ) );

	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int1" ), "n" );
	assert( doubleEq( n, 0.001 / NA_RATIO ) );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int2" ), "n" );
	assert( doubleEq( n, 0.001 / NA_RATIO ) );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int3" ), "n" );
	assert( doubleEq( n, 0.001 / NA_RATIO ) );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int4" ), "n" );
	assert( doubleEq( n, 0.001 / NA_RATIO ) );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int5" ), "n" );
	assert( doubleEq( n, 0.001 / NA_RATIO ) );

	double conc;
	// Original concs were in uM, but MOOSE uses mM.
	const double CONC_RATIO = 1.0e-3; 

	conc = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK" ), "conc" );
	assert( doubleEq( conc, 0.1 * CONC_RATIO ) );

	conc = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKK" ), "conc" );
	assert( doubleEq( conc, 0.3 * CONC_RATIO ) );

	double rate;
	// NumRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Neg_feedback" ), "kf" );
	assert( doubleEq( rate, 1.0 * NA_RATIO ) );
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Neg_feedback" ), "kb" );
	assert( doubleEq( rate, 0.009 ) );

	// conc rates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Neg_feedback" ), "Kf" );
	assert( doubleEq( rate, 1000.0 ) ); // In 1/mM/sec, which is 1000x 1/uM/sec
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Neg_feedback" ), "Kb" );
	assert( doubleEq( rate, 0.009 ) );

	///////////////////////////////////////////////////////////////////////
	// Now on to the enzymes.
	///////////////////////////////////////////////////////////////////////

	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Ras-MKKKK/1" ), "Km" );
	assert( doubleEq( rate, 0.01 * 1e-3 ) ); // Convert from uM to mM
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Ras-MKKKK/1" ), "kcat");
	assert( doubleEq( rate, 2.5 ) );

	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int1/2" ), "Km" );
	assert( doubleEq( rate, 0.008 * 1e-3 ) );
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int1/2" ), "kcat" );
	assert( doubleEq( rate, 0.25 ) );

	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK-P/3" ), "Km" );
	assert( doubleEq( rate, ( ( 0.1 + 0.025 ) / 8.3333 ) * 1e-3 ) );
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK-P/3" ), "kcat" );
	assert( doubleEq( rate, 0.025 ) );
	

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
	shell->doReinit();
	shell->doStart( 5001.0 );

	Id plotId( "/rkktest/graphs/conc1/MAPK-PP.Co" );
	assert( plotId != Id() );
	unsigned int size = Field< unsigned int >::get( plotId, "size" );
	// cout << "size = " << size << endl;
	assert( size == 501 ); // Note that dt was 10.

	// Scale the output from mM to uM
	bool ok = SetGet2< double, double >::set(
		plotId, "linearTransform", 1000, 0 );
	assert( ok );

	ok = SetGet2< string, string >::set(
		plotId, "xplot", "check.plot", "MAPK-PP.plot" );
	assert( ok );
	
	ok = SetGet3< string, string, string >::set(
		plotId, "compareXplot", "Kholodenko.plot", 
		"/graphs/conc1/MAPK-PP.Co", "rmsr" );
	assert( ok );

	// Returns -1 on failure, otherwise the (positive) rms ratio.
	double val = Field< double >::get( plotId, "outputValue" );
	assert( val >= 0 && val < TOLERANCE );

	/////////////////////////////////////////////////////////////////////
	// Change volume and run it again.
	/////////////////////////////////////////////////////////////////////
	double vol = LookupField< short, double >::get( stoichId, 
		"compartmentVolume", 0);
	vol *= 0.1;
	ok = LookupField< short, double >::set( stoichId, "compartmentVolume",
		0, vol );

	assert( ok );
	double side = pow( vol, 1.0/3.0 );
	ok = Field< double >::set( Id( "/rkktest/kinetics" ), "x1", side );
	assert( ok );
	ok = Field< double >::set( Id( "/rkktest/kinetics" ), "y1", side );
	assert( ok );
	ok = Field< double >::set( Id( "/rkktest/kinetics" ), "z1", side );
	assert( ok );
	double actualVol = 
		Field< double >::get( Id( "/rkktest/kinetics/mesh" ), "size" );
	assert( doubleEq( actualVol, vol ) );

	shell->doReinit();
	shell->doStart( 5001.0 );
	size = Field< unsigned int >::get( plotId, "size" );
	assert( size == 501 ); // Note that dt was 10.

	// Scale output to uM from mM.
	SetGet2< double, double >::set( plotId, "linearTransform", 1000, 0 );
	ok = SetGet2< string, string >::set(
		plotId, "xplot", "check.plot", "volscale_MAPK-PP.plot" );
	assert( ok );

	ok = SetGet3< string, string, string >::set(
		plotId, "compareXplot", "Kholodenko.plot", 
		"/graphs/conc1/MAPK-PP.Co", "rmsr" );
	val = Field< double >::get( plotId, "outputValue" );
	assert( val >= 0 && val < TOLERANCE );

	assert( ok );
	
	/////////////////////////////////////////////////////////////////////
	shell->doDelete( modelId );
	cout << "." << flush;
}

void checkCspaceParms()
{
	const double VOL = 1e-18; // m^3
	const double CONCSCALE = 1e-3; // Convert from uM to mM.
	const double VOLSCALE = VOL * NA; // Convert from conc in mM to #.

	Id temp( "/osc/a" );
	assert( temp != Id() );
	double conc = Field< double >::get( Id( "/osc/a" ), "concInit" );
	assert( doubleEq( conc, 3.5 * CONCSCALE ) );
	conc = Field< double >::get( Id( "/osc/b" ), "concInit" );
	assert( doubleEq( conc, 0 * CONCSCALE ) );
	conc = Field< double >::get( Id( "/osc/c" ), "concInit" );
	assert( doubleEq( conc, 0.1 * CONCSCALE ) );
	conc = Field< double >::get( Id( "/osc/d" ), "concInit" );
	assert( doubleEq( conc, 0 * CONCSCALE ) );
	conc = Field< double >::get( Id( "/osc/e" ), "concInit" );
	assert( doubleEq( conc, 0 * CONCSCALE ) );
	conc = Field< double >::get( Id( "/osc/f" ), "concInit" );
	assert( doubleEq( conc, 0.1 * CONCSCALE ) );

	double n = Field< double >::get( Id( "/osc/a" ), "nInit" );
	assert( doubleEq( n, 3.5 * VOLSCALE * CONCSCALE ) );
	n = Field< double >::get( Id( "/osc/b" ), "nInit" );
	assert( doubleEq( n, 0 * VOLSCALE * CONCSCALE ) );
	n = Field< double >::get( Id( "/osc/c" ), "nInit" );
	assert( doubleEq( n, 0.1 * VOLSCALE * CONCSCALE ) );
	n = Field< double >::get( Id( "/osc/d" ), "nInit" );
	assert( doubleEq( n, 0 * VOLSCALE * CONCSCALE ) );
	n = Field< double >::get( Id( "/osc/e" ), "nInit" );
	assert( doubleEq( n, 0 * VOLSCALE * CONCSCALE ) );
	n = Field< double >::get( Id( "/osc/f" ), "nInit" );
	assert( doubleEq( n, 0.1 * VOLSCALE * CONCSCALE ) );

	double rate = Field< double >::get( Id( "/osc/AabX" ), "Kf" );
	assert( doubleEq( rate, 0.01 ) );
	rate = Field< double >::get( Id( "/osc/AabX" ), "Kb" );
	assert( doubleEq( rate, 0.0 ) );

	rate = Field< double >::get( Id( "/osc/b/DabX" ), "k3" );
	assert( doubleEq( rate, 0.5 ) );
	rate = Field< double >::get( Id( "/osc/b/DabX" ), "k2" );
	assert( doubleEq( rate, 2 ) );
	rate = Field< double >::get( Id( "/osc/b/DabX" ), "k1" );
	assert( doubleEq( rate, 2.5 / ( VOLSCALE * CONCSCALE ) ) );
	rate = Field< double >::get( Id( "/osc/b/DabX" ), "Km" );
	assert( doubleEq( rate, 1 * CONCSCALE ) );

	rate = Field< double >::get( Id( "/osc/c/Jbca" ), "k3" );
	assert( doubleEq( rate, 1 ) );
	rate = Field< double >::get( Id( "/osc/c/Jbca" ), "k2" );
	assert( doubleEq( rate, 4 ) );
	rate = Field< double >::get( Id( "/osc/c/Jbca" ), "k1" );
	assert( doubleEq( rate, 100 / ( VOLSCALE * CONCSCALE ) ) );
	rate = Field< double >::get( Id( "/osc/c/Jbca" ), "Km" );
	assert( doubleEq( rate, 0.05 * CONCSCALE ) );

	rate = Field< double >::get( Id( "/osc/AdeX" ), "Kf" );
	assert( doubleEq( rate, 0.01 ) );
	rate = Field< double >::get( Id( "/osc/AdeX" ), "Kb" );
	assert( doubleEq( rate, 0.0 ) );

	rate = Field< double >::get( Id( "/osc/e/DdeX" ), "k3" );
	assert( doubleEq( rate, 0.5 ) );
	rate = Field< double >::get( Id( "/osc/e/DdeX" ), "Km" );
	assert( doubleEq( rate, 1 * CONCSCALE ) );

	rate = Field< double >::get( Id( "/osc/f/Jefd" ), "k3" );
	assert( doubleEq( rate, 1 ) );
	rate = Field< double >::get( Id( "/osc/f/Jefd" ), "Km" );
	assert( doubleEq( rate, 0.05 * CONCSCALE ) );

	rate = Field< double >::get( Id( "/osc/AadX" ), "Kf" );
	assert( doubleEq( rate, 0.01 ) );
	rate = Field< double >::get( Id( "/osc/AadX" ), "Kb" );
	assert( doubleEq( rate, 0.0 ) );

	rate = Field< double >::get( Id( "/osc/AbeX" ), "Kf" );
	assert( doubleEq( rate, 0.0 ) );
	rate = Field< double >::get( Id( "/osc/AbeX" ), "Kb" );
	assert( doubleEq( rate, 0.005 ) );
}

void rtReadCspace()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );

	Id kineticId = shell->doLoadModel( "Osc.cspace", "/osc", "Neutral" );
	assert( kineticId != Id() );

	checkCspaceParms();

	shell->doDelete( kineticId );
	cout << "." << flush;
}

void rtRunCspace()
{
	const double TOLERANCE = 1e-2;
	const double CONCSCALE = 1e-3; // Convert from uM to mM.

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

	checkCspaceParms();

	// cout << "After Reinit\n"; Qinfo::reportQ();
	shell->doStart( 2501.0 );

	Id plotId( "/osc/plotd" );
	assert( plotId != Id() );
	unsigned int size = Field< unsigned int >::get( plotId, "size" );
	// cout << "size = " << size << endl;
	assert( size == 251 ); // Note that dt was 10.

	// Scale the output from mM to uM
	bool ok = SetGet2< double, double >::set(
		plotId, "linearTransform", 1.0/CONCSCALE, 0 );
	assert( ok );
	
	ok = SetGet3< string, string, string >::set(
		plotId, "compareXplot", "Osc_cspace_ref_model.plot", 
		"plotd", "rmsr" );
	assert( ok );

	ok = SetGet2< string, string >::set(
		plotId, "xplot", "check.plot", "cspace_osc.plot" );
	assert( ok );

	/*
	Id plota( "/osc/plota" );
	Id plotb( "/osc/plotb" );
	Id plotc( "/osc/plotc" );
	Id plote( "/osc/plote" );
	Id plotf( "/osc/plotf" );
	SetGet2< string, string >::set( plota, "xplot", "check.plot", "a.plot");
	SetGet2< string, string >::set( plotb, "xplot", "check.plot", "b.plot");
	SetGet2< string, string >::set( plotc, "xplot", "check.plot", "c.plot");
	SetGet2< string, string >::set( plote, "xplot", "check.plot", "e.plot");
	SetGet2< string, string >::set( plotf, "xplot", "check.plot", "f.plot");
	*/

	// Returns -1 on failure, otherwise the (positive) rms ratio.
	double val = Field< double >::get( plotId, "outputValue" );
	assert( val >= 0 && val < TOLERANCE );

	/////////////////////////////////////////////////////////////////////
	shell->doDelete( kineticId );
	cout << "." << flush;
}
