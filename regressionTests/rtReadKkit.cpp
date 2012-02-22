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

void checkVolN( double v )
{
	const double NA_RATIO = 6e23 / NA;
	const double VOL = 1.666666666666e-21;
	double volscale = v / VOL;
	double n;
	assert( Id( "/rkktest/kinetics/MAPK/MKKK" ) != Id() );
	double vol = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK" ), "size" );
	assert( doubleEq( vol, v ) );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK" ), "n" );
	assert( doubleEq( n, volscale * 0.1 / NA_RATIO ) );

	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKK" ), "n" );
	assert( doubleEq( n, volscale * 0.3 / NA_RATIO ) );

	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MAPK" ), "n" );
	assert( doubleEq( n, volscale * 0.3 / NA_RATIO ) );

	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Ras-MKKKK" ), "n" );
	assert( doubleEq( n, volscale * 0.001 / NA_RATIO ) );

	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int1" ), "n" );
	assert( doubleEq( n, volscale * 0.001 / NA_RATIO ) );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int2" ), "n" );
	assert( doubleEq( n, volscale * 0.001 / NA_RATIO ) );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int3" ), "n" );
	assert( doubleEq( n, volscale * 0.001 / NA_RATIO ) );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int4" ), "n" );
	assert( doubleEq( n, volscale * 0.001 / NA_RATIO ) );
	n = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int5" ), "n" );
	assert( doubleEq( n, volscale * 0.001 / NA_RATIO ) );
}

void checkConc()
{
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
}

void checkReacRates( double v )
{
	const double NA_RATIO = 6e23 / NA;
	const double VOL = 1.666666666666e-21;
	double rate;

	////////////////////////////////////////////////////////////////////
	// Reac
	////////////////////////////////////////////////////////////////////
	// NumRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Neg_feedback" ), "kf" );
	assert( doubleEq( rate, 1.0 * NA_RATIO * VOL/v ) );
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Neg_feedback" ), "kb" );
	assert( doubleEq( rate, 0.009 ) );

	// conc rates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Neg_feedback" ), "Kf" );
	assert( doubleEq( rate, 1000.0 ) ); // In 1/mM/sec, which is 1000x 1/uM/sec
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Neg_feedback" ), "Kb" );
	assert( doubleEq( rate, 0.009 ) );
}

void checkEnzRates( double v )
{
	const double NA_RATIO = 6e23 / NA;
	const double VOL = 1.666666666666e-21;
	double rate;
	double volscale = v/VOL;
	////////////////////////////////////////////////////////////////////
	// MMEnz
	////////////////////////////////////////////////////////////////////
	// NumRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Ras-MKKKK/1" ), "numKm" );
	assert( doubleEq( rate, volscale * 0.01 / NA_RATIO ) );

	// ConcRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Ras-MKKKK/1" ), "Km" );
	assert( doubleEq( rate, 1e-5 ) );

	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/Ras-MKKKK/1" ), "kcat");
	assert( doubleEq( rate, 2.5 ) );

	// NumRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int1/2" ), "numKm" );
	assert( doubleEq( rate, volscale * 0.008 / NA_RATIO ) );

	// ConcRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int1/2" ), "Km" );
	assert( doubleEq( rate, 0.008 * 1e-3 ) ); // to get millimolar
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/int1/2" ), "kcat" );
	assert( doubleEq( rate, 0.25 ) );

	// NumRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK-P/3" ), "numKm" );
	assert( doubleEq( rate, volscale * ( ( 0.1 + 0.025 ) / 8.3333 ) / NA_RATIO ) );

	// ConcRates
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK-P/3" ), "Km" );
	assert( doubleEq( rate, ( ( 0.1 + 0.025 ) / 8.3333 ) * 1e-3 ) );
	rate = Field< double >::get( Id( "/rkktest/kinetics/MAPK/MKKK-P/3" ), "kcat" );
	assert( doubleEq( rate, 0.025 ) );
}

void rtReadKkit()
{
	const double VOL = 1.666666666666e-21;

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Shell::cleanSimulation();

	Id kineticId = shell->doLoadModel( "Kholodenko.g", "/rkktest", "Neutral" );
	assert( kineticId != Id() );

	checkVolN( VOL );
	checkConc();
	checkReacRates( VOL );
	checkEnzRates( VOL );
	
	/////////////////////////////////////////////////////////////////////
	// Now change the volume and do it again.
	/////////////////////////////////////////////////////////////////////
	Id parentCompartment( "/rkktest/kinetics" );
	double vol = Field< double >::get( parentCompartment, "size" );
	vol *= 0.1;
	bool ok = SetGet2< double, unsigned int >::set( parentCompartment,
		"buildDefaultMesh", vol, 1 );
	assert( ok );
	Qinfo::waitProcCycles( 2 );


	checkVolN( vol );
	checkConc();
	checkReacRates( vol );
	checkEnzRates( vol );

	/////////////////////////////////////////////////////////////////////
	shell->doDelete( kineticId );
	cout << "." << flush;
}

/// Reads specified model and does a superficial check that the
/// specified fields on the specified paths have the right values.
void rtReadKkitModels( const string& modelname, const char** path, 
	const char** field, const double* value, unsigned int num )
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Shell::cleanSimulation();

	Id mgr = shell->doLoadModel( modelname, "/kkit", "Neutral" );
	assert( mgr != Id() );
	for ( unsigned int i = 0; i < num; ++i ) {
		Id obj( path[i] );
		assert( obj != Id() );
		double y = Field< double >::get( obj, field[i] );
		assert( doubleEq( y, value[i] ) );

		string id2path = Field< string >::get( obj, "path" );
		assert( id2path == path[i] );
	}
	shell->doDelete( mgr );
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
	vector< Id > ret = LookupField< string, vector< Id > >::get( 
		plotId, "neighbours", "requestData" );
	assert( ret.size() == 1 );
	assert( ret[0] == Id( "/rkktest/kinetics/MAPK/MAPK-PP" ) );

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
	/*
	double vol = LookupField< short, double >::get( stoichId, 
		"compartmentVolume", 0);
	vol *= 0.1;
	ok = LookupField< short, double >::set( stoichId, "compartmentVolume",
		0, vol );

	assert( ok );
	*/
	Id parentCompartment( "/rkktest/kinetics" );
	double vol = Field< double >::get( parentCompartment, "size" );
	ok = SetGet2< double, unsigned int >::set( parentCompartment,
		"buildDefaultMesh", vol * 0.1, 1 );
	/*
	double side = pow( vol, 1.0/3.0 );
	ok = Field< double >::set( Id( "/rkktest/kinetics" ), "x1", side );
	assert( ok );
	ok = Field< double >::set( Id( "/rkktest/kinetics" ), "y1", side );
	assert( ok );
	ok = Field< double >::set( Id( "/rkktest/kinetics" ), "z1", side );
	assert( ok );
	*/
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

	Id temp( "/osc/kinetics/a" );
	assert( temp != Id() );
	double conc = Field< double >::get( Id( "/osc/kinetics/a" ), "concInit" );
	assert( doubleEq( conc, 3.5 * CONCSCALE ) );
	conc = Field< double >::get( Id( "/osc/kinetics/b" ), "concInit" );
	assert( doubleEq( conc, 0 * CONCSCALE ) );
	conc = Field< double >::get( Id( "/osc/kinetics/c" ), "concInit" );
	assert( doubleEq( conc, 0.1 * CONCSCALE ) );
	conc = Field< double >::get( Id( "/osc/kinetics/d" ), "concInit" );
	assert( doubleEq( conc, 0 * CONCSCALE ) );
	conc = Field< double >::get( Id( "/osc/kinetics/e" ), "concInit" );
	assert( doubleEq( conc, 0 * CONCSCALE ) );
	conc = Field< double >::get( Id( "/osc/kinetics/f" ), "concInit" );
	assert( doubleEq( conc, 0.1 * CONCSCALE ) );

	double n = Field< double >::get( Id( "/osc/kinetics/a" ), "nInit" );
	assert( doubleEq( n, 3.5 * VOLSCALE * CONCSCALE ) );
	n = Field< double >::get( Id( "/osc/kinetics/b" ), "nInit" );
	assert( doubleEq( n, 0 * VOLSCALE * CONCSCALE ) );
	n = Field< double >::get( Id( "/osc/kinetics/c" ), "nInit" );
	assert( doubleEq( n, 0.1 * VOLSCALE * CONCSCALE ) );
	n = Field< double >::get( Id( "/osc/kinetics/d" ), "nInit" );
	assert( doubleEq( n, 0 * VOLSCALE * CONCSCALE ) );
	n = Field< double >::get( Id( "/osc/kinetics/e" ), "nInit" );
	assert( doubleEq( n, 0 * VOLSCALE * CONCSCALE ) );
	n = Field< double >::get( Id( "/osc/kinetics/f" ), "nInit" );
	assert( doubleEq( n, 0.1 * VOLSCALE * CONCSCALE ) );

	double rate = Field< double >::get( Id( "/osc/kinetics/AabX" ), "Kf" );
	assert( doubleEq( rate, 0.01 ) );
	rate = Field< double >::get( Id( "/osc/kinetics/AabX" ), "Kb" );
	assert( doubleEq( rate, 0.0 ) );

	rate = Field< double >::get( Id( "/osc/kinetics/b/DabX" ), "k3" );
	assert( doubleEq( rate, 0.5 ) );
	rate = Field< double >::get( Id( "/osc/kinetics/b/DabX" ), "k2" );
	assert( doubleEq( rate, 2 ) );
	rate = Field< double >::get( Id( "/osc/kinetics/b/DabX" ), "k1" );
	assert( doubleEq( rate, 2.5 / ( VOLSCALE * CONCSCALE ) ) );
	rate = Field< double >::get( Id( "/osc/kinetics/b/DabX" ), "Km" );
	assert( doubleEq( rate, 1 * CONCSCALE ) );

	rate = Field< double >::get( Id( "/osc/kinetics/c/Jbca" ), "k3" );
	assert( doubleEq( rate, 1 ) );
	rate = Field< double >::get( Id( "/osc/kinetics/c/Jbca" ), "k2" );
	assert( doubleEq( rate, 4 ) );
	rate = Field< double >::get( Id( "/osc/kinetics/c/Jbca" ), "k1" );
	assert( doubleEq( rate, 100 / ( VOLSCALE * CONCSCALE ) ) );
	rate = Field< double >::get( Id( "/osc/kinetics/c/Jbca" ), "Km" );
	assert( doubleEq( rate, 0.05 * CONCSCALE ) );

	rate = Field< double >::get( Id( "/osc/kinetics/AdeX" ), "Kf" );
	assert( doubleEq( rate, 0.01 ) );
	rate = Field< double >::get( Id( "/osc/kinetics/AdeX" ), "Kb" );
	assert( doubleEq( rate, 0.0 ) );

	rate = Field< double >::get( Id( "/osc/kinetics/e/DdeX" ), "k3" );
	assert( doubleEq( rate, 0.5 ) );
	rate = Field< double >::get( Id( "/osc/kinetics/e/DdeX" ), "Km" );
	assert( doubleEq( rate, 1 * CONCSCALE ) );

	rate = Field< double >::get( Id( "/osc/kinetics/f/Jefd" ), "k3" );
	assert( doubleEq( rate, 1 ) );
	rate = Field< double >::get( Id( "/osc/kinetics/f/Jefd" ), "Km" );
	assert( doubleEq( rate, 0.05 * CONCSCALE ) );

	rate = Field< double >::get( Id( "/osc/kinetics/AadX" ), "Kf" );
	assert( doubleEq( rate, 0.01 ) );
	rate = Field< double >::get( Id( "/osc/kinetics/AadX" ), "Kb" );
	assert( doubleEq( rate, 0.0 ) );

	rate = Field< double >::get( Id( "/osc/kinetics/AbeX" ), "Kf" );
	assert( doubleEq( rate, 0.0 ) );
	rate = Field< double >::get( Id( "/osc/kinetics/AbeX" ), "Kb" );
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

	Id base = shell->doLoadModel( "Osc.cspace", "/osc", "gsl" );
	assert( base != Id() );
	Id stoich( "/osc/stoich" );
	unsigned int numVarMols = Field< unsigned int >::get( 
		stoich, "nVarPools" );
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

	Id plotId( "/osc/graphs/plotd" );
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
	shell->doDelete( base );
	cout << "." << flush;
}

void rtRunTabSumtot()
{
	const double CONCSCALE = 1e-3; // Convert from uM to mM.

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Shell::cleanSimulation();

	Id modelId = shell->doLoadModel( "tabsumtot.g", "/ts", "rk5" );
	assert( modelId != Id() );
	Id stoichId( "/ts/stoich" );
	assert( stoichId != Id() );
	Id comptId( "/ts/kinetics" );
	assert( comptId != Id() );
	unsigned int numVarMols = Field< unsigned int >::get( 
		stoichId, "nVarPools" );
	assert ( numVarMols == 3 );

	double n;
	Id a( "/ts/kinetics/A" );
	assert( a != Id() );
	n = Field< double >::get( Id( "/ts/kinetics/A" ), "concInit" );
	assert( doubleEq( n, 1 * CONCSCALE ) );
	n = Field< double >::get( Id( "/ts/kinetics/A" ), "conc" );
	assert( doubleEq( n, 1 * CONCSCALE ) );
	n = Field< double >::get( Id( "/ts/kinetics/B" ), "concInit" );
	assert( doubleEq( n, 0 ) );
	n = Field< double >::get( Id( "/ts/kinetics/C" ), "concInit" );
	assert( doubleEq( n, 0 ) );

	Id d( "/ts/kinetics/D" );
	assert( d != Id() );
	assert( d.element()->cinfo()->name() == "ZombieBufPool" );
	const Finfo* dsf = d.element()->cinfo()->findFinfo( "set_concInit" );
	assert( dsf );
	const Finfo* asf = a.element()->cinfo()->findFinfo( "set_concInit" );
	assert( asf );
	assert( dsf != asf );
	/*
	n = Field< double >::get( Id( "/ts/kinetics/tot1" ), "conc" );
	assert( doubleEq( n, 1 * CONCSCALE ) );
	n = Field< double >::get( Id( "/ts/kinetics/tot2" ), "conc" );
	assert( doubleEq( n, 0 ) );
	*/

	///////////////////////////////////////////////////////////////////////
	// Now run it.
	///////////////////////////////////////////////////////////////////////

	shell->doSetClock( 0, 0.1 );
	shell->doSetClock( 1, 0.1 );
	shell->doSetClock( 2, 0.1 );
	shell->doReinit();
	shell->doStart( 20.0 );

	Id tab( "/ts/kinetics/xtab" );
	vector< double > vec = Field< vector< double > >::get( tab, "vec" );
	assert( vec.size() == 101 );
	for ( unsigned int i = 0; i < vec.size(); ++i )
		assert( doubleApprox( vec[i] / CONCSCALE, 1.0 + sin( 2.0 * PI * i / 100.0 ) ) );


	Id plotD( "/ts/graphs/conc2/D.Co" );
	assert( plotD != Id() );
	vector< double > vec2 = Field< vector< double > >::get( plotD, "vec" );
	assert( vec2.size() == 201 );
	// cout << "\n" << "i" << ": " << "vec[i]" << ",	" << "vec2[i]" << ",	" << "y" << endl;
	for ( unsigned int i = 0; i < vec2.size(); ++i ) {
		double y = 1.0 + sin( ( 2.0 * PI * i ) / 100.0 );
		// cout << "\n" << i << ": " << vec[i] << ",	" << vec2[i] << ",	" << y << endl;
		assert( doubleApprox( vec2[i], y * CONCSCALE ) );
	}

	Id plotTot1( "/ts/graphs/conc2/tot1.Co" );
	assert( plotTot1 != Id() );
	unsigned int size = Field< unsigned int >::get( plotTot1, "size" );
	assert( size == 201 ); // Note that dt was 0.1.
	vector< Id > ret = LookupField< string, vector< Id > >::get( 
		plotTot1, "neighbours", "requestData" );
	assert( ret.size() == 1 );
	assert( ret[0] == Id( "/ts/kinetics/tot1" ) );
	// We'll analyze the results analytically.

	bool ok = SetGet2< string, string >::set(
		Id( "/ts/graphs/conc1/A.Co" ), "xplot", "check.plot", "A.plot" );
	assert( ok );
	ok = SetGet2< string, string >::set(
		Id( "/ts/graphs/conc1/B.Co" ), "xplot", "check.plot", "B.plot" );
	assert( ok );
	ok = SetGet2< string, string >::set(
		Id( "/ts/graphs/conc2/C.Co" ), "xplot", "check.plot", "C.plot" );
	assert( ok );
	ok = SetGet2< string, string >::set(
		Id( "/ts/graphs/conc2/D.Co" ), "xplot", "check.plot", "D.plot" );
	assert( ok );
	ok = SetGet2< string, string >::set(
		Id( "/ts/graphs/conc2/tot2.Co" ), "xplot", "check.plot", 
			"tot2.plot" );
	assert( ok );

	vec = Field< vector< double > >::get( plotTot1, "vec");
	assert( vec.size() == 201 );
	for ( unsigned int i = 0; i < vec.size(); ++i )
		assert( doubleEq( vec[i], 1 * CONCSCALE ) );

	vec = Field< vector< double > >::get( Id( "/ts/graphs/conc2/tot2.Co" ),
		"vec");
	assert( vec.size() == 201 );
	vec2 = Field< vector< double > >::get(  Id( "/ts/graphs/conc1/B.Co" ),
		"vec");
	assert( vec2.size() == 201 );
	for ( unsigned int i = 0; i < vec.size(); ++i )
		assert( doubleEq( vec[i], vec2[i] ) );
	
	// dA/dt = -0.2 * ( 1 + sin( 2 * PI * t / 10 ) ) * A + 0.1 * (1-A)

	/////////////////////////////////////////////////////////////////////
	shell->doDelete( modelId );
	cout << "." << flush;
}
