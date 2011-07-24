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


void rtFindModelType()
{
	ModelType findModelType( string filename, ifstream& fin );

	ifstream dotpfin( "ca1.p" );
	assert( dotpfin );
	assert( findModelType( "ca1.p", dotpfin ) == DOTP );

	ifstream kkitfin( "Kholodenko.g" );
	assert( kkitfin );
	assert( findModelType( "Kholodenko.g", kkitfin ) == KKIT );

	ifstream sbmlfin( "Kholodenko.xml" );
	assert( sbmlfin );
	assert( findModelType( "Kholodenko.xml", sbmlfin ) == UNKNOWN );
	cout << "." << flush;
}

void rtReadKkit()
{
	const double TOLERANCE = 2e-3;

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Shell::cleanSimulation();

	Id kineticId = shell->doLoadModel( "Kholodenko.g", "/rkktest" );
	assert( kineticId != Id() );
	unsigned int numVarMols = Field< unsigned int >::get( 
		kineticId, "nVarPools" );
	// assert ( numVarMols == 15 );

	Id gsl = shell->doCreate( "GslIntegrator", kineticId, "gsl", dims );
	bool ret = SetGet1< Id >::set( gsl, "stoich", kineticId );
	assert( ret );
	ret = Field< bool >::get( gsl, "isInitialized" );
	assert( ret );
	/*
	*/

	shell->doSetClock( 0, 10 );
	shell->doSetClock( 1, 10 );
	shell->doSetClock( 2, 10 );
	shell->doSetClock( 3, 0 );
	shell->doSetClock( 4, 0 );
	shell->doSetClock( 5, 0 );
	shell->doUseClock( "/rkktest/gsl", "process", 0 );
	shell->doUseClock( "/rkktest/graphs/##[TYPE=Table],/rkktest/moregraphs/##[TYPE=Table]", "process", 2 );

	cout << "Before Reinit\n";
	Qinfo::reportQ();
	shell->doReinit();
	cout << "Between Reinits\n";
	Qinfo::reportQ();
	shell->doReinit();
	cout << "After Reinit\n";
	Qinfo::reportQ();
	shell->doStart( 5001.0 );

	Id plotId( "/rkktest/graphs/conc1/MAPK-PP.Co" );
	assert( plotId != Id() );
	unsigned int size = Field< unsigned int >::get( plotId, "size" );
	assert( size == 502 ); // Note that dt was 10.
	
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
