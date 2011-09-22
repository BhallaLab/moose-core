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
#include "../kinetics/ReadCspace.h"

static void rtReplicateModels()
{
	static double expectedValueAtOneSec[] =
		{ 0.7908, 1.275, 1.628, 1.912, 2.154, 2.369, 2.564, 2.744 };
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );

	ReadCspace rc;
	Id modelId = rc.readModelString( "|AabX|Jacb| 1 1 1 0.01 0.1 0.1 1",
		"kinetics", Id(), "Neutral" );

	// Assign the CubeMesh dims as a 2x2x2 cube with mesh size of 1 um
	// to match the default in the ReadCspace.
	Id compt( "/kinetics/compartment" );
	assert( compt != Id() );
	vector< double > coords( 9, 0 );
	coords[0] = coords[1] = coords[2] = 0;
	coords[3] = coords[4] = coords[5] = 2e-6;
	coords[6] = coords[7] = coords[8] = 1e-6;
	
	bool ret = Field< bool >::set( compt, "preserveNumEntries", false );
	assert( ret );
	ret = Field< vector< double > >::set( compt, "coords", coords );
	assert( ret );
	Id mesh( "/kinetics/compartment/mesh" );
	assert( mesh != Id() );
	assert( mesh.element()->dataHandler()->localEntries() == 8 );

	shell->handleReMesh( mesh );
	// This should assign the same init conc to the new pool objects.

	Id a( "/kinetics/a" );
	assert( a != Id() );
	assert( a.element()->dataHandler()->localEntries() == 8 );

	vector< double > checkInit;
	Field< double >::getVec( a, "concInit", checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], 1 ) );

	// Here we create the stoich
	Id stoich = shell->doCreate( "Stoich", modelId, "stoich", dims );
	Field< string >::set( stoich, "path", "/kinetics/##" );
	unsigned int numVarMols = Field< unsigned int >::get( 
		stoich, "nVarPools" );
	assert ( numVarMols == 4 ); // 2 mols + 2 enz

	dims[0] = 8;
	Id gsl = shell->doCreate( "GslIntegrator", stoich, "gsl", dims );
	ret = SetGet1< Id >::setRepeat( gsl, "stoich", stoich );
	assert( ret);
	assert( gsl.element()->dataHandler()->localEntries() == 8 );

	// This second assertion on the # of localEntries on a is useful 
	// because it checks if something has gone wrong during zombification.
	assert( a.element()->dataHandler()->localEntries() == 8 );
	
	checkInit.resize( 0 );
	Field< double >::getVec( a, "concInit", checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], 1 ) );

	Field< double >::getVec( Id( "/kinetics/b" ), "concInit", checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], 1 ) );

	Field< double >::getVec( Id( "/kinetics/c" ), "concInit", checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], 1 ) );

	// Set up novel init conditions
	vector< double > init( 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		init[i] = i + 1;
	ret = Field< double >::setVec( a, "concInit", init );
	ret = Field< double >::setVec( Id( "/kinetics/b" ), "concInit", init );
	ret = Field< double >::setVec( Id( "/kinetics/c" ), "concInit", init );
	assert( ret);

	Field< double >::getVec( a, "concInit", checkInit );
	assert( checkInit.size() == 8 );
	for ( unsigned int i = 0; i < 8; ++i )
		assert( doubleEq( checkInit[i], init[i] ) );

	// Create the plots.
	Id plots = shell->doCreate( "Table", compt, "table", dims );
	MsgId mid = shell->doAddMsg( "OneToOne", plots, "requestData", a, "get_conc" );
	assert( mid != Msg::badMsg );

	// Set up scheduling.
	shell->doSetClock( 0, 0.1 );
	shell->doSetClock( 1, 0.1 );
	shell->doSetClock( 2, 0.1 );
	shell->doSetClock( 3, 0.1 );
	shell->doUseClock( "/kinetics/stoich/gsl", "process", 0 );
	shell->doUseClock( "/kinetics/##[TYPE=Table]", "process", 2 );
	// cout << "Before Reinit\n"; Qinfo::reportQ();
	shell->doReinit();

	// cout << "After Reinit\n"; Qinfo::reportQ();
	shell->doStart( 10 );

	unsigned int size = Field< unsigned int >::get( plots, "size" );
	// cout << "size = " << size << endl;
	assert( size == 101 ); // Note that dt was 1.
	
	/*
	ret = SetGet3< string, string, string >::set(
		plotId, "compareXplot", "Osc_cspace_ref_model.plot", 
		"plotd", "rmsr" );
	assert( ret );
	*/

	for ( unsigned int i = 0; i < 8; ++i ) {
		ObjId oid( plots, i );
		string name( "mesh" );
		name += '0' + i;
		ret = SetGet2< string, string >::set( oid, "xplot", "check.plot", 
			name );
		assert( ret );

		//Look up step # 10, starting from 0.
		double y = LookupField< unsigned int, double >::get( oid, "y", 10); 
		assert( doubleApprox( expectedValueAtOneSec[i], y ) );
		// cout << y << endl;
	}
	
	/*
	// Returns -1 on failure, otherwise the (positive) rms ratio.
	double val = Field< double >::get( plotId, "outputValue" );
	assert( val >= 0 && val < TOLERANCE );
	*/

	/////////////////////////////////////////////////////////////////////
	shell->doDelete( modelId );
	cout << "." << flush;
}

void rtReacDiff()
{
	rtReplicateModels();
}
