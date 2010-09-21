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
	const double TOLERANCE = 1e-6;

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );

	Id kineticId = shell->doLoadModel( "Kholodenko.g", "/rkktest" );
	assert( kineticId != Id() );

	/*
	Id rkId = shell->doCreate( "KkitReader", Id(), "rkkit", dims );
	assert( rkId != Id() );
	Eref rk = rkId.eref();

	bool ok = SetGet3< string, string, Id >::set( 
		rk, "read", "readKkitTest.g", "kinetics", Id() );
	assert( ok );

	// run <method> using settings found from the kkit file.
	//kNeed to decide if to have option to override runtime.
	ok = SetGet2< string, string >::set( rk, "run", "rk5" );
	assert( ok );
	*/

	shell->doSetClock( 0, 10 );
	shell->doSetClock( 1, 10 );
	shell->doSetClock( 2, 10 );

	shell->doStart( 5000.0 );

	Id plotId( "/rkktest/graphs/conc1/MAPK-PP.Co" );
	assert( plotId != Id() );
	
	bool ok = SetGet::strSet( 
		plotId.eref(), "compareXplot", "Kholodenko.plot,/graphs/conc1/MAPK-PP.Co,rmsr" );
	/*
	bool ok = SetGet3< string, string, string >::set(
		plotId.eref(), "compareXplot", "readKkitTest.plot", "rmsr" );
		*/
	assert( ok );

	double val = Field< double >::get( plotId.eref(), "outputValue" );
	assert( val < TOLERANCE );

	/////////////////////////////////////////////////////////////////////
	shell->doDelete( kineticId );
	cout << "." << flush;
}
