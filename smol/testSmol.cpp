/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "SmolHeader.h"
#include "ElementValueFinfo.h"
#include "ReduceBase.h"
#include "ReduceMax.h"
#include "../shell/Shell.h"


// This is a regression test
void testSmoldynZombify( )
{
	cout << "." << flush;
}

void testCreateSmolSim()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	unsigned int size = 1;
	vector< unsigned int > dims( 1, size );
	Id smolId = shell->doCreate("SmolSim", Id(), "smol", dims );
	Id paId = shell->doCreate("Neutral", Id(), "pa", dims );
	Id mol1Id = shell->doCreate("Mol", paId, "mol1", dims );
	Id mol2Id = shell->doCreate("Mol", paId, "mol2", dims );
	Id reacId = shell->doCreate("Reac", paId, "reac", dims );

	Field< double >::set( mol1Id, "nInit", 100 );
	Field< double >::set( mol2Id, "nInit", 0 );
	Field< double >::set( reacId, "kf", 2.0 );
	Field< double >::set( reacId, "kb", 1.0 );
	shell->doAddMsg( "Single", mol1Id, "reac", reacId, "sub" );
	shell->doAddMsg( "Single", mol2Id, "reac", reacId, "prd" );

	Id geomId = shell->doCreate( "Geometry", paId, "geom", dims );
	Id surfaceId = shell->doCreate( "Surface", geomId, "surf", dims );
	Id sphereId = shell->doCreate( "SpherePanel", surfaceId, "sphere", dims );
	vector< double > sphereCoords( 6, 0 );
	// coords 0 to 2 are the centre, let it be zero.
	sphereCoords[3] = 1; // radius.
	sphereCoords[4] = 1; // slice
	sphereCoords[5] = 1; // something
	Field< vector< double> >::set( sphereId, "coords", sphereCoords );

	SmolSim* ss = reinterpret_cast< SmolSim* >( smolId.eref().data() );
	assert( ss );

	simstruct* sim = ss->sim_;
	assert( sim == 0 );

	ss->setPath( smolId.eref(), 0, "/pa/##" );

	delete smolId();
	delete paId();
	cout << "#" << flush;
}

void testSmoldyn()
{
	testCreateSmolSim();
}
