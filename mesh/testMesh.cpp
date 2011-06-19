/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "../shell/Shell.h"
#include "MeshEntry.h"
#include "ChemMesh.h"
#include "CylMesh.h"

/**
 * Low-level tests for the CylMesh object: No MOOSE calls involved.
 */
void testCylMesh()
{
	CylMesh cm;
	vector< double > coords( 8 );
	coords[0] = 1;
	coords[1] = 2;
	coords[2] = 3;
	coords[3] = 1;

	coords[4] = 3;
	coords[5] = 5;
	coords[6] = 7;
	coords[7] = 2;

	cm.setCoords( coords );

	assert( doubleEq( cm.getX0(), 1 ) );
	assert( doubleEq( cm.getY0(), 2 ) );
	assert( doubleEq( cm.getZ0(), 3 ) );
	assert( doubleEq( cm.getR0(), 1 ) );

	assert( doubleEq( cm.getX1(), 3 ) );
	assert( doubleEq( cm.getY1(), 5 ) );
	assert( doubleEq( cm.getZ1(), 7 ) );
	assert( doubleEq( cm.getR1(), 2 ) );

	cm.setX0( 2 );
	cm.setY0( 3 );
	cm.setZ0( 4 );
	cm.setR0( 2 );

	cm.setX1( 4 );
	cm.setY1( 6 );
	cm.setZ1( 8 );
	cm.setR1( 3 );

	vector< double > temp = cm.getCoords();
	assert( temp.size() == 8 );
	for ( unsigned int i = 0; i < temp.size(); ++i )
		assert( doubleEq( temp[i], coords[i] + 1 ) );
	
	assert( doubleEq( cm.getTotLength() , sqrt( 29.0 ) ) );

	cm.setLambda( 1.0 );
	assert( cm.getNumEntries() == 5 );
	assert( doubleEq( cm.getLambda(), sqrt( 29.0 ) / 5 ) );

	cout << "." << flush;
}

void testMesh()
{
	testCylMesh();
}
