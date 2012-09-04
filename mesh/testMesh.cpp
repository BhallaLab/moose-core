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
#include "Boundary.h"
#include "MeshEntry.h"
#include "Stencil.h"
#include "ChemMesh.h"
#include "CylMesh.h"
#include "CubeMesh.h"
#include "CylBase.h"
#include "NeuroNode.h"
#include "NeuroStencil.h"

/**
 * Low-level test for Cylbase, no MOOSE calls
 */
void testCylBase()
{
	// CylBase( x, y, z, dia, len, numDivs );
	CylBase  a( 0, 0, 0, 1,   1, 1 );
	CylBase  b( 1, 2, 3, 1,   2, 10 );

	assert( doubleEq( b.volume( a ), PI * 0.25 * 2 ) );
	assert( a.getNumDivs() == 1 );
	assert( b.getNumDivs() == 10 );

	for ( unsigned int i = 0; i < 10; ++i ) {
		assert( doubleEq( b.voxelVolume( a, i ), PI * 0.25 * 2 * 0.1 ) );
		vector< double > coords = b.getCoordinates( a, i );
		double x = i;
		assert( doubleEq( coords[0], x / 10.0 ) );
		assert( doubleEq( coords[1], x * 2.0 / 10.0 ) );
		assert( doubleEq( coords[2], x * 3.0  / 10.0 ) );
		assert( doubleEq( coords[3], (1.0 + x) / 10.0 ) );
		assert( doubleEq( coords[4], (1.0 + x) * 2.0 / 10.0 ) );
		assert( doubleEq( coords[5], (1.0 + x) * 3.0  / 10.0 ) );
		assert( doubleEq( coords[6], 0.5 ) );
		assert( doubleEq( coords[7], 0.5 ) );
		assert( doubleEq( coords[8], 0.0 ) );
		assert( doubleEq( coords[9], 0.0 ) );

		assert( doubleEq( b.getDiffusionArea( a, i ), PI * 0.25 ) );
	}
	b.setDia( 2.0 );
	assert( doubleEq( b.getDia(), 2.0 ) );
	assert( doubleEq( b.volume( a ), PI * (2*(0.5*0.5+0.5+1)/3.0) ) );
	for ( unsigned int i = 0; i < 10; ++i ) {
		double x = static_cast< double >( i ) / 10.0;
		double r0 = 0.5 * ( a.getDia() * ( 1.0 - x ) + b.getDia() * x );
		x = x + 0.1;
		double r1 = 0.5 * ( a.getDia() * ( 1.0 - x ) + b.getDia() * x );
		// len = 2, and we have 10 segments of it.
		double vv = 0.2 * ( r0 * r0 + r0 * r1 + r1 * r1 ) * PI / 3.0;
		assert( doubleEq( b.voxelVolume( a, i ), vv ) );
		assert( doubleEq( b.getDiffusionArea( a, i ), PI * r0 * r0 ) );
	}

	cout << "." << flush;
}

/**
 * Low-level test for NeuroNode, no MOOSE calls
 */
void testNeuroNode()
{
	CylBase  a( 0, 0, 0, 1,   1, 1 );
	CylBase  b( 1, 2, 3, 2,   2, 10 );
	CylBase  dcb( 1, 2, 3, 2,   2, 0 );
	vector< unsigned int > twoKids( 2, 2 );
	twoKids[0] = 2;
	twoKids[1] = 4;
	vector< unsigned int > oneKid( 1, 2 );
	vector< unsigned int > noKids( 0 );
	NeuroNode na( a, 0, twoKids, 0, Id(), true );
	NeuroNode ndummy( dcb, 0, oneKid, 1, Id(), false );
	NeuroNode nb( b, 1, noKids, 1, Id(), false );

	assert( na.parent() == 0 );
	assert( na.startFid() == 0 );
	assert( na.elecCompt() == Id() );
	assert( na.isDummyNode() == false );
	assert( na.isSphere() == true );
	assert( na.isStartNode() == true );
	assert( na.children().size() == 2 );
	assert( na.children()[0] == 2 );
	assert( na.children()[1] == 4 );
	assert( doubleEq( na.volume( a ), PI * 0.25 ) );

	assert( ndummy.parent() == 0 );
	assert( ndummy.startFid() == 1 );
	assert( ndummy.elecCompt() == Id() );
	assert( ndummy.isDummyNode() == true );
	assert( ndummy.isSphere() == false );
	assert( ndummy.isStartNode() == false );
	assert( ndummy.children().size() == 1 );
	assert( ndummy.children()[0] == 2 );

	assert( nb.parent() == 1 );
	assert( nb.startFid() == 1 );
	assert( nb.elecCompt() == Id() );
	assert( nb.isDummyNode() == false );
	assert( nb.isSphere() == false );
	assert( nb.isStartNode() == false );
	assert( nb.children().size() == 0 );
	assert( doubleEq( nb.volume( a ), PI * (2*(0.5*0.5+0.5+1)/3.0) ) );

	cout << "." << flush;
}

void zebraConcPattern( vector< vector< double > >& S )
{
	// Set up an alternating pattern of zero/nonzero voxels to do test
	// calculations
	assert( S.size() == 44 );
	for ( unsigned int i = 0; i < S.size(); ++i )
		S[i][0] = 0;

	S[0][0] = 100; // soma
	S[2][0] = 10; // basal dend
	S[4][0] = 1; // basal
	S[6][0] = 10; // basal
	S[8][0] = 1; // basal
	S[10][0] = 10; // basal
	S[12][0] = 1; // basal
	S[14][0] = 10; // basal
	S[16][0] = 1; // basal
	S[18][0] = 10; // basal
	S[20][0] = 1; // basal tip

	S[22][0] = 10; // Apical left dend second compt.
	S[24][0] = 1; // Apical left dend
	S[26][0] = 10; // Apical left dend
	S[28][0] = 1; // Apical left dend
	S[30][0] = 10; // Apical left dend last compt

	// Tip compts are 31 and 32:
	S[32][0] = 1; // Apical middle Tip compartment.

	S[34][0] = 10; // Apical right dend second compt.
	S[36][0] = 1; // Apical right dend
	S[38][0] = 10; // Apical right dend
	S[40][0] = 1; // Apical right dend
	S[42][0] = 10; // Apical right dend last compt

	S[43][0] = 1; // Apical right tip.
}

/// Set up uniform concentrations in each voxel. Only the end voxels
/// should see a nonzero flux.
void uniformConcPattern( vector< vector< double > >& S, 
				vector< double >& vs )
{
	// Set up uniform concs throughout. This means to scale each # by 
	// voxel size.
	assert( S.size() == 44 );
	for ( unsigned int i = 0; i < S.size(); ++i )
		S[i][0] = 10 * vs[i];
}

/**
 * Low-level test for NeuroStencil, no MOOSE calls
 * This outlines the policy for how nodes and dummies are 
 * organized.
 * The set of nodes represents a neuron like this:
 *
 *  \/ /  1
 *   \/   2
 *    |   3
 *    |   4
 * The soma is at the top of the vertical section. Diameter = 10 microns.
 * Each little segment is a NeuroNode. Assume 10 microns each.
 * Level 1 has only 1 voxel in each segment. 1 micron at tip
 * Level 2 has 10 voxels in each segment. 1 micron at tip,
 *   2 microns at soma surface
 * The soma is between levels 2 and 3 and is a single voxel.
 *  10 microns diameter.
 * Level 3 has 10 voxels. 3 micron at soma, 2 microns at tip
 * Level 4 has 10 voxels. 1 micron at tip.
 * We have a total of 8 full nodes, plus 3 dummy nodes for 
 * the segments at 2 and 3, since they plug into a sphere.
	Order is depth first: 
			0: soma
			1: basal dummy
			2: basal segment at level 3
			3: basal segment at level 4
			4: left dummy
			5: left segment at level 2, apicalL1
			6: left segment at level 1, apicalL2
			7: other (middle) segment at level 1;  apicalM
			8: right dummy
			9: right segment at level 2, apicalR1
			10: right segment at level 1, apicalR2
 */
unsigned int buildNode( vector< NeuroNode >& nodes, unsigned int parent, 
		double x, double y, double z, double dia, 
		unsigned int numDivs, bool isDummy, unsigned int startFid )
{
	x *= 1e-6;
	y *= 1e-6;
	z *= 1e-6;
	dia *= 1e-6;
	CylBase cb( x, y, z, dia, dia/2, numDivs );
	vector< unsigned int > kids( 0 );
	if ( nodes.size() == 0 ) { // make soma
		NeuroNode nn( cb, parent, kids, startFid, Id(), true );
		nodes.push_back( nn );
	} else if ( isDummy ) { // make dummy
		NeuroNode nn( cb, parent, kids, startFid, Id(), false );
		nodes.push_back( nn );
	} else {
		NeuroNode nn( cb, parent, kids, startFid, Id(), false );
		nodes.push_back( nn );
	}
	NeuroNode& paNode = nodes[ parent ];
	nodes.back().calculateLength( paNode );
	return startFid + numDivs;
}

void testNeuroStencilFlux()
{
	NeuroStencil ns;
	vector< double > flux( 1, 0 );
	vector< double > tminus( 1, 10.0 );
	vector< double > t0( 1, 100.0 );
	vector< double > tplus( 1, 1000.0 );
	vector< double > diffConst( 1, 1.0 );

	ns.addHalfFlux( 0, flux, t0, tminus, 1, 1, 1, 1, diffConst );
	assert( doubleEq( flux[0], -90 ) );

	flux[0] = 0;
	ns.addLinearFlux( 0, flux, tminus, t0, tplus, 1, 1, 1, 1, 1, 1, diffConst );
	assert( doubleEq( flux[0], 1000 + 10 - 200 ) );

	cout << "." << flush;
}

void connectChildNodes( vector< NeuroNode >& nodes )
{
	assert( nodes.size() == 11 );
	for ( unsigned int i = 1; i < nodes.size(); ++i ) {
		assert( nodes[i].parent() < nodes.size() );
		NeuroNode& parent = nodes[ nodes[i].parent() ];
		parent.addChild( i );
	}
	////////////////////////////////////////////////////////////////////
	// Check children
	assert( nodes[0].children().size() == 3 );
	assert( nodes[0].children()[0] == 1 );
	assert( nodes[0].children()[1] == 4 );
	assert( nodes[0].children()[2] == 8 );
	assert( nodes[1].children().size() == 1 );
	assert( nodes[1].children()[0] == 2 );
	assert( nodes[2].children().size() == 1 );
	assert( nodes[2].children()[0] == 3 );
	assert( nodes[3].children().size() == 0 );
	assert( nodes[4].children().size() == 1 );
	assert( nodes[4].children()[0] == 5 );
	assert( nodes[5].children().size() == 2 );
	assert( nodes[5].children()[0] == 6 );
	assert( nodes[5].children()[1] == 7 );
	assert( nodes[6].children().size() == 0 );
	assert( nodes[7].children().size() == 0 );
	assert( nodes[8].children().size() == 1 );
	assert( nodes[8].children()[0] == 9 );
	assert( nodes[9].children().size() == 1 );
	assert( nodes[9].children()[0] == 10 );
	assert( nodes[10].children().size() == 0 );
}


void testNeuroStencil()
{
	const unsigned int numNodes = 11;
	const unsigned int numVoxels = 44; // 4 with 10 each, soma, 3 branch end
	vector< NeuroNode > nodes;
	vector< unsigned int> nodeIndex( numVoxels, 0 );
	vector< double > vs( numVoxels, 1 );
	vector< double > area( numVoxels, 1 );
	double sq = 10.0 / sqrt( 2 );
		
	// CylBase( x, y, z, dia, len, numDivs )
	// NeuroNode( cb, parent, children, startFid, elecCompt, isSphere )
	// buildNode( nodes, pa, x, y, z, dia, numDivs, isDummy, startFid )
	unsigned int startFid = 0;
	////////////////////////////////////////////////////////////////////
	// Build the nodes
	////////////////////////////////////////////////////////////////////
	startFid = buildNode( nodes, 0, 0, 0, 0, 10, 1, 0, startFid ); // soma
	startFid = buildNode( nodes, 0, 0, -5, 0, 3, 0, 1, startFid ); // dummy1
	startFid = buildNode( nodes, 1, 0, -15, 0, 2, 10, 0, startFid ); // B1
	startFid = buildNode( nodes, 2, 0, -25, 0, 1, 10, 0, startFid ); // B2

	startFid = buildNode( nodes, 0, 0, 5, 0, 2, 0, 1, startFid ); // dummy2
	startFid = buildNode( nodes, 4, -sq, 5+sq, 0, 1, 10, 0, startFid ); // AL1
	startFid = buildNode( nodes, 5, -sq*2, 5+sq*2, 0, 1, 1, 0, startFid ); // AL2
	startFid = buildNode( nodes, 5, 0, 5+sq*2, 0, 1, 1, 0, startFid ); // AM
	startFid = buildNode( nodes, 0, 0, 5, 0, 2, 0, 1, startFid ); // dummy3
	startFid = buildNode( nodes, 8, sq, 5+sq, 0, 1, 10, 0, startFid ); //AR1
	startFid = buildNode( nodes, 9, sq*2, 5+sq*2, 0, 1, 1, 0, startFid ); //AR2
	assert( startFid == numVoxels );
	assert( nodes.size() == numNodes );
	connectChildNodes( nodes );

	////////////////////////////////////////////////////////////////////
	// Check lengths
	for ( unsigned int i = 0; i < numNodes; ++i ) {
		if ( !( nodes[i].isDummyNode() || nodes[i].isSphere() ) )
			assert( doubleEq( nodes[i].getLength(), 10e-6 ) );
	}
	////////////////////////////////////////////////////////////////////
	// Build nodeIndex vs and area arrays.
	for ( unsigned int i = 0; i < numNodes; ++i ) {
		assert( nodes[i].parent() < numNodes );
		NeuroNode& parent = nodes[ nodes[i].parent() ];
		unsigned int end = nodes[i].startFid() + nodes[i].getNumDivs();
		for ( unsigned int j = nodes[i].startFid(); j < end; ++j ) {
			assert( j < numVoxels );
			nodeIndex[j] = i;
			unsigned int k = j - nodes[i].startFid();
			vs[j] = 0.001 * NA * nodes[i].voxelVolume( parent, k );
			area[j] = nodes[i].getDiffusionArea( parent, k );
		}
	}
	assert( doubleEq( vs[ numVoxels-1 ], NA * 0.001 * PI * 10e-6 * 0.25e-12 ) );
	assert( doubleEq( area[ numVoxels-1 ], PI * 0.25e-12 ) );
	////////////////////////////////////////////////////////////////////
	// Setup done. Now build stencil and test it.
	////////////////////////////////////////////////////////////////////
	vector< double > diffConst( 1, 1e-12 );
	vector< double > temp( 1, 0.0 );
	vector< vector< double > > flux( numVoxels, temp );
	vector< double > molNum( 1, 0 ); // We only have one pool

	// S[meshIndex][poolIndex]
	vector< vector< double > > S( numVoxels, molNum ); 

	NeuroStencil ns( nodes, nodeIndex, vs, area );

	zebraConcPattern( S );

	for ( unsigned int i = 0; i < numVoxels; ++i ) {
		ns.addFlux( i, flux[i], S, diffConst );
	}
	/*
	for ( unsigned int i = 0; i < numVoxels; ++i ) {
		cout << "S[" << i << "][0] = " << S[i][0] << 
			", flux[" << i << "][0] = " << flux[i][0] << endl;
	}
	*/

	uniformConcPattern( S, vs );

	for ( unsigned int i = 0; i < numVoxels; ++i ) {
		flux[i][0] = 0.0;
		ns.addFlux( i, flux[i], S, diffConst );
	}
	/*
	for ( unsigned int i = 0; i < numVoxels; ++i ) {
		cout << "S[" << i << "][0] = " << S[i][0] << 
			", flux[" << i << "][0] = " << flux[i][0] << endl;
	}
	*/
	
	cout << "." << flush;
}

/**
 * Low-level tests for the CylMesh object: No MOOSE calls involved.
 */
void testCylMesh()
{
	CylMesh cm;
	assert( cm.getMeshType( 0 ) == CYL );
	assert( cm.getMeshDimensions( 0 ) == 3 );
	assert( cm.getDimensions() == 3 );

	vector< double > coords( 9 );
	coords[0] = 1; // X0
	coords[1] = 2; // Y0
	coords[2] = 3; // Z0

	coords[3] = 3; // X1
	coords[4] = 5; // Y1
	coords[5] = 7; // Z1

	coords[6] = 1; // R0
	coords[7] = 2; // R1

	coords[8] = 1; // lambda

	cm.innerSetCoords( coords );

	assert( doubleEq( cm.getX0(), 1 ) );
	assert( doubleEq( cm.getY0(), 2 ) );
	assert( doubleEq( cm.getZ0(), 3 ) );
	assert( doubleEq( cm.getR0(), 1 ) );

	assert( doubleEq( cm.getX1(), 3 ) );
	assert( doubleEq( cm.getY1(), 5 ) );
	assert( doubleEq( cm.getZ1(), 7 ) );
	assert( doubleEq( cm.getR1(), 2 ) );

	assert( doubleEq( cm.getLambda(), sqrt( 29.0 ) / 5.0 ) );

	cm.setX0( 2 );
	cm.setY0( 3 );
	cm.setZ0( 4 );
	cm.setR0( 2 );

	cm.setX1( 4 );
	cm.setY1( 6 );
	cm.setZ1( 8 );
	cm.setR1( 3 );

	cm.setLambda( 2.0 );

	vector< double > temp = cm.getCoords( Id().eref(), 0 );
	assert( temp.size() == 9 );
	// Can't check on the last coord as it is lambda, it changes.
	for ( unsigned int i = 0; i < temp.size() - 1; ++i )
		assert( doubleEq( temp[i], coords[i] + 1 ) );
	
	double totLen = sqrt( 29.0 );
	assert( doubleEq( cm.getTotLength() , totLen ) );

	cm.setLambda( 1.0 );
	assert( cm.getNumEntries() == 5 );
	assert( doubleEq( cm.getLambda(), totLen / 5 ) );

	///////////////////////////////////////////////////////////////
	assert( doubleEq( cm.getMeshEntrySize( 2 ), 2.5 * 2.5 * PI * totLen / 5 ) );

	///////////////////////////////////////////////////////////////
	// LenSlope/totLen = 0.016 = 
	// 	1/numEntries * (r1-r0)/numEntries * 2/(r0+r1) = 1/25 * 1 * 2/5
	// Here are the fractional positions
	// part0 = 1/5 - 0.032: end= 0.2 - 0.032
	// part1 = 1/5 - 0.016: end = 0.4 - 0.048
	// part2 = 1/5			: end = 0.6 - 0.048
	// part3 = 1/5 + 0.016	: end = 0.8 - 0.032
	// part4 = 1/5 + 0.032	: end = 1

	coords = cm.getCoordinates( 2 );
	assert( coords.size() == 10 );
	assert( doubleEq( coords[0], 2 + (0.4 - 0.048) * 2 ) );
	assert( doubleEq( coords[1], 3 + (0.4 - 0.048) * 3 ) );
	assert( doubleEq( coords[2], 4 + (0.4 - 0.048) * 4 ) );

	assert( doubleEq( coords[3], 2 + (0.6 - 0.048) * 2 ) );
	assert( doubleEq( coords[4], 3 + (0.6 - 0.048) * 3 ) );
	assert( doubleEq( coords[5], 4 + (0.6 - 0.048) * 4 ) );

	assert( doubleEq( coords[6], 2.4 ) );
	assert( doubleEq( coords[7], 2.6 ) );

	///////////////////////////////////////////////////////////////
	vector< unsigned int > neighbors = cm.getNeighbors( 2 );
	assert( neighbors.size() == 2 );
	assert( neighbors[0] == 1 );
	assert( neighbors[1] == 3 );

	///////////////////////////////////////////////////////////////
	coords = cm.getDiffusionArea( 2 );
	assert( coords.size() == 2 );
	assert( doubleEq( coords[0], 2.4 * 2.4 * PI ) );
	assert( doubleEq( coords[1], 2.6 * 2.6 * PI ) );

	cout << "." << flush;
}


/**
 * mid-level tests for the CylMesh object, using MOOSE calls.
 */
void testMidLevelCylMesh()
{
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );

	Id cylId = s->doCreate( "CylMesh", Id(), "cyl", dims, 0 );
	Id meshId( cylId.value() + 1 );

	vector< double > coords( 9 );
	coords[0] = 1; // X0
	coords[1] = 2; // Y0
	coords[2] = 3; // Z0

	coords[3] = 3; // X1
	coords[4] = 5; // Y1
	coords[5] = 7; // Z1

	coords[6] = 1; // R0
	coords[7] = 2; // R1

	coords[8] = 1; // lambda

	bool ret = Field< vector< double > >::set( cylId, "coords", coords );
	assert( ret );

	assert( doubleEq( Field< double >::get( cylId, "x0" ), 1 ) );
	assert( doubleEq( Field< double >::get( cylId, "y0" ), 2 ) );
	assert( doubleEq( Field< double >::get( cylId, "z0" ), 3 ) );
	assert( doubleEq( Field< double >::get( cylId, "x1" ), 3 ) );
	assert( doubleEq( Field< double >::get( cylId, "y1" ), 5 ) );
	assert( doubleEq( Field< double >::get( cylId, "z1" ), 7 ) );
	assert( doubleEq( Field< double >::get( cylId, "r0" ), 1 ) );
	assert( doubleEq( Field< double >::get( cylId, "r1" ), 2 ) );

	ret = Field< double >::set( cylId, "lambda", 1 );
	assert( ret );
	meshId.element()->syncFieldDim();

	assert( meshId()->dataHandler()->localEntries() == 5 );

	unsigned int n = Field< unsigned int >::get( cylId, "num_mesh" );
	assert( n == 5 );

	ObjId oid( meshId, DataId( 2 ) );

	double totLen = sqrt( 29.0 );
	assert( doubleEq( Field< double >::get( oid, "size" ),
		1.5 * 1.5 * PI * totLen / 5 ) );

	vector< unsigned int > neighbors = 
		Field< vector< unsigned int > >::get( oid, "neighbors" );
	assert( neighbors.size() == 2 );
	assert( neighbors[0] = 1 );
	assert( neighbors[1] = 3 );

	s->doDelete( cylId );

	cout << "." << flush;
}

/**
 * Low-level tests for the CubeMesh object: No MOOSE calls involved.
 */
void testCubeMesh()
{
	CubeMesh cm;
	cm.setPreserveNumEntries( 0 );
	assert( cm.getMeshType( 0 ) == CUBOID );
	assert( cm.getMeshDimensions( 0 ) == 3 );
	assert( cm.getDimensions() == 3 );

	vector< double > coords( 9 );
	coords[0] = 0; // X0
	coords[1] = 0; // Y0
	coords[2] = 0; // Z0

	coords[3] = 2; // X1
	coords[4] = 4; // Y1
	coords[5] = 8; // Z1

	coords[6] = 1; // DX
	coords[7] = 1; // DY
	coords[8] = 1; // DZ

	cm.innerSetCoords( coords );

	assert( doubleEq( cm.getX0(), 0 ) );
	assert( doubleEq( cm.getY0(), 0 ) );
	assert( doubleEq( cm.getZ0(), 0 ) );

	assert( doubleEq( cm.getX1(), 2 ) );
	assert( doubleEq( cm.getY1(), 4 ) );
	assert( doubleEq( cm.getZ1(), 8 ) );

	assert( doubleEq( cm.getDx(), 1 ) );
	assert( doubleEq( cm.getDy(), 1 ) );
	assert( doubleEq( cm.getDz(), 1 ) );

	cm.setX0( 1 );
	cm.setY0( 2 );
	cm.setZ0( 4 );

	cm.setX1( 5 );
	cm.setY1( 6 );
	cm.setZ1( 8 );

	vector< double > temp = cm.getCoords( Id().eref(), 0 );
	assert( temp.size() == 9 );
	assert( doubleEq( temp[0], 1 ) );
	assert( doubleEq( temp[1], 2 ) );
	assert( doubleEq( temp[2], 4 ) );
	assert( doubleEq( temp[3], 5 ) );
	assert( doubleEq( temp[4], 6 ) );
	assert( doubleEq( temp[5], 8 ) );
	assert( doubleEq( temp[6], 1 ) );
	assert( doubleEq( temp[7], 1 ) );
	assert( doubleEq( temp[8], 1 ) );

	vector< unsigned int > m2s;
	vector< unsigned int > s2m( 64, ~0);
	for ( unsigned int i = 0; i < 4; ++i ) {
		for ( unsigned int j = 0; j < 4; ++j ) {
			for ( unsigned int k = 0; k < 4; ++k ) {
				unsigned int index = ( i * 4 + j ) * 4 + k;
				if ( i*i + j * j + k * k < 15 ) { 
					// include shell around 1 octant of sphere
					s2m[index] = m2s.size();
					m2s.push_back( index );
				} else {
					s2m[index] = ~0;
				}
			}
		}
	}
	unsigned int numEntries = m2s.size();
	cm.setMeshToSpace( m2s );
	cm.setSpaceToMesh( s2m );

	assert( cm.getNumEntries() == numEntries );
	assert( cm.getMeshEntrySize( 0 ) == 1 );

	coords = cm.getCoordinates( 0 );
	assert( doubleEq( coords[0], 1 ) );
	assert( doubleEq( coords[1], 2 ) );
	assert( doubleEq( coords[2], 4 ) );
	assert( doubleEq( coords[3], 2 ) );
	assert( doubleEq( coords[4], 3 ) );
	assert( doubleEq( coords[5], 5 ) );

	vector< unsigned int > neighbors = cm.getNeighbors( 0 );
	assert( neighbors.size() == 3 );
	assert( neighbors[0] = 1 );
	assert( neighbors[1] = s2m[ 4 ] );
	assert( neighbors[2] = s2m[ 16 ] );



	cm.setPreserveNumEntries( 1 );
	assert( cm.getNx() == 4 );
	assert( cm.getNy() == 4 );
	assert( cm.getNz() == 4 );
	assert( doubleEq( cm.getDx(), 1.0 ) );
	assert( doubleEq( cm.getDy(), 1.0 ) );
	assert( doubleEq( cm.getDz(), 1.0 ) );

	cm.setX0( 0 );
	cm.setY0( 0 );
	cm.setZ0( 0 );
	// x1 is 5, y1 is 6 and z1 is 8

	assert( doubleEq( cm.getDx(), 1.25 ) );
	assert( doubleEq( cm.getDy(), 1.5 ) );
	assert( doubleEq( cm.getDz(), 2.0 ) );

	cout << "." << flush;
}

/**
 * Tests scaling of volume and # of entries upon mesh resize
 */
void testReMesh()
{
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );
	Id base = s->doCreate( "Neutral", Id(), "base", dims );

	Id pool = s->doCreate( "Pool", base, "pool", dims );
	Id cube = s->doCreate( "CubeMesh", base, "cube", dims );
	bool ret = SetGet2< double, unsigned int >::set( 
		cube, "buildDefaultMesh", 1.0, 1 );
	assert( ret );
	Id mesh( "/base/cube/mesh" );
	assert( mesh != Id() );

	MsgId mid = s->doAddMsg( "OneToOne", pool, "mesh", mesh, "mesh" );
	assert( mid != Msg::bad );

/////////////////////////////////////////////////////////////////
	// 1 millimolar in 1 m^3 is 1 mole per liter.
	unsigned int linsize = Field< unsigned int >::get( pool, "linearSize" );
	assert( linsize == 1 );

	ret = Field< double >::set( pool, "conc", 1 );
	assert( ret );
	double n = Field< double >::get( pool, "n" );
	assert( doubleEq( n, NA ) );

	ret = SetGet2< double, unsigned int >::set( 
		cube, "buildDefaultMesh", 1.0e-3, 1 );
	Field< double >::set( pool, "conc", 1 );
	n = Field< double >::get( pool, "n" );
	assert( doubleEq( n, NA / 1000.0 ) );

/////////////////////////////////////////////////////////////////
	// Next we do the remeshing.
	double x = 1.234;
	Field< double >::set( pool, "concInit", x );
	ret = SetGet2< double, unsigned int >::set( 
		cube, "buildDefaultMesh", 1, 8 );
	// This is nasty, needs the change to propagate through messages.
	// Qinfo::waitProcCycles( 2 );
	linsize = Field< unsigned int >::get( pool, "linearSize" );
	assert( linsize == 8 );

	n = Field< double >::get( ObjId( pool, 0 ), "concInit" );
	assert( doubleEq( n, x ) );
	n = Field< double >::get( ObjId( pool, 7 ), "concInit" );
	assert( doubleEq( n, x ) );
	n = Field< double >::get( ObjId( pool, 0 ), "nInit" );
	assert( doubleEq( n, x * NA / 8.0 ) );
	n = Field< double >::get( ObjId( pool, 7 ), "nInit" );
	assert( doubleEq( n, x * NA / 8.0 ) );
	n = Field< double >::get( ObjId( pool, 0 ), "conc" );
	assert( doubleEq( n, x ) );
	n = Field< double >::get( ObjId( pool, 7 ), "conc" );
	assert( doubleEq( n, x ) );

/////////////////////////////////////////////////////////////////
	s->doDelete( base );
	cout << "." << flush;
}

void testMesh()
{
	testCylBase();
	testNeuroNode();
	testNeuroStencilFlux();
	testNeuroStencil();
	testCylMesh();
	testMidLevelCylMesh();
	testCubeMesh();
	testReMesh();
}
