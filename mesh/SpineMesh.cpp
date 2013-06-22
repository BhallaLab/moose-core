/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <cctype>
#include "header.h"
#include "SparseMatrix.h"
#include "Vec.h"

#include "ElementValueFinfo.h"
#include "Boundary.h"
#include "MeshEntry.h"
#include "ChemCompt.h"
#include "MeshCompt.h"
#include "CubeMesh.h"
#include "CylBase.h"
#include "NeuroNode.h"
#include "NeuroMesh.h"
#include "SpineEntry.h"
#include "SpineMesh.h"
#include "../utility/numutil.h"

/*
static SrcFinfo3< Id, vector< double >, vector< unsigned int > >* 
	psdListOut()
{
	static SrcFinfo3< Id, vector< double >, vector< unsigned int > >
   		psdListOut(
		"psdListOut",
		"Tells PsdMesh to build a mesh. "
		"Arguments: Cell Id, Coordinates of each psd, "
		"index of matching parent voxels for each spine"
		"The coordinates each have 8 entries:"
		"xyz of centre of psd, xyz of vector perpendicular to psd, "
		"psd diameter, "
		" diffusion distance from parent compartment to PSD"
	);
	return &psdListOut;
}
*/

const Cinfo* SpineMesh::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo spineList( "spineList",
			"Specifies the list of electrical compartments for the spine,"
			"and the associated parent voxel"
			"Arguments: cell container, shaft compartments, "
			"head compartments, parent voxel index ",
			new EpFunc4< SpineMesh, Id, vector< Id >, vector< Id >,
		   	vector< unsigned int > >(
				&SpineMesh::handleSpineList )
		);

		//////////////////////////////////////////////////////////////
		// Field Elements
		//////////////////////////////////////////////////////////////

	static Finfo* spineMeshFinfos[] = {
		&spineList,			// DestFinfo
		// psdListOut(),		// SrcFinfo
	};

	static Cinfo spineMeshCinfo (
		"SpineMesh",
		ChemCompt::initCinfo(),
		spineMeshFinfos,
		sizeof( spineMeshFinfos ) / sizeof ( Finfo* ),
		new Dinfo< SpineMesh >()
	);

	return &spineMeshCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* spineMeshCinfo = SpineMesh::initCinfo();

//////////////////////////////////////////////////////////////////
// Class stuff.
//////////////////////////////////////////////////////////////////
SpineMesh::SpineMesh()
	:
		surfaceGranularity_( 0.1 )
{;}

SpineMesh::SpineMesh( const SpineMesh& other )
	: 
		spines_( other.spines_ ),
		surfaceGranularity_( other.surfaceGranularity_ )
{;}

SpineMesh::~SpineMesh()
{
	;
}

//////////////////////////////////////////////////////////////////
// Field assignment stuff
//////////////////////////////////////////////////////////////////

/**
 * This assumes that lambda is the quantity to preserve, over numEntries.
 * So when the compartment changes size, so does numEntries.
 * Assumes that the soma node is at index 0.
 */
void SpineMesh::updateCoords()
{
	buildStencil();
}

Id SpineMesh::getCell() const
{
	return cell_;
}

unsigned int SpineMesh::innerGetDimensions() const
{
	return 3;
}

// Here we set up the spines. We don't permit heads without shafts.
void SpineMesh::handleSpineList( 
		const Eref& e, const Qinfo* q, 
		Id cell,
		vector< Id > shaft, vector< Id > head, 
		vector< unsigned int > parentVoxel )
{
		double oldVol = getMeshEntrySize( 0 );
		assert( head.size() == parentVoxel.size() );
		assert( head.size() == shaft.size() );
		spines_.resize( head.size() );
		cell_ = cell;

		vector< double > ret;
		vector< double > psdCoords;
		vector< unsigned int > index( head.size(), 0 );
		for ( unsigned int i = 0; i < head.size(); ++i ) {
			spines_[i] = SpineEntry( shaft[i], head[i], parentVoxel[i] );
			// ret = spines_[i].psdCoords();
			// assert( ret.size() == 8 );
			// psdCoords.insert( psdCoords.end(), ret.begin(), ret.end() );
			// index[i] = i;
		}
		// psdListOut()->send( e, q->threadNum(), cell_, psdCoords, index );

		updateCoords();
		Id meshEntry( e.id().value() + 1 );
		
		vector< unsigned int > localIndices( head.size() );
		vector< double > vols( head.size() );
		for ( unsigned int i = 0; i < head.size(); ++i ) {
			localIndices[i] = i;
			vols[i] = spines_[i].volume();
		}
		vector< vector< unsigned int > > outgoingEntries;
		vector< vector< unsigned int > > incomingEntries;
		meshSplit()->fastSend( e, q->threadNum(), oldVol, vols,
						localIndices, outgoingEntries, incomingEntries );
		lookupEntry( 0 )->triggerRemesh( meshEntry.eref(), q->threadNum(),
						oldVol, 0, localIndices, vols );
}

//////////////////////////////////////////////////////////////////
// FieldElement assignment stuff for MeshEntries
//////////////////////////////////////////////////////////////////

/// Virtual function to return MeshType of specified entry.
unsigned int SpineMesh::getMeshType( unsigned int fid ) const
{
	assert( fid < spines_.size() );
	return CYL;
}

/// Virtual function to return dimensions of specified entry.
unsigned int SpineMesh::getMeshDimensions( unsigned int fid ) const
{
	return 3;
}

/// Virtual function to return volume of mesh Entry.
double SpineMesh::getMeshEntrySize( unsigned int fid ) const
{
	if ( spines_.size() == 0 )
		return 1.0;
	assert( fid < spines_.size() );
	return spines_[ fid ].volume();
}

/// Virtual function to return coords of mesh Entry.
/// For SpineMesh, coords are x1y1z1 x2y2z2 x3y3z3 r0 r1
vector< double > SpineMesh::getCoordinates( unsigned int fid ) const
{
	vector< double > ret;
	return ret;
}

/// Virtual function to return diffusion X-section area for each neighbor
vector< double > SpineMesh::getDiffusionArea( unsigned int fid ) const
{
	vector< double > ret;
	return ret;
}

/// Virtual function to return scale factor for diffusion.
/// I think all dendite tips need to return just one entry of 1.
//  Regular points return vector( 2, 1.0 );
vector< double > SpineMesh::getDiffusionScaling( unsigned int fid ) const
{
	return vector< double >( 2, 1.0 );
}

/// Virtual function to return volume of mesh Entry, including
/// for diffusively coupled voxels from other solvers.
double SpineMesh::extendedMeshEntrySize( unsigned int fid ) const
{
	if ( fid < spines_.size() ) {
		return getMeshEntrySize( fid );
	} else {
		return MeshCompt::extendedMeshEntrySize( fid - spines_.size() );
	}
}



//////////////////////////////////////////////////////////////////
// Dest funcsl
//////////////////////////////////////////////////////////////////

/// More inherited virtual funcs: request comes in for mesh stats
/// Not clear what this does.
void SpineMesh::innerHandleRequestMeshStats( const Eref& e, const Qinfo* q, 
		const SrcFinfo2< unsigned int, vector< double > >* meshStatsFinfo
	)
{
		;
}

void SpineMesh::innerHandleNodeInfo(
			const Eref& e, const Qinfo* q, 
			unsigned int numNodes, unsigned int numThreads )
{
}
//////////////////////////////////////////////////////////////////

/**
 * Inherited virtual func. Returns number of MeshEntry in array
 */
unsigned int SpineMesh::innerGetNumEntries() const
{
	return spines_.size();
}

/**
 * Inherited virtual func. Assigns number of MeshEntries.
 * Doesn't do anything, we have to set spine # from geometry.
 */
void SpineMesh::innerSetNumEntries( unsigned int n )
{
}


/**
 * This is a bit odd, effectively asks to build an imaginary neuron and
 * then subdivide it. I'll make do with a ball-and-stick model: Soma with
 * a single apical dendrite with reasonable diameter. I will interpret
 * size as total length of neuron, not as volume. 
 * Soma will have a diameter of up to 20 microns, anything bigger than
 * this is treated as soma of 20 microns + 
 * dendrite of (specified length - 10 microns) for radius of soma.
 * This means we avoid having miniscule dendrites protruding from soma,
 * the shortest one will be 10 microns.
 */
void SpineMesh::innerBuildDefaultMesh( const Eref& e, const Qinfo* q,
	double size, unsigned int numEntries )
{
	cout << "Warning: attempt to build a default spine: not permitted\n";
}

//////////////////////////////////////////////////////////////////
const vector< SpineEntry >& SpineMesh::spines() const
{
		return spines_;
}

//////////////////////////////////////////////////////////////////
// Utility function to set up Stencil for diffusion
//////////////////////////////////////////////////////////////////
void SpineMesh::buildStencil()
{
// stencil_[0] = new NeuroStencil( nodes_, nodeIndex_, vs_, area_);
	setStencilSize( spines_.size(), spines_.size() );
	innerResetStencil();
}

//////////////////////////////////////////////////////////////////
// Utility function for junctions
//////////////////////////////////////////////////////////////////

void SpineMesh::matchMeshEntries( const ChemCompt* other,
	   vector< VoxelJunction >& ret ) const
{
	const CubeMesh* cm = dynamic_cast< const CubeMesh* >( other );
	if ( cm ) {
		matchCubeMeshEntries( other, ret );
		return;
	}
	const NeuroMesh* nm = dynamic_cast< const NeuroMesh* >( other );
	if ( nm ) {
		matchNeuroMeshEntries( other, ret );
		return;
	}
	cout << "Warning: SpineMesh::matchMeshEntries: unknown class\n";
}

void SpineMesh::indexToSpace( unsigned int index,
			double& x, double& y, double& z ) const 
{
	if ( index >= innerGetNumEntries() )
		return;
	spines_[ index ].mid( x, y, z );
}

double SpineMesh::nearest( double x, double y, double z, 
				unsigned int& index ) const
{
	double best = 1e12;
	index = 0;
	for( unsigned int i = 0; i < spines_.size(); ++i ) {
		const SpineEntry& se = spines_[i];
		double a0, a1, a2;
		se.mid( a0, a1, a2 );
		Vec a( a0, a1, a2 );
		Vec b( x, y, z );
		double d = a.distance( b );
		if ( best > d ) {
			best = d;
			index = i;
		}
	}
	if ( best == 1e12 )
		return -1;
	return best;
}

void SpineMesh::matchSpineMeshEntries( const ChemCompt* other,
	   vector< VoxelJunction >& ret ) const
{
}

void SpineMesh::matchNeuroMeshEntries( const ChemCompt* other,
	   vector< VoxelJunction >& ret ) const
{
	const NeuroMesh* nm = dynamic_cast< const NeuroMesh* >( other );
	assert( nm );
	// Check if NeuroMesh is parent of spines. If so, simple.
	if ( nm->getCell() == getCell() ) {
		for ( unsigned int i = 0; i < spines_.size(); ++i ) {
			double xda = spines_[i].rootArea() / spines_[i].diffusionLength();
			ret.push_back( VoxelJunction( i, spines_[i].parent(), xda ) );
		}
	} else {
		assert( 0 ); // Don't know how to do this yet.
	}
}

void SpineMesh::matchCubeMeshEntries( const ChemCompt* other,
	   vector< VoxelJunction >& ret ) const
{
	for( unsigned int i = 0; i < spines_.size(); ++i ) {
		const SpineEntry& se = spines_[i];
		se.matchCubeMeshEntriesToHead( other, i, surfaceGranularity_, ret );
	}
}
