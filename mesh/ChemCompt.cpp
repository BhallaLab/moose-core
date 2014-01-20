/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "LookupElementValueFinfo.h"
#include "Boundary.h"
#include "MeshEntry.h"
#include "ChemCompt.h"
//#include "../ksolve/StoichHeaders.h"

/*
SrcFinfo5< 
	double,
	vector< double >,
	vector< unsigned int>, 
	vector< vector< unsigned int > >, 
	vector< vector< unsigned int > >
	>* meshSplitOut()
{
	static SrcFinfo5< 
			double,
			vector< double >,
			vector< unsigned int >, 
			vector< vector< unsigned int > >, 
			vector< vector< unsigned int > >
		>
	meshSplitOut(
		"meshSplitOut",
		"Defines how meshEntries communicate between nodes."
		"Args: oldVol, volListOfAllEntries, localEntryList, "
		"outgoingDiffusion[node#][entry#], incomingDiffusion[node#][entry#]"
		"This message is meant to go to the SimManager and Stoich."
	);
	return &meshSplitOut;
}

static SrcFinfo2< unsigned int, vector< double > >* meshStats()
{
	static SrcFinfo2< unsigned int, vector< double > > meshStats(
		"meshStats",
		"Basic statistics for mesh: Total # of entries, and a vector of"
		"unique volumes of voxels"
	);
	return &meshStats;
}
*/

const Cinfo* ChemCompt::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ChemCompt, double > volume(
			"volume",
			"Volume of entire chemical domain."
			"Assigning this assumes that the geometry is that of the "
			"default mesh, which may not be what you want. If so, use"
			"a more specific mesh assignment function.",
			&ChemCompt::setEntireVolume,
			&ChemCompt::getEntireVolume
		);

		static ReadOnlyLookupElementValueFinfo< 
				ChemCompt, unsigned int, double > 
			voxelVolume(
			"voxelVolume",
			"Volume of specified voxel.",
			&ChemCompt::getVoxelVolume
		);

		static ReadOnlyValueFinfo< ChemCompt, unsigned int > numDimensions(
			"numDimensions",
			"Number of spatial dimensions of this compartment. Usually 3 or 2",
			&ChemCompt::getDimensions
		);

		static ValueFinfo< ChemCompt, string > method(
			"method",
			"Advisory field for SimManager to check when assigning "
			"solution methods. Doesn't do anything unless SimManager scans",
			&ChemCompt::setMethod,
			&ChemCompt::getMethod
		);

		static ReadOnlyLookupValueFinfo< ChemCompt, unsigned int, vector< double > > stencilRate(
			"stencilRate",
			"vector of diffusion rates in the stencil for specified voxel."
			"The identity of the coupled voxels is given by the partner "
			"field 'stencilIndex'."
			"Returns an empty vector for non-voxelized compartments.",
			&ChemCompt::getStencilRate
		);

		static ReadOnlyLookupValueFinfo< ChemCompt, unsigned int, vector< unsigned int > > stencilIndex(
			"stencilIndex",
			"vector of voxels diffusively coupled to the specified voxel."
			"The diffusion rates into the coupled voxels is given by the "
			"partner field 'stencilRate'."
			"Returns an empty vector for non-voxelized compartments.",
			&ChemCompt::getStencilIndex 
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo buildDefaultMesh( "buildDefaultMesh",
			"Tells ChemCompt derived class to build a default mesh with the"
			"specified volume and number of meshEntries.",
			new EpFunc2< ChemCompt, double, unsigned int >( 
				&ChemCompt::buildDefaultMesh )
		);

		/*
		static DestFinfo handleRequestMeshStats( "handleRequestMeshStats",
			"Handles request from SimManager for mesh stats",
			new EpFunc0< ChemCompt >(
				&ChemCompt::handleRequestMeshStats
			)
		);
		*/

		static DestFinfo handleNodeInfo( "handleNodeInfo",
			"Tells ChemCompt how many nodes and threads per node it is "
			"allowed to use. Triggers a return meshSplitOut message.",
			new EpFunc2< ChemCompt, unsigned int, unsigned int >(
				&ChemCompt::handleNodeInfo )
		);

		static DestFinfo resetStencil( "resetStencil",
			"Resets the diffusion stencil to the core stencil that only "
			"includes the within-mesh diffusion. This is needed prior to "
			"building up the cross-mesh diffusion through junctions.",
			new OpFunc0< ChemCompt >(
				&ChemCompt::resetStencil )
		);


		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////

		/*
		static Finfo* nodeMeshingShared[] = {
			meshSplitOut(), meshStats(), 
			&handleRequestMeshStats, &handleNodeInfo
		};

		static SharedFinfo nodeMeshing( "nodeMeshing",
			"Connects to SimManager to coordinate meshing with parallel"
			"decomposition and with the Stoich",
			nodeMeshingShared, sizeof( nodeMeshingShared ) / sizeof( const Finfo* )
		);
		*/

		/*
		static Finfo* geomShared[] = {
			&requestSize, &handleSize
		};

		static SharedFinfo geom( "geom",
			"Connects to Geometry tree(s) defining compt",
			geomShared, sizeof( geomShared ) / sizeof( const Finfo* )
		);
		*/

		//////////////////////////////////////////////////////////////
		// Field Elements
		//////////////////////////////////////////////////////////////
		/*
		static FieldElementFinfo< ChemCompt, Boundary > boundaryFinfo( 
			"boundary", 
			"Field Element for Boundaries",
			Boundary::initCinfo(),
			&ChemCompt::lookupBoundary,
			&ChemCompt::setNumBoundary,
			&ChemCompt::getNumBoundary,
			4
		);
		*/

		static FieldElementFinfo< ChemCompt, MeshEntry > entryFinfo( 
			"mesh", 
			"Field Element for mesh entries",
			MeshEntry::initCinfo(),
			&ChemCompt::lookupEntry,
			&ChemCompt::setNumEntries,
			&ChemCompt::getNumEntries,
			false
		);

	static Finfo* chemMeshFinfos[] = {
		&volume,			// Value
		&voxelVolume,		// ReadOnlyLookupValue
		&numDimensions,	// ReadOnlyValue
		&method,		// Value
		&stencilRate,	// ReadOnlyLookupValue
		&stencilIndex,	// ReadOnlyLookupValue
		&buildDefaultMesh,	// DestFinfo
		&resetStencil,	// DestFinfo
		// &nodeMeshing,	// SharedFinfo
		&entryFinfo,	// FieldElementFinfo
	};

	static Dinfo< short > dinfo;
	static Cinfo chemMeshCinfo (
		"ChemCompt",
		Neutral::initCinfo(),
		chemMeshFinfos,
		sizeof( chemMeshFinfos ) / sizeof ( Finfo* ),
		&dinfo
	);

	return &chemMeshCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* chemMeshCinfo = ChemCompt::initCinfo();

ChemCompt::ChemCompt()
	: 
		volume_( 1.0 ),
		entry_( this )
{
	;
}

ChemCompt::~ChemCompt()
{ 
		/*
	for ( unsigned int i = 0; i < stencil_.size(); ++i ) {
		if ( stencil_[i] )
			delete stencil_[i];
	}
	*/
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void ChemCompt::buildDefaultMesh( const Eref& e,
	double volume, unsigned int numEntries )
{
	this->innerBuildDefaultMesh( e, volume, numEntries );
}

/*
void ChemCompt::handleRequestMeshStats( const Eref& e )
{
	// Pass it down to derived classes along with the SrcFinfo
	innerHandleRequestMeshStats( e, meshStats() );
}
*/

void ChemCompt::handleNodeInfo( const Eref& e,
	unsigned int numNodes, unsigned int numThreads )
{
	// Pass it down to derived classes along with the SrcFinfo
	innerHandleNodeInfo( e, numNodes, numThreads );
}

void ChemCompt::resetStencil()
{
	this->innerResetStencil();
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

double ChemCompt::getEntireVolume( const Eref& e ) const
{
	return volume_;
}

void ChemCompt::setEntireVolume( const Eref& e, double volume )
{
	buildDefaultMesh( e, volume, getNumEntries() );
}

double ChemCompt::getVoxelVolume( const Eref& e, unsigned int dataIndex ) const
{
	return this->getMeshEntryVolume( dataIndex );
}


unsigned int ChemCompt::getDimensions() const
{
	return this->innerGetDimensions();
}

string ChemCompt::getMethod() const
{
	return method_;
}

void ChemCompt::setMethod( string method )
{
	method_ = method;
}

vector< double > ChemCompt::getStencilRate( unsigned int row ) const
{
	return this->innerGetStencilRate( row );
}

vector< unsigned int > ChemCompt::getStencilIndex( unsigned int row ) const
{
	return this->getNeighbors( row );
}


//////////////////////////////////////////////////////////////
// Element Field Definitions
//////////////////////////////////////////////////////////////

MeshEntry* ChemCompt::lookupEntry( unsigned int index )
{
	return &entry_;
}

void ChemCompt::setNumEntries( unsigned int num )
{
	this->innerSetNumEntries( num );
	// cout << "Warning: ChemCompt::setNumEntries: No effect. Use subclass-specific functions\nto build or resize mesh.\n";
}

unsigned int ChemCompt::getNumEntries() const
{
	return this->innerGetNumEntries();
}

//////////////////////////////////////////////////////////////
// Element Field Definitions for boundary
//////////////////////////////////////////////////////////////

/*
Boundary* ChemCompt::lookupBoundary( unsigned int index )
{
	if ( index < boundaries_.size() )
		return &( boundaries_[index] );
	cout << "Error: ChemCompt::lookupBoundary: Index " << index << 
		" >= vector size " << boundaries_.size() << endl;
	return 0;
}

void ChemCompt::setNumBoundary( unsigned int num )
{
	assert( num < 1000 ); // Pretty unlikely upper limit
	boundaries_.resize( num );
}

unsigned int ChemCompt::getNumBoundary() const
{
	return boundaries_.size();
}
*/

//////////////////////////////////////////////////////////////
// Build the junction between this and another ChemCompt.
// This one function does the work for both meshes.
//////////////////////////////////////////////////////////////
void ChemCompt::buildJunction( ChemCompt* other, vector< VoxelJunction >& ret)
{
	matchMeshEntries( other, ret );
	extendStencil( other, ret );
	/*
	 * No longer having diffusion to abutting voxels in the follower
	 * compartment.
	 *
	flipRet( ret );
	other->extendStencil( this, ret );
	flipRet( ret );
	*/
}

void ChemCompt::flipRet( vector< VoxelJunction >& ret ) const
{
   vector< VoxelJunction >::iterator i;
   for ( i = ret.begin(); i != ret.end(); ++i ) {
		  unsigned int temp = i->first;
		  i->first = i->second;
		  i->second = temp;
   }
}

//////////////////////////////////////////////////////////////
// Orchestrate diffusion calculations in Stoich. This simply updates
// the flux terms (du/dt due to diffusion). Virtual func, has to be
// defined for every Mesh class if it differs from below.
// Called from the MeshEntry.
//////////////////////////////////////////////////////////////

/*
void ChemCompt::lookupStoich( ObjId me ) const
{
	ChemCompt* cm = reinterpret_cast< ChemCompt* >( me.data() );
	assert( cm == this );
	vector< Id > stoichVec;
	unsigned int num = 
		me.element()->getNeighbours( stoichVec, meshSplitOut());
	if ( num == 1 ) // The solver has been created
		cm->stoich_ = stoichVec[0];
}
*/

/*
void ChemCompt::updateDiffusion( unsigned int meshIndex ) const
{
	// Later we'll have provision for multiple stoich targets.
	if ( stoich_ != Id() ) {
		Stoich* s = reinterpret_cast< Stoich* >( stoich_.eref().data() );
		s->updateDiffusion( meshIndex, stencil_ );
	}
}
*/

////////////////////////////////////////////////////////////////////////
// Utility function

double ChemCompt::distance( double x, double y, double z ) 
{
	return sqrt( x * x + y * y + z * z );
}
