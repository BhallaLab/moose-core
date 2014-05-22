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

const Cinfo* ChemCompt::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ChemCompt, double > volume(
			"volume",
			"Volume of entire chemical domain."
			"Assigning this only works if the chemical compartment has"
			"only a single voxel. Otherwise ignored."
			"This function goes through all objects below this on the"
			"tree, and rescales their molecule #s and rates as per the"
			"volume change. This keeps concentration the same, and also"
			"maintains rates as expressed in volume units.",
			&ChemCompt::setEntireVolume,
			&ChemCompt::getEntireVolume
		);

		static ReadOnlyValueFinfo< ChemCompt, vector< double > > 
				voxelVolume(
			"voxelVolume",
			"Vector of volumes of each of the voxels.",
			&ChemCompt::getVoxelVolume
		);

		static ReadOnlyLookupElementValueFinfo< 
				ChemCompt, unsigned int, double > 
			oneVoxelVolume(
			"oneVoxelVolume",
			"Volume of specified voxel.",
			&ChemCompt::getOneVoxelVolume
		);

		static ReadOnlyValueFinfo< ChemCompt, unsigned int > numDimensions(
			"numDimensions",
			"Number of spatial dimensions of this compartment. Usually 3 or 2",
			&ChemCompt::getDimensions
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

		static DestFinfo setVolumeNotRates( "setVolumeNotRates",
			"Changes volume but does not notify any child objects."
			"Only works if the ChemCompt has just one voxel."
			"This function will invalidate any concentration term in"
			"the model. If you don't know why you would want to do this,"
			"then you shouldn't use this function.",
			new OpFunc1< ChemCompt, double >( 
				&ChemCompt::setVolumeNotRates )
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
		&oneVoxelVolume,	// ReadOnlyLookupValue
		&numDimensions,	// ReadOnlyValue
		&stencilRate,	// ReadOnlyLookupValue
		&stencilIndex,	// ReadOnlyLookupValue
		&buildDefaultMesh,	// DestFinfo
		&setVolumeNotRates,		// DestFinfo
		&resetStencil,	// DestFinfo
		// &nodeMeshing,	// SharedFinfo
		&entryFinfo,	// FieldElementFinfo
	};

	static string doc[] = {
		"Name", "ChemCompt",
		"Author", "Upi Bhalla",
		"Description", "Pure virtual base class for chemical compartments"

	};
	static Dinfo< short > dinfo;
	static Cinfo chemMeshCinfo (
		"ChemCompt",
		Neutral::initCinfo(),
		chemMeshFinfos,
		sizeof( chemMeshFinfos ) / sizeof ( Finfo* ),
		&dinfo,
		doc,
		sizeof(doc)/sizeof( string ),
		true // This IS a FieldElement, not be be created directly.
	);

	return &chemMeshCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* chemMeshCinfo = ChemCompt::initCinfo();

ChemCompt::ChemCompt()
	: 
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

void ChemCompt::setEntireVolume( const Eref& e, double volume )
{
	vector< double > childConcs;
	getChildConcs( e, childConcs );
	if ( vSetVolumeNotRates( volume ) ) {
		// buildDefaultMesh( e, volume, getNumEntries() );
		// vector< double >::const_iterator conc = childConcs.begin();
		setChildConcs( e, childConcs, 0 );
		// assert( conc == childConcs.end() );
	}
}

double ChemCompt::getEntireVolume( const Eref& e ) const
{
	return vGetEntireVolume();
}

void ChemCompt::getChildConcs( const Eref& e, vector< double >& childConcs )
	   	const
{
	vector< Id > kids;
	Neutral::children( e, kids );
	for ( vector < Id >::iterator i = kids.begin(); i != kids.end(); ++i )
	{
		if ( i->element()->cinfo()->isA( "PoolBase" ) ) {
			childConcs.push_back( Field< double >::get( *i, "conc" ) );
			childConcs.push_back( Field< double >::get( *i, "concInit" ) );
		} else if ( i->element()->cinfo()->isA( "ReacBase" ) ) {
			childConcs.push_back( Field< double >::get( *i, "Kf" ) );
			childConcs.push_back( Field< double >::get( *i, "Kb" ) );
		} else if ( i->element()->cinfo()->isA( "EnzBase" ) ) {
			childConcs.push_back( Field< double >::get( *i, "Km" ) );
		} else if ( i->element()->cinfo()->isA( "ChemCompt" ) ) {
			// Do NOT traverse into child ChemCompts, they look after their
			// own volumes.
			continue;
		}
		getChildConcs( i->eref(), childConcs );
	}
}

unsigned int ChemCompt::setChildConcs( const Eref& e, 
		const vector< double >& conc, unsigned int start ) const
{
	vector< Id > kids;
	Neutral::children( e, kids );
	for ( vector < Id >::iterator i = kids.begin(); i != kids.end(); ++i )
	{
		if ( i->element()->cinfo()->isA( "PoolBase" ) ) {
			Field< double >::set( *i, "conc", conc[ start++ ] );
			Field< double >::set( *i, "concInit", conc[start++] );
		} else if ( i->element()->cinfo()->isA( "ReacBase" ) ) {
			Field< double >::set( *i, "Kf", conc[ start++ ] );
			Field< double >::set( *i, "Kb", conc[ start++ ] );
		} else if ( i->element()->cinfo()->isA( "EnzBase" ) ) {
			Field< double >::set( *i, "Km", conc[ start++ ] );
		} else if ( i->element()->cinfo()->isA( "ChemCompt" ) ) {
			// Do NOT traverse into child ChemCompts, they look after their
			// own volumes.
			continue;
		}
		start = setChildConcs( i->eref(), conc, start );
	}
	return start;
}

vector< double > ChemCompt::getVoxelVolume() const
{
	return this->vGetVoxelVolume();
}

double ChemCompt::getOneVoxelVolume( const Eref& e, unsigned int dataIndex ) const
{
	return this->getMeshEntryVolume( dataIndex );
}


unsigned int ChemCompt::getDimensions() const
{
	return this->innerGetDimensions();
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
// Dest Definitions
//////////////////////////////////////////////////////////////

void ChemCompt::setVolumeNotRates( double volume )
{
	vSetVolumeNotRates( volume ); // Pass on to derived classes.
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

////////////////////////////////////////////////////////////////////////
// Utility function

double ChemCompt::distance( double x, double y, double z ) 
{
	return sqrt( x * x + y * y + z * z );
}
