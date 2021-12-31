/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "../basecode/header.h"
#include "../basecode/SparseMatrix.h"
#include "../basecode/ElementValueFinfo.h"
#include "Boundary.h"
#include "MeshEntry.h"
#include "ChemCompt.h"
#include "MeshCompt.h"
#include "PresynMesh.h"
#include "../utility/Vec.h"

const double DEFAULT_BOUTON_VOLUME = 5e-20;
/// We assume the bouton to be a hemisphere.

const Cinfo* PresynMesh::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ReadOnlyValueFinfo< PresynMesh, vector< ObjId > > postsynCompts(
			"postsynCompts",
			"Return list of all postsyn compts, one per bouton. Note that "
			"there may be repeats since a compt may receive multiple "
			"synaptic inputs.",
			&PresynMesh::getPostsynCompts
		);

		static ReadOnlyValueFinfo< PresynMesh, vector< Id > > elecComptMap(
			"elecComptMap",
			"Return list of Ids of all postsyn compts, one per bouton. "
			"Provided to match with the other Mesh classes. Note that "
			"there may be repeats since a compt may receive multiple "
			"synaptic inputs. Identical to elecComptList",
			&PresynMesh::getElecComptMap
		);

		static ReadOnlyValueFinfo< PresynMesh, vector< Id > > elecComptList(
			"elecComptList",
			"Return list of Ids of all postsyn compts, one per bouton. "
			"Provided to match with the other Mesh classes. Note that "
			"there may be repeats since a compt may receive multiple "
			"synaptic inputs. Identical to elecComptMap.",
			&PresynMesh::getElecComptMap
		);

		static ReadOnlyValueFinfo< PresynMesh, vector< unsigned int > > startVoxelInCompt(
			"startVoxelInCompt",
			"Index of starting voxel that maps to each electrical compartment.",
			&PresynMesh::getStartVoxelInCompt
		);

		static ReadOnlyValueFinfo< PresynMesh, vector< unsigned int > > endVoxelInCompt(
			"endVoxelInCompt",
			"Index of end voxel that maps to each electrical compartment, "
		    "using C++ indexing. So if there was 1 voxel only it would "
			"return 1+startVoxelInCompt",
			&PresynMesh::getEndVoxelInCompt
		);

		static ReadOnlyValueFinfo< PresynMesh, double > boutonSpacing(
			"boutonSpacing",
			"Spacing in metres between boutons on a dendrite. "
			"Assigned when creating boutons on a dendrite. "
			"Not defined if the boutons are on spines, which are one-to-one.",
			&PresynMesh::getBoutonSpacing
		);

		static ReadOnlyValueFinfo< PresynMesh, unsigned int > numBoutons(
			"numBoutons",
			"Total number of boutons defined in this mesh.",
			&PresynMesh::getNumBoutons
		);

		static ReadOnlyValueFinfo< PresynMesh, bool > isOnSpines(
			"isOnSpines",
			"Flag to report if PresynMesh is connected to spines,"
			"in which case the connections are one-to-one. "
			"If false, then the mesh is conneced to the dendrite. ",
			&PresynMesh::isOnSpines
		);

    	static ReadOnlyLookupValueFinfo< PresynMesh, ObjId, int >
    		startBoutonIndexFromCompartment(
        	"startBoutonIndexFromCompartment",
        	"Look up index of first bouton on specified compartment.",
        	&PresynMesh::getStartBoutonIndexFromCompartment
    	);

    	static ReadOnlyLookupValueFinfo< PresynMesh, ObjId, int >
    		numBoutonsOnCompartment(
        	"numBoutonsOnCompartment",
        	"Number of boutons on specified electrical compartment.",
        	&PresynMesh::getNumBoutonsOnCompartment
    	);


		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		
    	static DestFinfo buildOnSpineHeads( "buildOnSpineHeads",
			"Assigns list of dendritic spine heads to which the boutons "
			"should connect. Sets isOnSpines to true.",
			new OpFunc1< PresynMesh, vector< ObjId > >(
				&PresynMesh::buildOnSpineHeads ) );
		
    	static DestFinfo buildOnDendrites( "buildOnDendrites",
			"Assigns list of dendritic compartments to which the boutons "
			"should connect. Also assigns spacing. "
			"Sets isOnSpines to false.",
			new OpFunc2< PresynMesh, vector< ObjId >, double >(
				&PresynMesh::buildOnDendrites ) );

    	static DestFinfo setRadiusStats( "setRadiusStats",
			"Assigns radius to the presyn boutons. SI units. "
			"Assumes bouton is a hemisphere. Vol = 2/3 PI R^3 "
			"If isOnSpines is true, then it assigns as a scale factor to "
			"the postsynaptic radius. "
			"If isOnSpines is false, then it assigns absolute radius. "
			"Arguments are radius, standard_deviation. ",
			new OpFunc2< PresynMesh, double, double >(
				&PresynMesh::setRadiusStats ) );

		//////////////////////////////////////////////////////////////
		// Field Elements
		//////////////////////////////////////////////////////////////

	static Finfo* presynMeshFinfos[] = {
		&postsynCompts,			// ReadOnlyValue
		&elecComptMap,			// ReadOnlyValue
		&elecComptList,			// ReadOnlyValue
		&startVoxelInCompt,		// ReadOnlyValue
		&endVoxelInCompt,		// ReadOnlyValue
		&boutonSpacing,			// ReadOnlyValue
		&numBoutons,			// ReadOnlyValue
		&isOnSpines,			// ReadOnlyValue
		&startBoutonIndexFromCompartment,		// ReadOnlyValueLookup
		&numBoutonsOnCompartment,				// ReadOnlyValueLookup
		&buildOnSpineHeads,			// DestFinfo
		&buildOnDendrites,			// DestFinfo
		&setRadiusStats,			// DestFinfo
	};

	static Dinfo< PresynMesh > dinfo;
	static Cinfo presynMeshCinfo (
		"PresynMesh",
		ChemCompt::initCinfo(),
		presynMeshFinfos,
		sizeof( presynMeshFinfos ) / sizeof ( Finfo* ),
		&dinfo
	);

	return &presynMeshCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* presynMeshCinfo = PresynMesh::initCinfo();

//////////////////////////////////////////////////////////////////
// Class stuff.
//////////////////////////////////////////////////////////////////

Bouton::Bouton()
		:
			x_( 0.0 ),
			y_( 0.0 ),
			z_( 0.0 ),
			vx_( 0.0 ),
			vy_( 0.0 ),
			vz_( 0.0 ),
			volume_( DEFAULT_BOUTON_VOLUME )
{;}

PresynMesh::PresynMesh()
	:
		isOnSpines_( false ),
		spacing_( 100e-6 )
{
	;
}

PresynMesh::~PresynMesh()
{
	;
}

//////////////////////////////////////////////////////////////////
// Field assignment stuff
//////////////////////////////////////////////////////////////////

vector< ObjId > PresynMesh::getPostsynCompts() const
{
	vector< ObjId > ret( boutons_.size() );

	for( unsigned int i = 0; i < boutons_.size(); i++ )
		ret[i] = boutons_[i].postsynCompt_;

	return ret;
}

vector< Id > PresynMesh::getElecComptMap() const
{
	vector< Id > ret( boutons_.size() );

	for( unsigned int i = 0; i < boutons_.size(); i++ )
		ret[i] = boutons_[i].postsynCompt_;

	return ret;
}

vector< unsigned int > PresynMesh::getStartVoxelInCompt() const
{
	vector< unsigned int > ret( boutons_.size() );
	auto lastCompt = ObjId();
	unsigned int lastIndex = 0;

	for( unsigned int i = 0; i < boutons_.size(); i++ ) {
		if ( lastCompt != boutons_[i].postsynCompt_ ) {
			lastCompt = boutons_[i].postsynCompt_;
			lastIndex = i;
		}
		ret[i] = lastIndex;
	}

	return ret;
}

vector< unsigned int > PresynMesh::getEndVoxelInCompt() const
{
	vector< unsigned int > ret( boutons_.size() );
	auto lastCompt = ObjId();
	unsigned int lastIndex = 0;

	for( unsigned int i = 0; i < boutons_.size(); i++ ) {
		if ( lastCompt != boutons_[i].postsynCompt_ ) {
			lastCompt = boutons_[i].postsynCompt_;
			lastIndex = i;
		}
		if ( i > 0 )
			ret[i-1] = lastIndex;
	}
	ret[ret.size() - 1] = boutons_.size();

	return ret;
}

double PresynMesh::getBoutonSpacing() const
{
	return spacing_;
}

unsigned int PresynMesh::getNumBoutons() const
{
	return boutons_.size();
}

bool PresynMesh::isOnSpines() const
{
	return isOnSpines_;
}

int PresynMesh::getStartBoutonIndexFromCompartment( ObjId c ) const
{
	for( unsigned int i = 0; i < boutons_.size(); i++ ) {
		if (boutons_[i].postsynCompt_ == c)
			return i;
	}
	return -1;
}

int PresynMesh::getNumBoutonsOnCompartment( ObjId c ) const
{
	int ret = 0;
	for( const Bouton& b : boutons_ )
		ret += b.postsynCompt_ == c;
	return ret;
}

//////////////////////////////////////////////////////////////////
// FieldElement assignment stuff for MeshEntries
//////////////////////////////////////////////////////////////////

/// Virtual function to return MeshType of specified entry.
unsigned int PresynMesh::getMeshType( unsigned int fid ) const
{
	return PRESYN;
}

/// Virtual function to return dimensions of specified entry.
unsigned int PresynMesh::getMeshDimensions( unsigned int fid ) const
{
	return 3;
}

/// Virtual function to return # of spatial dimensions of mesh
unsigned int PresynMesh::innerGetDimensions() const
{
	return 3;
}
/// Virtual function to return volume of mesh Entry.
double PresynMesh::getMeshEntryVolume( unsigned int fid ) const
{
	if ( fid < boutons_.size() )
		return( boutons_[fid].volume_ );
	return 0.0;
}

/// Virtual function to return coords of mesh Entry.
/// Assume that it is a hemisphere with the flat face toward postsyn.
/// midpoint, direction, diameter.
vector< double > PresynMesh::getCoordinates( unsigned int fid ) const
{
	assert( fid < boutons_.size() );
	const Bouton& b = boutons_[fid];
	vector< double > ret(7);
	ret[0] = b.x_;	// Midpoint
	ret[1] = b.y_;
	ret[2] = b.z_;
	ret[3] = b.vx_; // direction
	ret[4] = b.vy_;
	ret[5] = b.vz_;
	ret[6] = 2.0 * pow( b.volume_ * 1.5 / PI, 1.0/3.0 );

	return ret;
}

/// Virtual function to return diffusion X-section area for each neighbor
/// There is no diffusion so it is zero.
vector< double > PresynMesh::getDiffusionArea( unsigned int fid ) const
{
	return vector< double >( boutons_.size(), 0.0 );
}

/// Virtual function to return scale factor for diffusion.
/// There is no diffusion so it is zero.
vector< double > PresynMesh::getDiffusionScaling( unsigned int fid ) const
{
	return vector< double >( boutons_.size(), 0.0 );
}

/// Virtual function to return volume of mesh Entry.
double PresynMesh::extendedMeshEntryVolume( unsigned int fid ) const
{
	assert( fid < boutons_.size() );
	return( boutons_[fid].volume_ );
}

//////////////////////////////////////////////////////////////////
// Dest funcs
//////////////////////////////////////////////////////////////////

/// More inherited virtual funcs: request comes in for mesh stats
void PresynMesh::innerHandleRequestMeshStats( const Eref& e,
		const SrcFinfo2< unsigned int, vector< double > >* meshStatsFinfo
	)
{
	double aveVol = vGetEntireVolume();
	if ( boutons_.size() > 0 )
		aveVol /= boutons_.size();
	vector< double > ret( 1, aveVol );
	meshStatsFinfo->send( e, 1, ret );
}

void PresynMesh::innerHandleNodeInfo(
			const Eref& e,
			unsigned int numNodes, unsigned int numThreads )
{
}

/// Requires that all objects in v are the head compartments of spines.
void PresynMesh::buildOnSpineHeads( vector< ObjId > v )
{
	isOnSpines_ = true;
	spacing_ = 0; // Indicate that the spacing term isn't being used.
	boutons_.resize( v.size() );
	for (unsigned int i = 0; i < v.size(); ++i ) {
		if ( !v[i].element()->cinfo()->isA( "CompartmentBase" ) )  {
			cout << "Error: Attempt to assign PresynMesh to a non_compartment: " << v[i].id.path() << endl;
			assert( 0 );
		}

		Bouton& b = boutons_[i];
		b.postsynCompt_ = v[i];
		double length = Field< double >::get( v[i], "length" );
		double diameter = Field< double >::get( v[i], "diameter" );
		// Set up coords on end of compt
		b.x_ = Field< double >::get( v[i], "x" );
		b.y_ = Field< double >::get( v[i], "y" );
		b.z_ = Field< double >::get( v[i], "z" );
		// Define unit vector of direction of presyn from x to x0. (toward)
		b.vx_ = (Field< double >::get( v[i], "x0" ) - b.x_) / length;
		b.vy_ = (Field< double >::get( v[i], "y0" ) - b.y_) / length;
		b.vz_ = (Field< double >::get( v[i], "z0" ) - b.z_) / length;
		// Shift bouton coords 20 nanometres away from spine end.
		b.x_ -= b.vx_ * 20e-9;
		b.y_ -= b.vy_ * 20e-9;
		b.z_ -= b.vz_ * 20e-9;
		b.volume_ = length * diameter * diameter * 0.25 * PI;
	}
}

void PresynMesh::buildOnDendrites( vector< ObjId > compts, double spacing )
{
	isOnSpines_ = false;
	spacing_ = spacing;
	boutons_.clear();
	for ( ObjId& v : compts ) {
		if ( !v.element()->cinfo()->isA( "CompartmentBase" ) )  {
			cout << "Error: Attempt to assign PresynMesh to a non_compartment: " << v.id.path() << endl;
			assert( 0 );
		}
		double length = Field< double >::get( v, "length" );
		double diameter = Field< double >::get( v, "diameter" );
		double x = Field< double >::get( v, "x" );
		double y = Field< double >::get( v, "y" );
		double z = Field< double >::get( v, "z" );
		double x0 = Field< double >::get( v, "x0" );
		double y0 = Field< double >::get( v, "y0" );
		double z0 = Field< double >::get( v, "z0" );

		Vec v0( x0, y0, z0 );
		Vec v1( x, y, z );
		Vec vAxial( v1 - v0 );
		Vec vRadial1( 0, 0, 0);
		Vec vRadial2( 0, 0, 0);
		vAxial.orthogonalAxes( vRadial1, vRadial2 );
		
		for (double r = 0.5 * spacing; r < length; r += spacing) {
			Bouton b;
			b.postsynCompt_ = v;
			b.vx_ = -vRadial1.a0();	// Components of the radial unit vector
			b.vy_ = -vRadial1.a1();
			b.vz_ = -vRadial1.a2();
			b.x_ = x0 + (r/length) * (x - x0) - (diameter + 20e-9) * b.vx_;
			b.y_ = y0 + (r/length) * (y - y0) - (diameter + 20e-9) * b.vy_;
			b.z_ = z0 + (r/length) * (z - z0) - (diameter + 20e-9) * b.vz_;
			b.volume_ = DEFAULT_BOUTON_VOLUME;
			boutons_.push_back( b );
		}
	}
}

/**
 * setRadiusStats is meant to be called before the presynMesh is populated
 * with the chemical system, but after the boutons are created.
 */
void PresynMesh::setRadiusStats( double r, double sdev )
{
	// As a placeholder, just go through assigning vols without the sdev.
	if ( isOnSpines_ ) {
		for ( Bouton& b : boutons_ )
			b.volume_ *= r*r*r; // Scale relative to postsyn.
	} else {
		double v = r*r*r*PI*2.0/3.0; // Hemisphere
		for ( Bouton& b : boutons_ )
			b.volume_ = v; // Absolute assignment.
	}
}

//////////////////////////////////////////////////////////////////

/**
 * Inherited virtual func. Returns number of MeshEntry in array
 */
unsigned int PresynMesh::innerGetNumEntries() const
{
	return boutons_.size();
}

/**
 * Inherited virtual func. 
 */
void PresynMesh::innerSetNumEntries( unsigned int n )
{
	boutons_.resize( n );
}


void PresynMesh::innerBuildDefaultMesh( const Eref& e,
	double volume, unsigned int numEntries )
{
	boutons_.resize( numEntries );
	for ( Bouton& b : boutons_ ) {
		b.volume_ = volume;
	}
}

/**
 * This means the sibling voxel immediately adjacent to it, not the
 * voxel surrounding it. Should we wish to do diffusion we would need
 * to use the parent voxel of the surround voxel. Otherewise use
 * EMTPY_VOXEL == -1U.
 */
vector< unsigned int > PresynMesh::getParentVoxel() const
{
	vector< unsigned int > ret( boutons_.size(), -1U);
	return ret;
}

const vector< double >& PresynMesh::vGetVoxelVolume() const
{
	static vector< double > ret;
	ret.clear();
	for ( auto i : boutons_ )
		ret.push_back( i.volume_ );
	return ret;
}

const vector< double >& PresynMesh::vGetVoxelMidpoint() const
{
	static vector< double > ret;
	ret.clear();
	for ( auto i : boutons_ ) ret.push_back( i.x_ );
	for ( auto i : boutons_ ) ret.push_back( i.y_ );
	for ( auto i : boutons_ ) ret.push_back( i.z_ );
	return ret;
}

const vector< double >& PresynMesh::getVoxelArea() const
{
	static vector< double > ret;
	ret = vGetVoxelVolume();
	for ( auto i = ret.begin(); i != ret.end(); ++i )
		*i = pow( *i, 2.0/3.0 );
	return ret;
}

const vector< double >& PresynMesh::getVoxelLength() const
{
	static vector< double > ret;
	ret = vGetVoxelVolume();
	for ( auto i = ret.begin(); i != ret.end(); ++i )
		*i = pow( *i, 1.0/3.0 );
	return ret;
}

double PresynMesh::vGetEntireVolume() const
{
	double vol = 0.0;
	auto vec = vGetVoxelVolume();
	for ( auto i : vec )
		vol += i;
	return vol;
}

bool PresynMesh::vSetVolumeNotRates( double volume )
{
	return true; // maybe should be false? Unsure
}

//////////////////////////////////////////////////////////////////
// Utility function for junctions
//////////////////////////////////////////////////////////////////

void PresynMesh::matchMeshEntries( const ChemCompt* other,
	   vector< VoxelJunction >& ret ) const
{
	ret.clear(); // No diffusion here.
}

// This function returns the distance to the nearest mesh entry, and
// passes back its index. 
double PresynMesh::nearest( double x, double y, double z,
				unsigned int& index ) const
{
	double distance = 1e19;
	index = 0;
	for ( unsigned int i = 0; i < boutons_.size(); i++ ) {
		const Bouton& b = boutons_[i];
		double r = (b.x_ - x) * (b.x_ - x) + (b.y_ - y) * (b.y_ - y ) + (b.z_ - z) * (b.z_ - z);
		if ( distance > r ) {
			index = i;
			distance = r;
		}
	}
	if (distance == 1e19)
		return -1;
	return distance;
}

// This function returns coords of the voxel at the specified index.
void PresynMesh::indexToSpace( unsigned int index,
				double &x, double &y, double &z ) const
{
	if ( index >= boutons_.size() )
		return;
	x = boutons_[index].x_;
	y = boutons_[index].y_;
	z = boutons_[index].z_;
}
