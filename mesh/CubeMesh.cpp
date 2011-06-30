/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Boundary.h"
#include "MeshEntry.h"
#include "ChemMesh.h"
#include "CubeMesh.h"

static const unsigned int EMPTY = ~0;

const Cinfo* CubeMesh::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< CubeMesh, double > x0(
			"x0",
			"X coord of one end",
			&CubeMesh::setY0,
			&CubeMesh::getY0
		);
		static ValueFinfo< CubeMesh, double > y0(
			"y0",
			"Y coord of one end",
			&CubeMesh::setY0,
			&CubeMesh::getY0
		);
		static ValueFinfo< CubeMesh, double > z0(
			"z0",
			"Z coord of one end",
			&CubeMesh::setZ0,
			&CubeMesh::getZ0
		);
		static ValueFinfo< CubeMesh, double > x1(
			"x1",
			"X coord of other end",
			&CubeMesh::setX1,
			&CubeMesh::getX1
		);
		static ValueFinfo< CubeMesh, double > y1(
			"y1",
			"Y coord of other end",
			&CubeMesh::setY1,
			&CubeMesh::getY1
		);
		static ValueFinfo< CubeMesh, double > z1(
			"z1",
			"Z coord of other end",
			&CubeMesh::setZ1,
			&CubeMesh::getZ1
		);

		static ValueFinfo< CubeMesh, double > dx(
			"dx",
			"X size for mesh",
			&CubeMesh::setDx,
			&CubeMesh::getDx
		);
		static ValueFinfo< CubeMesh, double > dy(
			"dy",
			"Y size for mesh",
			&CubeMesh::setDy,
			&CubeMesh::getDy
		);
		static ValueFinfo< CubeMesh, double > dz(
			"dz",
			"Z size for mesh",
			&CubeMesh::setDz,
			&CubeMesh::getDz
		);

		static ValueFinfo< CubeMesh, vector< double > > coords(
			"coords",
			"Set all the coords of the cuboid at once. Order is:"
			"x0 y0 z0   x1 y1 z1   dx dy dz",
			&CubeMesh::setCoords,
			&CubeMesh::getCoords
		);

		static ValueFinfo< CubeMesh, vector< unsigned int > > meshToSpace(
			"meshToSpace",
			"Array in which each mesh entry stores spatial (cubic) index",
			&CubeMesh::setMeshToSpace,
			&CubeMesh::getMeshToSpace
		);

		static ValueFinfo< CubeMesh, vector< unsigned int > > spaceToMesh(
			"spaceToMesh",
			"Array in which each space index (obtained by linearizing "
			"the xyz coords) specifies which meshIndex is present."
			"In many cases the index will store the EMPTY flag if there is"
			"no mesh entry at that spatial location",
			&CubeMesh::setSpaceToMesh,
			&CubeMesh::getSpaceToMesh
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo buildMesh( "buildMesh",
			"Build cubical mesh for geom surface specified by Id, using"
			"specified x y z coords as an inside point in mesh",
			new OpFunc4< CubeMesh, Id, double, double, double >(
				&CubeMesh::buildMesh )
		);

		//////////////////////////////////////////////////////////////
		// Field Elements
		//////////////////////////////////////////////////////////////

	static Finfo* cubeMeshFinfos[] = {
		&x0,			// Value
		&y0,			// Value
		&z0,			// Value
		&x1,			// Value
		&y1,			// Value
		&z1,			// Value
		&dx,			// Value
		&dy,			// Value
		&dz,			// Value
		&coords,		// Value
		&meshToSpace,	// Value
		&spaceToMesh,	// Value
	};

	static Cinfo cubeMeshCinfo (
		"CubeMesh",
		ChemMesh::initCinfo(),
		cubeMeshFinfos,
		sizeof( cubeMeshFinfos ) / sizeof ( Finfo* ),
		new Dinfo< CubeMesh >()
	);

	return &cubeMeshCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* cubeMeshCinfo = CubeMesh::initCinfo();

//////////////////////////////////////////////////////////////////
// Class stuff.
//////////////////////////////////////////////////////////////////
CubeMesh::CubeMesh()
	:
		isToroid_( 0 ),
		x0_( 0.0 ),
		y0_( 0.0 ),
		z0_( 0.0 ),
		x1_( 1.0 ),
		y1_( 1.0 ),
		z1_( 1.0 ),
		dx_( 1.0 ),
		dy_( 1.0 ),
		dz_( 1.0 ),
		nx_( 1 ),
		ny_( 1 ),
		nz_( 1 ),
		m2s_( 1, 0 ),
		s2m_( 1, 0 )
{
	;
}

CubeMesh::~CubeMesh()
{
	;
}

//////////////////////////////////////////////////////////////////
// Field assignment stuff
//////////////////////////////////////////////////////////////////

/**
 * This assumes that dx, dy, dz are the quantities to preserve, over 
 * numEntries.
 * So when the compartment changes size, so does numEntries. dx, dy, dz
 * do not change, some of the sub-cuboids will partially be outside.
 */
void CubeMesh::updateCoords()
{
	nx_ = ceil( (x1_ - x0_) / dx_ );
	ny_ = ceil( (y1_ - y0_) / dy_ );
	nz_ = ceil( (z1_ - z0_) / dz_ );

	if ( nx_ == 0 ) nx_ = 1;
	if ( ny_ == 0 ) ny_ = 1;
	if ( nz_ == 0 ) nz_ = 1;

	/// Temporarily fill out the whole cube.
	unsigned int size = nx_ * ny_ * nz_;
	m2s_.resize( size );
	s2m_.resize( size );
	for ( unsigned int i = 0; i < size; ++i )
		m2s_[i] = s2m_[i] = i;
}

void CubeMesh::setX0( double v )
{
	x0_ = v;
	updateCoords();
}

double CubeMesh::getX0() const
{
	return x0_;
}

void CubeMesh::setY0( double v )
{
	y0_ = v;
	updateCoords();
}

double CubeMesh::getY0() const
{
	return y0_;
}

void CubeMesh::setZ0( double v )
{
	z0_ = v;
	updateCoords();
}

double CubeMesh::getZ0() const
{
	return z0_;
}

void CubeMesh::setX1( double v )
{
	x1_ = v;
	updateCoords();
}

double CubeMesh::getX1() const
{
	return x1_;
}

void CubeMesh::setY1( double v )
{
	y1_ = v;
	updateCoords();
}

double CubeMesh::getY1() const
{
	return y1_;
}

void CubeMesh::setZ1( double v )
{
	z1_ = v;
	updateCoords();
}

double CubeMesh::getZ1() const
{
	return z1_;
}

void CubeMesh::setDx( double v )
{
	dx_ = v;
	updateCoords();
}

double CubeMesh::getDx() const
{
	return dx_;
}


void CubeMesh::setDy( double v )
{
	dy_ = v;
	updateCoords();
}

double CubeMesh::getDy() const
{
	return dy_;
}


void CubeMesh::setDz( double v )
{
	dz_ = v;
	updateCoords();
}

double CubeMesh::getDz() const
{
	return dz_;
}

void CubeMesh::setCoords( vector< double > v )
{
	if ( v.size() < 9 ) {
		cout << "CubeMesh::setCoords: Warning: size of argument vec should be >= 9, was " << v.size() << endl;
	}
	x0_ = v[0];
	y0_ = v[1];
	z0_ = v[2];

	x1_ = v[3];
	y1_ = v[4];
	z1_ = v[5];

	dx_ = v[6];
	dy_ = v[7];
	dz_ = v[8];

	updateCoords();
}

vector< double > CubeMesh::getCoords() const
{
	vector< double > ret( 9 );

	ret[0] = x0_;
	ret[1] = y0_;
	ret[2] = z0_;

	ret[3] = x1_;
	ret[4] = y1_;
	ret[5] = z1_;

	ret[6] = dx_;
	ret[7] = dy_;
	ret[8] = dz_;

	return ret;
}

void CubeMesh::setMeshToSpace( vector< unsigned int > v )
{
	m2s_ = v;
}

vector< unsigned int > CubeMesh::getMeshToSpace() const
{
	return m2s_;
}

void CubeMesh::setSpaceToMesh( vector< unsigned int > v )
{
	s2m_ = v;
}

vector< unsigned int > CubeMesh::getSpaceToMesh() const
{
	return s2m_;
}

unsigned int CubeMesh::innerGetDimensions() const
{
	return 3;
}


//////////////////////////////////////////////////////////////////
// DestFinfos
//////////////////////////////////////////////////////////////////

void CubeMesh::buildMesh( Id geom, double x, double y, double z )
{
	;
}

//////////////////////////////////////////////////////////////////
// FieldElement assignment stuff for MeshEntries
//////////////////////////////////////////////////////////////////

/// Virtual function to return MeshType of specified entry.
unsigned int CubeMesh::getMeshType( unsigned int fid ) const
{
	return CUBOID;
}

/// Virtual function to return dimensions of specified entry.
unsigned int CubeMesh::getMeshDimensions( unsigned int fid ) const
{
	return 3;
}

/// Virtual function to return volume of mesh Entry.
double CubeMesh::getMeshEntrySize( unsigned int fid ) const
{
	return dx_ * dy_ * dz_;
}

/// Virtual function to return coords of mesh Entry.
/// For Cuboid mesh, coords are x1y1z1 x2y2z2
vector< double > CubeMesh::getCoordinates( unsigned int fid ) const
{
	assert( fid < m2s_.size() );
	unsigned int spaceIndex = m2s_[fid];

	unsigned int ix = spaceIndex % nx_;
	unsigned int iy = (spaceIndex / nx_) % ny_;
	unsigned int iz = (spaceIndex / ( nx_ * ny_ )) % nz_;

	vector< double > ret( 6 );
	ret[0] = x0_ + ix * dx_;
	ret[1] = y0_ + iy * dy_;
	ret[2] = z0_ + iz * dz_;

	ret[3] = x0_ + ix * dx_ + dx_;
	ret[4] = y0_ + iy * dy_ + dx_;
	ret[5] = z0_ + iz * dz_ + dx_;

	return ret;
}

unsigned int CubeMesh::neighbor( unsigned int spaceIndex, 
	int dx, int dy, int dz ) const
{
	int ix = spaceIndex % nx_;
	int iy = (spaceIndex / nx_) % ny_;
	int iz = (spaceIndex / ( nx_ * ny_ )) % nz_;

	ix += dx;
	iy += dy;
	iz += dz;

	if ( ix < 0 || ix >= static_cast< int >( nx_ ) )
		return EMPTY;
	if ( iy < 0 || iy >= static_cast< int >( ny_ ) )
		return EMPTY;
	if ( iz < 0 || iz >= static_cast< int >( nz_ ) )
		return EMPTY;

	unsigned int nIndex = ( ( iz * ny_ ) + iy ) * nx_ + ix;

	return s2m_[nIndex];
}

/// Virtual function to return info on Entries connected to this one
vector< unsigned int > CubeMesh::getNeighbors( unsigned int fid ) const
{
	assert( fid < m2s_.size() );

	vector< unsigned int > ret;
	unsigned int spaceIndex = m2s_[fid];

	unsigned int nIndex = neighbor( spaceIndex, 0, 0, 1 );
	if ( nIndex != EMPTY ) 
		ret.push_back( nIndex );

	nIndex = neighbor( spaceIndex, 0, 0, -1 );
	if ( nIndex != EMPTY ) 
		ret.push_back( nIndex );

	nIndex = neighbor( spaceIndex, 0, 1, 0 );
	if ( nIndex != EMPTY ) 
		ret.push_back( nIndex );

	nIndex = neighbor( spaceIndex, 0, -1, 0 );
	if ( nIndex != EMPTY ) 
		ret.push_back( nIndex );

	nIndex = neighbor( spaceIndex, 1, 0, 0 );
	if ( nIndex != EMPTY ) 
		ret.push_back( nIndex );

	nIndex = neighbor( spaceIndex, -1, 0, 0 );
	if ( nIndex != EMPTY ) 
		ret.push_back( nIndex );

	return ret;	
}

/// Virtual function to return diffusion X-section area for each neighbor
vector< double > CubeMesh::getDiffusionArea( unsigned int fid ) const
{
	assert( fid < m2s_.size() );

	vector< double > ret;
	unsigned int spaceIndex = m2s_[fid];

	unsigned int nIndex = neighbor( spaceIndex, 0, 0, 1 );
	if ( nIndex != EMPTY ) 
		ret.push_back( dy_ * dz_ );

	nIndex = neighbor( spaceIndex, 0, 0, -1 );
	if ( nIndex != EMPTY ) 
		ret.push_back( dy_ * dz_ );

	nIndex = neighbor( spaceIndex, 0, 1, 0 );
	if ( nIndex != EMPTY ) 
		ret.push_back( dz_ * dx_ );

	nIndex = neighbor( spaceIndex, 0, -1, 0 );
	if ( nIndex != EMPTY ) 
		ret.push_back( dz_ * dx_ );

	nIndex = neighbor( spaceIndex, 1, 0, 0 );
	if ( nIndex != EMPTY ) 
		ret.push_back( dx_ * dy_ );

	nIndex = neighbor( spaceIndex, -1, 0, 0 );
	if ( nIndex != EMPTY ) 
		ret.push_back( dx_ * dy_ );

	return ret;
}

/// Virtual function to return scale factor for diffusion. 1 here.
vector< double > CubeMesh::getDiffusionScaling( unsigned int fid ) const
{
	return vector< double >( 6, 1.0 );
}

//////////////////////////////////////////////////////////////////

/**
 * Inherited virtual func. Returns number of MeshEntry in array
 */
unsigned int CubeMesh::innerGetNumEntries() const
{
	return m2s_.size();
}

/**
 * Inherited virtual func. Assigns number of MeshEntries.
 */
void CubeMesh::innerSetNumEntries( unsigned int n )
{
	cout << "Warning: CubeMesh::innerSetNumEntries is readonly.\n";
}
