/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "MeshEntry.h"
#include "ChemMesh.h"
#include "CylMesh.h"

//////////////////////////////////////////////////////////////////
// Class stuff.
//////////////////////////////////////////////////////////////////
CylMesh::CylMesh()
	:
		numEntries_( 1 ),
		useCaps_( 0 ),
		isToroid_( 0 ),
		r0_( 1.0e-6 ),
		x0_( 0.0 ),
		y0_( 0.0 ),
		z0_( 0.0 ),
		r1_( 1.0e-6 ),
		x1_( 10.0e-6 ),
		y1_( 10.0e-6 ),
		z1_( 10.0e-6 ),
		lambda_( 5.0e-6 ),
		rSlope_( 0.0 ),
		lenSlope_( 0.0 )
{
	;
}

CylMesh::~CylMesh()
{
	;
}

//////////////////////////////////////////////////////////////////
// Field assignment stuff
//////////////////////////////////////////////////////////////////

unsigned int CylMesh::innerGetDimensions() const
{
	return 3;
}

//////////////////////////////////////////////////////////////////
// FieldElement assignment stuff for MeshEntries
//////////////////////////////////////////////////////////////////

/// Virtual function to return MeshType of specified entry.
unsigned int CylMesh::getMeshType( unsigned int fid ) const
{
	if ( !isToroid_ && useCaps_ && ( fid == 0 || fid == numEntries_ - 1 ) )
		return SPHERE_SHELL_SEG;

	return CYL;
}

/// Virtual function to return dimensions of specified entry.
unsigned int CylMesh::getMeshDimensions( unsigned int fid ) const
{
	return 3;
}

/**
 * lambda = length constant for diffusive spread
 * len = length of each mesh entry
 * totLen = total length of cylinder
 * lambda = k / r^2
 * Each entry has the same number of lambdas, L = len/lambda.
 * Thinner entries have shorter lambda.
 * This gives a moderately nasty quadratic.
 * However, as len(i) is prop to lambda(i),
 * and lambda(i) is prop to r(i)^2
 * and the cyl-mesh is assumed a gently sloping cone
 * we get len(i) is prop to (r0 + slope.x)^2
 * and ignoring the 2nd-order term we have
 * len(i) is approx proportional to x position.
 *
 * dr/dx = (r1-r0)/len
 * ri = r0 + i * dr/dx
 * r(i+1)-ri = (r1-r0)/numEntries
 * dlen/dx = dr/dx * dlen/dr = ( (r1-r0)/len ) * 2r
 * To linearize, let 2r = r0 + r1.
 * so dlen/dx = ( (r1-r0)/len ) * ( r0 + r1 )
 * len(i) = len0 + i * dlen/dx
 * len0 = totLen/numEntries - ( numEntries/2 ) * dlen/dx 
 */

/// Virtual function to return volume of mesh Entry.
double CylMesh::getMeshEntrySize( unsigned int fid ) const
{
 	double len0 = totLen_/numEntries_ - ( numEntries_/2 ) * lenSlope_;
	double ri = r0_ + (fid + 0.5) * rSlope_;
	return (len0 + fid * lenSlope_) * ri * ri * PI;
}

/// Virtual function to return coords of mesh Entry.
/// For Cylindrical mesh, coords are x1y1z1 x2y2z2 r0 r1 phi0 phi1
vector< double > CylMesh::getCoordinates( unsigned int fid ) const
{
	vector< double > ret(10, 0.0);
	double frac = static_cast< double >( numEntries_ )/2.0;

	double axialStart = 
		fid * totLen_/numEntries_ + (fid - frac ) * lenSlope_;
	double axialEnd = 
		(fid + 1) * totLen_/numEntries_ + (fid - frac + 1.0) * lenSlope_;

	ret[0] = x0_ + (x1_ - x0_ ) * axialStart/totLen_;
	ret[1] = y0_ + (y1_ - y0_ ) * axialStart/totLen_;
	ret[2] = z0_ + (z1_ - z0_ ) * axialStart/totLen_;

	ret[3] = x0_ + (x1_ - x0_ ) * axialEnd/totLen_;
	ret[4] = y0_ + (y1_ - y0_ ) * axialEnd/totLen_;
	ret[5] = z0_ + (z1_ - z0_ ) * axialEnd/totLen_;

	ret[6] = r0_ + fid * rSlope_;
	ret[7] = r0_ + (fid + 1.0) * rSlope_;

	ret[8] = 0;
	ret[9] = 0;
	
	return ret;
}
/// Virtual function to return info on Entries connected to this one
vector< unsigned int > CylMesh::getNeighbors( unsigned int fid ) const
{
	if ( numEntries_ <= 1 )
		return vector< unsigned int >( 0 );
	
	if ( isToroid_ ) {
		vector< unsigned int > ret( 2, 0 );
		ret[0] = ( fid == 0 ) ? numEntries_ - 1 : fid - 1;
		ret[1] = ( fid == numEntries_ - 1 ) ? 0 : fid + 1;
		return ret;
	}

	if ( fid == 0 )
		return vector< unsigned int >( 1, 1 );
	else if ( fid == numEntries_ - 1 )
		return vector< unsigned int >( 1, numEntries_ - 2 );
		
	vector< unsigned int > ret( 2, 0 );
	ret[0] = fid - 1;
	ret[1] = fid + 1;
	return ret;	
}

/// Virtual function to return diffusion X-section area for each neighbor
vector< double > CylMesh::getDiffusionArea( unsigned int fid ) const
{
	if ( numEntries_ <= 1 )
		return vector< double >( 0 );

	double rlow = r0_ + fid * rSlope_;
	double rhigh = r0_ + (fid + 1.0) * rSlope_;

	if ( fid == 0 ) {
		if ( isToroid_ ) {
			vector < double > ret( 2 );
			ret[0] = rlow * rlow * PI;
			ret[1] = rhigh * rhigh * PI;
			return ret;
		} else {
			return vector < double >( 1, rhigh * rhigh * PI );
		}
	}

	if ( fid == (numEntries_ - 1 ) ) {
		if ( isToroid_ ) {
			vector < double > ret( 2 );
			ret[0] = rlow * rlow * PI;
			ret[1] = r0_ * r0_ * PI; // Wrapping around
			return ret;
		} else {
			return vector < double >( 1, rlow * rlow * PI );
		}
	}
	vector< double > ret( 2 );
	ret[0] = rlow * rlow * PI;
	ret[1] = rhigh * rhigh * PI;
	return ret;
}

/// Virtual function to return scale factor for diffusion. 1 here.
vector< double > CylMesh::getDiffusionScaling( unsigned int fid ) const
{
	if ( numEntries_ <= 1 )
		return vector< double >( 0 );

	if ( !isToroid_ && ( fid == 0 || fid == (numEntries_ - 1) ) )
		return vector< double >( 1, 1.0 );

	return vector< double >( 2, 1.0 );
}

//////////////////////////////////////////////////////////////////

/**
 * Inherited virtual func. Returns number of MeshEntry in array
 */
unsigned int CylMesh::innerNumEntries() const
{
	return numEntries_;
}
