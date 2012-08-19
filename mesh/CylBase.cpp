/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <vector>
#include <cassert>
using namespace std;
#include "CylBase.h"
extern const double PI; // defined in consts.cpp

CylBase::CylBase( double x, double y, double z, 
					double dia, double length, unsigned int numDivs )
	:
		x_( x ),
		y_( y ),
		z_( z ),
		dia_( dia ),
		length_( length ),
		numDivs_( numDivs )
{
	;
}

CylBase::CylBase()
	:
		x_( 0.0 ),
		y_( 0.0 ),
		z_( 0.0 ),
		dia_( 1.0 ),
		length_( 1.0 ),
		numDivs_( 1.0 )
{
	;
}

void CylBase::setX( double v )
{
	x_ = v;
}

double CylBase::getX() const
{
	return x_;
}

void CylBase::setY( double v )
{
	y_ = v;
}

double CylBase::getY() const
{
	return y_;
}

void CylBase::setZ( double v )
{
	z_ = v;
}

double CylBase::getZ() const
{
	return z_;
}

void CylBase::setDia( double v )
{
	dia_ = v;
}

double CylBase::getDia() const
{
	return dia_;
}

void CylBase::setLength( double v )
{
	length_ = v;
}

double CylBase::getLength() const
{
	return length_;
}

void CylBase::setNumDivs( unsigned int v )
{
	numDivs_ = v;
}

unsigned int CylBase::getNumDivs() const
{
	return numDivs_;
}
//////////////////////////////////////////////////////////////////
// FieldElement assignment stuff for MeshEntries
//////////////////////////////////////////////////////////////////

/**
 * The entire volume for a truncated cone is given by:
 * V = 1/3 pi.length.(r0^2 + r0.r1 + r1^2)
 * where the length is the length of the cone
 * r0 is radius at base
 * r1 is radius at top.
 * Note that this converges to volume of a cone if r0 or r1 is zero, and
 * to the volume of a cylinder if r0 == r1.
 */
double CylBase::volume( const CylBase& parent ) const
{
	double r0 = parent.dia_/2.0;
	double r1 = dia_/2.0;
	return length_ * ( r0*r0 + r0 *r1 + r1 * r1 ) * PI / 3.0;
}

/**
 * Returns volume of MeshEntry.
 * This isn't the best subdivision of the cylinder from the viewpoint of
 * keeping the length constants all the same for different volumes.
 * Ideally the thinner segments should have a smaller length.
 * But this is simple and so is the diffusion calculation, so leave it.
 * Easy to fine-tune later by modifying how one computes frac0 and frac1.
 */
double CylBase::voxelVolume( const CylBase& parent, unsigned int fid ) const
{
	assert( numDivs_ > fid );
 	double frac0 = ( static_cast< double >( fid ) ) / 
				static_cast< double >( numDivs_ );
 	double frac1 = ( static_cast< double >( fid + 1 ) ) / 
				static_cast< double >( numDivs_ );
	double r0 = 0.5 * ( parent.dia_ * ( 1.0 - frac0 ) + dia_ * frac0 );
	double r1 = 0.5 * ( parent.dia_ * ( 1.0 - frac1 ) + dia_ * frac1 );
	double s0 = length_ * frac0;
	double s1 = length_ * frac1;

	return (s1 - s0) * ( r0*r0 + r0 *r1 + r1 * r1 ) * PI / 3.0;
}

/// Virtual function to return coords of mesh Entry.
/// For Cylindrical mesh, coords are x1y1z1 x2y2z2 r0 r1 phi0 phi1
vector< double > CylBase::getCoordinates( 
					const CylBase& parent, unsigned int fid ) const
{
	assert( numDivs_ > fid );
 	double frac0 = ( static_cast< double >( fid ) ) / 
				static_cast< double >( numDivs_ );
 	double frac1 = ( static_cast< double >( fid + 1 ) ) / 
				static_cast< double >( numDivs_ );

	double r0 = 0.5 * ( parent.dia_ * ( 1.0 - frac0 ) + dia_ * frac0 );
	// equivalent: double r0 = parent.dia_ + frac0 * ( dia_ - parent.dia_ );
	double r1 = 0.5 * ( parent.dia_ * ( 1.0 - frac1 ) + dia_ * frac1 );

	vector< double > ret( 10, 0.0 );
	ret[0] = parent.x_ + frac0 * ( x_ - parent.x_ );
	ret[1] = parent.y_ + frac0 * ( y_ - parent.y_ );
	ret[2] = parent.z_ + frac0 * ( z_ - parent.z_ );
	ret[3] = parent.x_ + frac1 * ( x_ - parent.x_ );
	ret[4] = parent.y_ + frac1 * ( y_ - parent.y_ );
	ret[5] = parent.z_ + frac1 * ( z_ - parent.z_ );
	ret[6] = r0;
	ret[7] = r1;
	ret[8] = 0;
	ret[9] = 0;
	
	return ret;
}

/**
 * Returns diffusion cross-section from specified index to next.
 * For index 0, this is cross-section to parent.
 * For index numDivs-1, it is the cross-section from the second-last to
 * the last voxel in this CylBase.
 * Thus there is no valid value for (index == numDivs - 1), it has
 * to be computed external to the CylBase, typically by calling the
 * getDiffusionArea for the child CylBase.
 */
double CylBase::getDiffusionArea( 
				const CylBase& parent, unsigned int fid ) const
{
	assert( fid < numDivs_ );
 	double frac0 = ( static_cast< double >( fid ) ) / 
				static_cast< double >( numDivs_ );
	double r0 = 0.5 * ( parent.dia_ * ( 1.0 - frac0 ) + dia_ * frac0 );
	return PI * r0 * r0;
}

