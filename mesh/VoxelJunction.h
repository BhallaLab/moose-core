/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _VOXEL_JUNCTION_H
#define _VOXEL_JUNCTION_H

/**
 * This class specifies a single junction between abutting voxels.
 * All it has are the respective meshIndices (not spatialIndices) of the
 * abutting voxels, and a scale factor for diffusion between them.
 */
class VoxelJunction
{
	public:
		VoxelJunction( unsigned int f, unsigned int s, double d = 1.0 )
				: first( f ), second( s ), diffScale( d )
		{;}
		VoxelJunction()
				: first( ~0 ), second( ~0 ), diffScale( 1.0 )
		{;}

		unsigned int first; /// MeshIndex for first compartment
		unsigned int second; /// MeshIndex for second compartment
		double diffScale; /// smaller of the cross-section areas / diffLen
};

#endif	// _VOXEL_JUNCTION_H
