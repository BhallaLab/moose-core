/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SPINE_ENTRY_H
#define _SPINE_ENTRY_H

/**
 * Helper class for the SpineMesh. Defines a single spine.
 */

class SpineEntry
{
	public:
		/**
		 * This builds the node using info from the compartment.
		 */
		SpineEntry( Id shaft, Id head, unsigned int parent );
		/**
		 * Empty constructor for vectors
		 */
		SpineEntry();

		/// Returns index of parent entry located on NeuroMesh
		unsigned int parent() const;
		/// Assigns index of parent entry located on NeuroMesh
		void setParent( unsigned int parent );

		/// Returns index of self. Only a single voxel.
		unsigned int fid() const;

		Id shaftId() const; /// Returns Id of shaft electrical compartment.
		Id headId() const; /// Returns Id of head electrical compartment.

		/**
		 * Generate list of matching CubeMesh entries to the single Head
		 * entry.
		 */
		void matchCubeMeshEntriesToHead( const ChemCompt* compt,
				unsigned int myIndex,
				double granularity, vector< VoxelJunction >& ret ) const;

		/**
		 * Generate list of matching CubeMeshEntries to the single PSD.
		 */
		void matchCubeMeshEntriesToPSD( const ChemCompt* compt,
				unsigned int myIndex,
				double granularity, vector< VoxelJunction >& ret ) const;

		/**
		 * Find the matching matching NeuroMesh entry index to the 
		 * root of the shaft of this spine. Also compute the area and
		 * diffusion length of the shaft.
		 */
		unsigned int matchNeuroMeshEntriesToShaft( const ChemCompt* compt,
			unsigned int myIndex,
	   		double& area, double& length ) const;

		/// Return volume of spine. Ignores shaft volume. Virtual func.
		double volume() const;

		void mid( double& x, double& y, double& z ) const;

		void matchCubeMeshEntries( const ChemCompt* other, 
			unsigned int myIndex,
			double granularity, vector< VoxelJunction >& ret );

	private:
		CylBase root_; /// Anchor point on dendrite
		CylBase shaft_; /// Shaft cylinder
		CylBase head_; /// head cylinder.
		/**
		 * Index of parent entry on NeuroMesh.
		 */
		unsigned int parent_; 

		/// Id of electrical compartment in which this diffusive compt lives
		Id shaftId_; 
		Id headId_; 
};

#endif	// _SPINE_ENTRY_H
