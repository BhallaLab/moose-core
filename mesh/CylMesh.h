/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _CYL_MESH_H
#define _CYL_MESH_H

/**
 * The CylMesh represents a chemically identified compartment shaped
 * like an extended cylinder. This is psuedo-1 dimension: Only the
 * axial dimension is considered for diffusion and subdivisions.
 * Typically used in modelling small segments of dendrite
 */
class CylMesh: public ChemCompt
{
	public: 
		CylMesh();
		~CylMesh();
		//////////////////////////////////////////////////////////////////
		//  Utility func
		//////////////////////////////////////////////////////////////////
		/**
		 * Recomputes all local coordinate and meshing data following
		 * a change in any of the coord parameters
		 */

		void updateCoords();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setX0( double v );
		double getX0() const;
		void setY0( double v );
		double getY0() const;
		void setZ0( double v );
		double getZ0() const;
		void setR0( double v );
		double getR0() const;

		void setX1( double v );
		double getX1() const;
		void setY1( double v );
		double getY1() const;
		void setZ1( double v );
		double getZ1() const;
		void setR1( double v );
		double getR1() const;

		void innerSetCoords( const vector< double >& v);
		void setCoords( const Eref& e, const Qinfo* q, vector< double > v );
		vector< double > getCoords( const Eref& e, const Qinfo* q ) const;

		void setLambda( double v );
		double getLambda() const;

		double getTotLength() const;

		unsigned int innerGetDimensions() const;

		//////////////////////////////////////////////////////////////////
		// FieldElement assignment stuff for MeshEntries
		//////////////////////////////////////////////////////////////////

		/// Virtual function to return MeshType of specified entry.
		unsigned int getMeshType( unsigned int fid ) const;
		/// Virtual function to return dimensions of specified entry.
		unsigned int getMeshDimensions( unsigned int fid ) const;
		/// Virtual function to return volume of mesh Entry.
		double getMeshEntrySize( unsigned int fid ) const;
		/// Virtual function to return coords of mesh Entry.
		vector< double > getCoordinates( unsigned int fid ) const;
		/// Virtual function to return info on Entries connected to this one
		vector< unsigned int > getNeighbors( unsigned int fid ) const;
		/// Virtual function to return diffusion X-section area
		vector< double > getDiffusionArea( unsigned int fid ) const;
		/// Virtual function to return scale factor for diffusion. 1 here.
		vector< double > getDiffusionScaling( unsigned int fid ) const;
		/// Volume of mesh Entry including abutting diff-coupled voxels
		double extendedMeshEntrySize( unsigned int fid ) const;

		//////////////////////////////////////////////////////////////////
		/// Inherited virtual.
		void clearExtendedMeshEntrySize();

		/**
		 * Inherited virtual func. Returns number of MeshEntry in array
		 */
		unsigned int innerGetNumEntries() const;
		/// Inherited virtual func.
		void innerSetNumEntries( unsigned int n );
		
		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		/// Virtual func to make a mesh with specified size and numEntries
		void innerBuildDefaultMesh( const Eref& e, const Qinfo* q,
			double size, unsigned int numEntries );

		void innerHandleRequestMeshStats(
			const Eref& e, const Qinfo* q,
			const SrcFinfo2< unsigned int, vector< double > >*
				meshStatsFinfo
		);

		void innerHandleNodeInfo(
			const Eref& e, const Qinfo* q, 
			unsigned int numNodes, unsigned int numThreads );

		void transmitChange( const Eref& e, const Qinfo* q );

		void buildStencil();

		unsigned int getStencil( unsigned int meshIndex,
			const double** entry, const unsigned int** colIndex ) const;

		void extendStencil( 
		   	const ChemCompt* other, const vector< VoxelJunction >& vj );

		/// virtual func implemented here.
		void innerResetStencil();
		
		//////////////////////////////////////////////////////////////////
		// inherited virtual funcs for Boundary
		//////////////////////////////////////////////////////////////////
		
		void matchMeshEntries( const ChemCompt* other, 
			vector< VoxelJunction > & ret ) const;

		double nearest( double x, double y, double z, 
						unsigned int& index ) const;
	
		void indexToSpace( unsigned int index, 
						double& x, double& y, double& z ) const;
		
		//////////////////////////////////////////////////////////////////
		// Inner specific functions needed by matchMeshEntries.
		//////////////////////////////////////////////////////////////////
		void matchCylMeshEntries( const CylMesh* other,
			vector< VoxelJunction >& ret ) const;
		void matchCubeMeshEntries( const CubeMesh* other,
			vector< VoxelJunction >& ret ) const;
		void matchNeuroMeshEntries( const NeuroMesh* other,
			vector< VoxelJunction >& ret ) const;

		//////////////////////////////////////////////////////////////////

		static const Cinfo* initCinfo();

	private:
		double size_; /// Total Volume
		unsigned int numEntries_; // Number of subdivisions to use
		bool useCaps_; // Flag: Should the ends have hemispherical caps?
		bool isToroid_; // Flag: Should the ends loop around mathemagically?

		double x0_; /// coords
		double y0_; /// coords
		double z0_; /// coords

		double x1_; /// coords
		double y1_; /// coords
		double z1_; /// coords

		double r0_;	/// Radius at one end
		double r1_; /// Radius at other end

		double lambda_;	/// Length constant for diffusion. Equal to dx.

		double totLen_;	/// Utility value: Total length of cylinder
		double rSlope_;	/// Utility value: dr/dx
		double lenSlope_; /// Utility value: dlen/dx

//		double dx2_[2]; /// Used as stencil for 2 entries, each = lambda_.

		// Handles core stencil for self.
		SparseMatrix< double > coreStencil_;

		// Handles extended stencil including core + abutting voxels.
		SparseMatrix< double > m_;
};

#endif	// _CYL_MESH_H
