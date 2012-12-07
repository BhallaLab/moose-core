/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _CHEM_MESH_H
#define _CHEM_MESH_H

#include "VoxelJunction.h"

/**
 * The ChemMesh represents a chemically identified compartment.
 * This may be spatially extended, and may even be discontinuous.
 * The same set of reactions and molecules populates any given compartment.
 * Examples of compartments might be: nucleus, cell membrane, 
 * early endosomes, spine heads.
 * Connects to one or more 'Geometry' elements to define its boundaries.
 */
class ChemMesh
{
	public: 
		ChemMesh();
		virtual ~ChemMesh();
		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		double getEntireSize( const Eref& e, const Qinfo* q ) const;
		/**
		 * This is a little nasty. It calls buildDefaultMesh with the 
		 * current numEntries. Should not be used if the mesh has been
		 * changed to something more interesting.
		 * Perhaps I need to do something like changeVolOfExistingMesh.
		 */
		void setEntireSize( const Eref& e, const Qinfo* q, double size );

		unsigned int getDimensions() const;
		virtual unsigned int innerGetDimensions() const = 0;

		//////////////////////////////////////////////////////////////////
		// Dest Finfo
		//////////////////////////////////////////////////////////////////

		/**
		 * buildDefaultMesh tells the ChemMesh to make a standard mesh 
		 * partitioning with the specified total size (typically volume)
		 * and the specified number of subdivisions. For example, 
		 * a CubeMesh of size 8 and subdivisions 8 would make a 2x2x2 mesh.
		 * This function is specialized in each derived class.
		 */
		void buildDefaultMesh( const Eref& e, const Qinfo* q,
			double size, unsigned int numEntries );
		virtual void innerBuildDefaultMesh(
			const Eref& e, const Qinfo* q,
			double size, unsigned int numEntries ) = 0;

		void handleRequestMeshStats( const Eref& e, const Qinfo* q );
		virtual void innerHandleRequestMeshStats(
			const Eref& e, const Qinfo* q,
			const SrcFinfo2< unsigned int, vector< double > >*
				meshStatsFinfo
		) = 0;

		void handleNodeInfo( const Eref& e, const Qinfo* q, 
			unsigned int numNodes, unsigned int numThreads );
		virtual void innerHandleNodeInfo(
			const Eref& e, const Qinfo* q, 
			unsigned int numNodes, unsigned int numThreads ) = 0;


		//////////////////////////////////////////////////////////////////
		// FieldElementFinfo stuff for MeshEntry lookup
		//////////////////////////////////////////////////////////////////
		/**
		 * Returns the number of MeshEntries on this ChemMesh
		 */
		unsigned int getNumEntries() const;
		virtual unsigned int innerGetNumEntries() const = 0;

		/**
		 * Dummy function. The numEntries is only set by other functions
		 * that define compartment decomposition.
		 */
		void setNumEntries( unsigned int num );
		virtual void innerSetNumEntries( unsigned int n ) = 0;

		/**
		 * Returns the matched lookupEntry
		 */
		MeshEntry* lookupEntry( unsigned int index );

		//////////////////////////////////////////////////////////////////
		// Utility function for diffusion handling
		//////////////////////////////////////////////////////////////////
		/**
		 * Orchestrates diffusion calculations in the connected Stoich,
		 * if any. Basically acts as a conduit for the execution of the
		 * process call by subsidiary MeshEntries, and funnels these calls
		 * to all attached Stoichs with the incorporation of the stencil
		 * for the actual diffusion calculations.
		 * In due course this should become a virtual function so that
		 * we can have this handled by any ChemMesh class.
		 */
		virtual void updateDiffusion( unsigned int meshIndex ) const;

		/**
		 * Looks up Id of the Stoich object to which this mesh should be
		 * connected. Is called at reinit by the child MeshEntry.
		 */
		void lookupStoich( ObjId me ) const;

		//////////////////////////////////////////////////////////////////
		// Lookup funcs for Boundary
		//////////////////////////////////////////////////////////////////
		Boundary* lookupBoundary( unsigned int index );
		void setNumBoundary( unsigned int num );
		unsigned int getNumBoundary( ) const;

		/**
		 * Wrapper function to buld junction between two meshes, and to
		 * extend the meshes so that their stencils also handle update to
		 * the voxels abutting the boundary on the neighbour mesh.
		 */
		void buildJunction( ChemMesh* other, vector< VoxelJunction >& ret );

		/**
		 * Returns the meshIndices (NOT spatial indices) of all adjacent
		 * mesh entry pairs on ether side of the (self, other) junction.
		 * meshIndices are the indices that look up entries in the vector
		 * of pools.
		 * spatialIndices are (iz * ny + iy) * nx + ix, that is, a linear
		 * conversion of cartesian spatial indices.
		 * So, for two touching cubes, the return vector is the paired 
		 * meshIndices on either side of the plane of contact. If one mesh
		 * has a finer mesh than the other, or if there are more than one
		 * contact points from self to other (for example, at a corner),
		 * then we just have multiple pairs using the same meshIndex of
		 * the repeated voxel.
		 */
		virtual void matchMeshEntries( const ChemMesh* other, 
			vector< VoxelJunction > & ret ) const = 0;


		virtual double nearest( double x, double y, double z, 
						unsigned int& index ) const = 0;
	
		virtual void indexToSpace( unsigned int index, 
						double& x, double& y, double& z ) const = 0;

		/// Utility function for swapping first and second in VoxelJunctions
		void flipRet( vector< VoxelJunction >& ret ) const;

		//////////////////////////////////////////////////////////////////
		// FieldElement assignment stuff for MeshEntries
		//////////////////////////////////////////////////////////////////
		/// Virtual function to return MeshType of specified entry.
		virtual unsigned int getMeshType( unsigned int fid )
			const = 0;
		/// Virtual function to return dimensions of specified entry.
		virtual unsigned int getMeshDimensions( unsigned int fid )
			const = 0;
		/// Virtual function to return volume of mesh Entry.
		virtual double getMeshEntrySize( unsigned int fid ) 
			const = 0;
		/// Virtual function to return coords of mesh Entry.
		virtual vector< double > getCoordinates( unsigned int fid ) 
			const = 0;
		/// Virtual function to return info on Entries connected to this one
		virtual vector< unsigned int > getNeighbors( unsigned int fid ) 
			const = 0;
		/// Virtual function to return diffusion X-section area per neighbor
		virtual vector< double > getDiffusionArea( unsigned int fid )
			const = 0;
		/// Virtual function to return scale factor for diffusion. 1 here.
		virtual vector< double > getDiffusionScaling( unsigned int fid ) 
			const = 0;

		/// Volume of mesh Entry including abutting diff-coupled voxels
		virtual double extendedMeshEntrySize( unsigned int fid ) 
			const = 0;

		/**
		 * Function to look up scale factor derived from area and length
		 * of compartment junction, for all the mesh entries connected to
		 * the specified one. 
		 * Modeled on equivalent function in SparseMatrix.
		 * meshIndex: index of reference mesh entry
		 * entry: array of values of scale factor
		 * colIndex: array of relative indices for each entry. The values
		 * 	returned here are the offset from the meshIndex.
		 * Returns number of entries and colIndexes.
		 * For a 1-D mesh, there will be 2 except at boundaries
		 * For a 2-D mesh, there will be 4 except at boundaries
		 * For a 3-D mesh, there will be 6 except at boundaries
		 * For a neuromesh, there will be a variable number depending on
		 * branching.
		 * For a CylMesh there are 2 except at boundaries.
		 */
		virtual unsigned int getStencil( unsigned int meshIndex,
				const double** entry, const unsigned int** colIndex )
			   	const = 0;


		/**
		 * Function to add voxels for boundaries. This is done so that the
		 * solver can do reaction-diffusion computations on the entire mesh
		 * including voxels of neighbouring solvers abutting the boundary.
		 * It uses these to stitch together the computations that span
		 * multiple solvers and compartments.
		 */
		virtual void extendStencil( 
			const ChemMesh* other, const vector< VoxelJunction >& vj ) = 0;
		//////////////////////////////////////////////////////////////////

		static const Cinfo* initCinfo();

	protected:
		double size_; /// Volume or area
		Id stoich_; /// Identifier for stoich object doing diffusion.

		/**
		 * defines how to combine neighbouring
		 * mesh elements to set up the diffusion du/dt term, using the
		 * method of lines.
		 */
		vector< const Stencil* > stencil_;
	private:
		MeshEntry entry_; /// Wrapper for self ptr

		/**
		 * The Boundaries are Element Fields. They appear as distinct
		 * Elements, though they are controlled by the ChemCompt.
		 * These are the interfaces between compartments, or just
		 * the boundaries of the current one. Each Boundary can be
		 * diffusive, reflective, or an interface where molecules in
		 * different compartments can talk to each other.
		 * All boundaries have a message to a Geometry. The Geometries
		 * may be shared, which is why the boundary isn't a Geometry itself.
		 * If it is an interface (diffusive or other) then the boundary 
		 * also contains a message to the adjacent compartment.
		 */
		vector< Boundary > boundaries_;
};

extern SrcFinfo5< 
	double,
	vector< double >, 
	vector< unsigned int>, 
	vector< vector< unsigned int > >, 
	vector< vector< unsigned int > >
	>* meshSplit();

#endif	// _CHEM_MESH_H
