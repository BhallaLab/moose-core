/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _NEURO_MESH_H
#define _NEURO_MESH_H

/**
 * The NeuroMesh represents sections of a neuron whose spatial attributes
 * are obtained from a neuronal model.
 * Like the CylMesh, this is pseudo-1 dimension: Only the
 * axial dimension is considered for diffusion and subdivisions. Branching
 * is also handled.
 *
 *
 * Dendritic spines typically contain different reaction systems from the
 * dendrite, but each spine has the same reactions. So they deserve
 * their own mesh: SpineMesh.
 * The idea is the the SpineMesh has just the spine head compartment,
 * which duplicate the same reactions, but does not diffuse to other
 * spine heads.
 * Instead it has an effective diffusion constant to the parent
 * dendrite compartment, obtained by treating the spine neck as a 
 * diffusion barrier with zero volume.
 */
class NeuroMesh: public MeshCompt
{
	public: 
		NeuroMesh();
		NeuroMesh( const NeuroMesh& other );
		~NeuroMesh();
		NeuroMesh& operator=( const NeuroMesh& other );
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

		/**
		 * Assigns the parent of all the cell compartments.
		 */
		void setCell( const Eref& e, const Qinfo* q, Id cellmodel );
		Id getCell( const Eref& e, const Qinfo* q ) const;
		Id getCell() const;

		/** 
		 * This overloaded function sets up a presumed contiguous set of
 		 * compartments, complains if they are not contiguous due to the 
		 * check in NeuroNode::traverse.
 		 * 
		 * I assume 'cell' is the parent of the compartment tree.
		 * The 'path' argument specifies a wildcard list of compartments, 
		 * which can be also a comma-separated explicit list. Does not 
		 * have to be in any particular order.
		 * The 'path' argument is based off the cell path.
 		 */
		void setCellPortion( const Eref& e, const Qinfo* q, Id cell,
			string path	);
		/**
		 * Assigns a group of compartments to be used for the mesh.
		 */
		void setCellPortion( const Eref& e, const Qinfo* q, 
							Id cell, vector< Id > portion );

		/**
		 * Separates out the spines attached to the selected groups of
		 * compartments. Fills out spine, and if needed, psd list.
		 */
		void separateOutSpines( const Eref& e );

		/**
		 * The SubTree is a contiguous set of compartments to model.
		 * The first entry is the root of the tree, closest to the soma.
		 * The remaining entries are end-branches of the tree (inclusive).
		 * If the tree goes all the way out to the end of a particular 
		 * branch then no entry needs to be put in.
		 */
		void setSubTree( vector< Id > compartments );
		vector< Id > getSubTree() const;

		/**
		 * Flag. True if NeuroMesh should configure a separate SpineMesh.
		 * The process is that both the NeuroMesh and SpineMesh should have
		 * been created, and a spineList message sent from the NeuroMesh
		 * to the SpineMesh. This may cascade down to PsdMesh.
		 */
		void setSeparateSpines( bool v );
		bool getSeparateSpines() const;

		unsigned int getNumSegments() const;
		unsigned int getNumDiffCompts() const;

		void setDiffLength( double v );
		double getDiffLength() const;

		void setGeometryPolicy( string v );
		string getGeometryPolicy() const;

		unsigned int innerGetDimensions() const;

		//////////////////////////////////////////////////////////////////
		// FieldElement assignment stuff for MeshEntries
		//////////////////////////////////////////////////////////////////

		/// Virtual function to return MeshType of specified entry.
		unsigned int getMeshType( unsigned int fid ) const;
		/// Virtual function to return dimensions of specified entry.
		unsigned int getMeshDimensions( unsigned int fid ) const;
		/// Virtual function to return volume of mesh Entry.
		double getMeshEntryVolume( unsigned int fid ) const;
		/// Virtual function to return coords of mesh Entry.
		vector< double > getCoordinates( unsigned int fid ) const;
		/// Virtual function to return diffusion X-section area
		vector< double > getDiffusionArea( unsigned int fid ) const;
		/// Virtual function to return scale factor for diffusion. 1 here.
		vector< double > getDiffusionScaling( unsigned int fid ) const;
		/// Vol of all mesh Entries including abutting diff-coupled voxels
		double extendedMeshEntryVolume( unsigned int fid ) const;

		//////////////////////////////////////////////////////////////////
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

		void transmitChange( const Eref& e, const Qinfo* q, double oldVol );

		/**
		 * Helper function for buildStencil, calculates diffusion term
		 * adx for rate between current compartment curr, and parent.
		 * By product: also passes back parent compartment index.
		 */
		double getAdx( unsigned int curr, unsigned int& parentFid ) const;

		/// Utility function to set up Stencil for diffusion in NeuroMesh
		void buildStencil();

		//////////////////////////////////////////////////////////////////
		// inherited virtual funcs for Boundary
		//////////////////////////////////////////////////////////////////
		
		void matchMeshEntries( const ChemCompt* other, 
			vector< VoxelJunction > & ret ) const;

		void matchSpineMeshEntries( const ChemCompt* other, 
			vector< VoxelJunction > & ret ) const;

		void matchCubeMeshEntries( const ChemCompt* other, 
			vector< VoxelJunction > & ret ) const;

		void matchNeuroMeshEntries( const ChemCompt* other, 
			vector< VoxelJunction > & ret ) const;

		/**
		 * This works a little different from other subclass versions of
		 * the function. It finds the index of the
		 * mesh entry whose centre is closest to the specified coords,
		 * and returns the distance to the centre.
		 * Doesn't worry about whether this distance is inside or outside
		 * cell.
		 */
		double nearest( double x, double y, double z, 
						unsigned int& index ) const;
	
		void indexToSpace( unsigned int index, 
						double& x, double& y, double& z ) const;
		

		//////////////////////////////////////////////////////////////////
		// Utility functions for building tree.
		//////////////////////////////////////////////////////////////////

		/**
		 * Puts in a dummy node between parent and self. Used to
		 * set up the correct size of proximal compartments.
		 */
		void insertSingleDummy( unsigned int parent, unsigned int self,
			   double x, double y, double z	);

		/**
		 * Puts in all the required dummy nodes for the tree.
		 */
		void insertDummyNodes();

		/// This shuffles the nodes_ vector to put soma node at the start
		Id putSomaAtStart( Id origSoma, unsigned int maxDiaIndex );

		/**
		 * buildNodeTree: This connects up parent and child nodes
		 * and if needed inserts dummy nodes to build up the model tree.
		 */
		void buildNodeTree( const map< Id, unsigned int >& comptMap );

		/**
		 * Returns true if it finds a compartment name that looks like
		 * it ought to be on a spine. It filters out the names
		 * "neck", "shaft", "spine" and "head".
		 * The latter two are classified into the head_ vector.
		 * The first two are classified into the shaft_ vector.
		 */
		bool filterSpines( Id compt );
		/** 
		 * converts the parents_ vector from identifying the parent 
		 * NeuroNode to identifying the parent voxel, for each shaft entry.
 		 */
		void updateShaftParents();

		//////////////////////////////////////////////////////////////////
		// Utility functions for testing
		// const Stencil* getStencil() const;
		const vector< NeuroNode >& getNodes() const;

		//////////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();

	private:
		/**
		 * Array of geometry specifiers for each segment of the neuronal
		 * model. Includes information equivalent to
		 * the CylMesh, as well as information to tie the node back to the
		 * original 'compartment' from the neuronal model.
		 */
		vector< NeuroNode > nodes_;

		/**
		 * nodeIndex_[fid_for_MeshEntry].
		 * Looks up index of NeuroNode from the fid of each MeshEntry.
		 */
		vector< unsigned int > nodeIndex_;

		/**
		 * Volscale pre-calculations for each MeshEntry. 
		 * vs = #molecules / vol
		 * where vol is expressed in m^3.
		 */
		vector< double > vs_;

		/**
		 * Mesh junction area pre-calculations for each MeshEntry.
		 * If we consider the Entry specified by the Index, the area
		 * specified is the one more proximal, that is, closer to soma.
		 */
		vector< double > area_;

		double size_; /// Total Volume
		double diffLength_;	/// Max permitted length constant for diffusion
		Id cell_; /// Base object for cell model.

		/**
		 * Flag. True if NeuroMesh should configure a separate SpineMesh.
		 * The process is that both the NeuroMesh and SpineMesh should have
		 * been created, and a spineList message sent from the NeuroMesh
		 * to the SpineMesh.
		 */
		bool separateSpines_; 

		string geometryPolicy_;

		/**
		 * Decides how finely to subdivide diffLength_ or radius or cubic
		 * mesh side when computing surfacearea of intersections with 
		 * CubeMesh. Defaults to 0.1.
		 */
		double surfaceGranularity_;

		/*
		NeuroStencil ns_;
		bool useCaps_; // Flag: Should the ends have hemispherical caps?
		double totLen_;	/// Utility value: Total length of cylinder
		double rSlope_;	/// Utility value: dr/dx
		double lenSlope_; /// Utility value: dlen/dx
		*/
		/**
		 * The shaft vector and the matching head vector track the dendritic
		 * spines. The parent is the voxel to which the spine  is attached.
		 */
		vector< Id > shaft_; /// Id of shaft compartment.
		vector< Id > head_;	/// Id of head compartment
		vector< unsigned int > parent_; /// Index of parent voxel
};


#endif	// _NEURO_MESH_H
