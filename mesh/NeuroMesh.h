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
class NeuroMesh: public ChemMesh
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

		void setCell( Id cellmodel );
		Id getCell() const;
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
		 * Flag. When true, the mesh ignores any compartment with the 
		 * string 'spine' or 'neck' in it. The spine head is below the neck
		 * so it too gets dropped.
		 */
		void setSkipSpines( bool v );
		bool getSkipSpines() const;

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
		double getMeshEntrySize( unsigned int fid ) const;
		/// Virtual function to return coords of mesh Entry.
		vector< double > getCoordinates( unsigned int fid ) const;
		/// Virtual function to return info on Entries connected to this one
		vector< unsigned int > getNeighbors( unsigned int fid ) const;
		/// Virtual function to return diffusion X-section area
		vector< double > getDiffusionArea( unsigned int fid ) const;
		/// Virtual function to return scale factor for diffusion. 1 here.
		vector< double > getDiffusionScaling( unsigned int fid ) const;

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

		void transmitChange( const Eref& e, const Qinfo* q );

		void buildStencil();

		//////////////////////////////////////////////////////////////////
		// Utility functions for building tree.
		/// This shuffles the nodes_ vector to put soma node at the start
		Id putSomaAtStart( Id origSoma, unsigned int maxDiaIndex );

		/**
		 * buildNodeTree: This connects up parent and child nodes
		 * and if needed inserts dummy nodes to build up the model tree.
		 */
		void buildNodeTree( const map< Id, unsigned int >& comptMap );

		//////////////////////////////////////////////////////////////////
		// Utility functions for testing
		const Stencil* getStencil() const;
		const vector< NeuroNode >& getNodes() const;

		//////////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();

	private:
		/**
		 * Array of gemetry specifiers for each segment of the neuronal
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

		/// Flag. True if mesh should ignore spines when scanning dend tree.
		bool skipSpines_; 

		string geometryPolicy_;

		/*
		NeuroStencil ns_;
		bool useCaps_; // Flag: Should the ends have hemispherical caps?
		double totLen_;	/// Utility value: Total length of cylinder
		double rSlope_;	/// Utility value: dr/dx
		double lenSlope_; /// Utility value: dlen/dx
		*/
};


#endif	// _NEURO_MESH_H
