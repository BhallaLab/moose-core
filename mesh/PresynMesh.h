/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2021 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _PRESYN_MESH_H
#define _PRESYN_MESH_H

class Bouton
{
	public:
		/// coords for position of bouton
		double x_;
		double y_;
		double z_;

		/// coords for orientation of bouton. Currently for display,
		/// but in due course may inform how release happens.
		/// In theory I could compute on the fly based on postsyn coords.
		double vx_;
		double vy_;
		double vz_;

		double volume_;

		ObjId postsynCompt_;
	Bouton();
};

/**
 * The PresynMesh is a set of synaptic boutons, all having the same
 * reactions. No diffusion between them so it is easy.
 * Two ways to set up: by matching one-to-one to a list of spines, or
 * by matching many-to-one to dendritic segments.
 */
class PresynMesh: public MeshCompt
{
	public:
		PresynMesh();
		~PresynMesh();
		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////
		//void setPostsynMesh( const Eref& e, ObjId v );
		// ObjId getPostsynMesh( const Eref& e ) const;

		/// Return list of postynaptic compartments, one per bouton.
		/// Note that there may be repeats.
		vector< ObjId >getPostsynCompts() const;
		/// Complementary func returning vec of Ids, to match other Meshes.
		vector< Id > getElecComptMap() const;

		vector< unsigned int > getStartVoxelInCompt() const;
		vector< unsigned int > getEndVoxelInCompt() const;

		/// Spacing between boutons if compartments are dendrite segments.
		double getBoutonSpacing() const;

		/// Report the number of boutons defined on mesh.
		unsigned int getNumBoutons() const;

		/// Return volumes for each bouton. This is already handled by
		/// the inherited function getVoxelVolume, but that is readonly.
		vector< double > getAllBoutonVolumes() const;
		void setAllBoutonVolumes( vector< double >& vols );

		/// Already handled by voxelMidpoint
		//vector< double > getBoutonCoords( unsigned int idx ) const;
		vector< double > getAllBoutonOrientation() const;

		/// Already handled by oneVoxelVolume
		// double getBoutonVolume( unsigned int idx ) const;

		int getStartBoutonIndexFromCompartment( ObjId b ) const;
		int getNumBoutonsOnCompartment( ObjId b ) const;

		/// Reports if this set of boutons is connected to spines vs dend.
		bool isOnSpines() const;

		/// Returns the compartments to which each bouton projects.
		vector< ObjId > getTargetCompartments() const;

		/// Returns xyz triplets for each bouton, for a len of 3*numBoutons
		/// Already handled by getVoxelMidpoint
		// vector< double > getAllBoutonCoords() const;

		/// Returns index of each bouton on its target compartment
		/// For spines all entries will be zero.
		vector< int > getBoutonIndexOnTargetCompartments() const;


		//////////////////////////////////////////////////////////////////
		// FieldElement assignment stuff for MeshEntries
		//////////////////////////////////////////////////////////////////

		/// Virtual function to return MeshType of specified entry.
		unsigned int getMeshType( unsigned int fid ) const;
		/// Virtual function to return dimensions of specified entry.
		unsigned int getMeshDimensions( unsigned int fid ) const;
		unsigned int innerGetDimensions() const;
		/// Virtual function to return volume of mesh Entry.
		double getMeshEntryVolume( unsigned int fid ) const;
		/// Virtual function to return coords of mesh Entry.
		vector< double > getCoordinates( unsigned int fid ) const;
		/// Virtual function to return diffusion X-section area
		vector< double > getDiffusionArea( unsigned int fid ) const;
		/// Virtual function to return scale factor for diffusion. 1 here.
		vector< double > getDiffusionScaling( unsigned int fid ) const;
		/// Volume of mesh Entry including abutting diff-coupled voxels
		double extendedMeshEntryVolume( unsigned int fid ) const;

		//////////////////////////////////////////////////////////////////

		/**
		 * Inherited virtual func. Returns number of MeshEntry in array
		 */
		unsigned int innerGetNumEntries() const;
		/// Inherited virtual func.
		void innerSetNumEntries( unsigned int n );

		/// Inherited virtual, do nothing for now.
		vector< unsigned int > getParentVoxel() const;
		const vector< double >& vGetVoxelVolume() const;
		const vector< double >& vGetVoxelMidpoint() const;
		const vector< double >& getVoxelArea() const;
		const vector< double >& getVoxelLength() const;

		/// Inherited virtual. Returns entire volume of compartment.
		double vGetEntireVolume() const;

		/// Inherited virtual. Resizes len and dia of each voxel.
		bool vSetVolumeNotRates( double volume );
		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void buildOnSpineHeads( vector< ObjId > v );
		void buildOnDendrites( vector< ObjId > v, double spacing );
		void setRadiusStats( double r, double sdev );

		/// Virtual func to make a mesh with specified Volume and numEntries
		void innerBuildDefaultMesh( const Eref& e,
			double volume, unsigned int numEntries );

		void innerHandleRequestMeshStats(
			const Eref& e,
			const SrcFinfo2< unsigned int, vector< double > >*
				meshStatsFinfo
		);

		void innerHandleNodeInfo(
			const Eref& e,
			unsigned int numNodes, unsigned int numThreads );

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

		static const Cinfo* initCinfo();

	private:
		/// Flag: Is it on spines or along a dendritic tree?
		bool isOnSpines_;

		/// spacing of presyn boutons along the NeuroMesh (dendrite).
		double spacing_;

		/// These are the data structures for each of the boutons.
		vector< Bouton > boutons_;
};

#endif	// _PRESYN_MESH_H
