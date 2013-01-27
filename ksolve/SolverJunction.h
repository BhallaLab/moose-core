/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SOLVER_JUNCTION_H
#define _SOLVER_JUNCTION_H

/**
 * The SolverJunction is a one-to-one connection point between solvers
 * which need to exchange information about molecular interchange. There
 * is precisely one SolverJunction between any two solvers.
 * The SolverJunction handles molecular flow through reactions, diffusion,
 * and motors.
 * The SolverJunction supports molecular flow between arbitrary mesh
 * geometries, but itself does not know about any of these details.
 * An overview of the process follows.
 *
 * 1. At each interfacing mesh entry, we have a set of reactions and
 * 	diffusion processes. These are common between all interfacing entries
 * 	at the junction.
 * 		1.1 The total amount of information sent from compartment A is
 *			nM * ( nR + nD )
 *		where nM = number of meshEntries at the junction on the A side
 *		nR = number of reaction terms going across the junction.
 *		nD = number of diffusing molecules.
 *		1.2 This information is sent in a single vector of doubles of length
 *			nM * ( nR + nD )
 *		1.3 The implicit indexing of the vector is like 
 *				vec[meshEntry][Reac,then Diff term]
 *			with the meshEntry varying slower.
 *		1.4 Both nM and nR may differ going the other way across the 
 *			junction. nD must be identical.
 *
 * 2. The SolverJunction handles data structures needed for Sending
 * 	data as well as for Receiving it.
 *		2.1 It sends locally computed reaction and diffusion rates. To do
 *			this it must map to local Stoich::rateTerm data structures.
 *		2.2 It receives externally computed reaction and diffusion rates.
 *			To do this it must map the arriving data to local mesh and
 *			pool indices.
 *
 * 3. The nR reaction rates are all computed in a portion of the regular
 * 		StoichCore::rates_ vector, which contains RateTerms. These
 * 		are updated in StoichCore::updateJunctionRates. The indices of
 * 		RateTerms that are updated are stored in SolverJunction::reacTerms_
 * 		This is passed to StoichCore by GslStoich::updateJunction 
 * 		or equivalent.
 *
 * 4. The nD diffusion rates are all computed in 
 * 		GslStoich::updateJunctionDiffusion or equivalent. The poolIndices
 * 		of diffusing molecules are stored in SolverJunction::diffTerms_
 * 		This is also passed by the GslStoich::updateJunction
 *
 * 5. Each of the above calculations in 3 and 4 are computed for all nM
 * 		meshEntries on the junction on the local compartment. The list 
 * 		of relevant meshEntries is stored in SolverJunction::meshIndex_
 *
 * 6. The combined vector of nM * ( nR + nD ) doubles is sent to the other
 * 	solver. These are all rates, or fluxes. They need to be mapped onto
 * 	the meshIndices and poolIndices of the target solver. Any given mesh
 * 	on the source may map onto one or more on target. Any given rate term
 * 	on the vector may map onto one or more on target.
 * 	With these mapped fluxes, but not with this vector, we can do numerical 
 * 	integration using the method of choice (for now Forward Euler).
 *
 * 7. The received vector is mapped onto Pools by SolverJunction as follows:
 * 	The targetMols_ vector::first = vecIndex % (nR+nD)
 * 	The targetMols_ vector::second = poolIndex 
 * 		This happens in SolverJunction.cpp::incrementTargets.
 * 	
 * 8. The received vector is mapped onto meshEntries (voxels) as follows:
 * 	The targetMeshIndices_ vector::first = vecIndex / (nR + nD )
 * 	The targetMeshIndices_ vector::second = meshIndex
 * 		This happens in SolverJunction.cpp::incrementTargets.
 *
 */
class SolverJunction
{
	public:
		SolverJunction();
		~SolverJunction();
		//////////////////////////////////////////////////////////////////
		// Fields
		//////////////////////////////////////////////////////////////////
		
		unsigned int getNumReacs() const;
		unsigned int getNumDiffMols() const;
		unsigned int getNumMeshIndex() const;
		Id getMyCompartment() const;
		Id getOtherCompartment() const;
		//////////////////////////////////////////////////////////////////
		// Utility functions
		//////////////////////////////////////////////////////////////////

		/**
		 * An index into the rate entries for cross-solver reactions.
		 * The rates are stored in the regular StoichCore rates_ vector,
		 * here we just index it.
		 * Therefore the StoichCore manages all the reactions in the usual
		 * way.
		 */
		const vector< unsigned int >& reacTerms() const;

		/**
		 * an index into the varPools vector, and the corresponding 
		 * DiffConst vector.
		 */
		const vector< unsigned int >& diffTerms() const;

		/**
		 * The map of meshIndex and diffTerm to incoming vector index.
		 */
		const vector< VoxelJunction >& meshMap() const;

		/**
		 * remoteReacPools are the local poolIndices for proxy pools for
		 * pools which live on a remote solver, but which participate in a
		 * reaction located on the current solver.
		 */
		const vector< unsigned int >& remoteReacPools() const;

		/**
		 * localReacPools are local poolIndices of pools which live on the
		 * current solver, but have a cross-solver reaction. 
		 */
		const vector< unsigned int >& localReacPools() const;

		/**
		 * Pool indices of pools whose num will be sent across junction
		 */
		const vector< unsigned int >& sendPoolIndex() const;

		/**
		 * mesh indices of voxels whose pools will be sent across junction
		 */
		const vector< unsigned int >& sendMeshIndex() const;

		/**
		 * Pool indices of pools whose num will be recv across junction.
		 * These indices apply to the entries in the abutting voxels.
		 */
		const vector< unsigned int >& abutPoolIndex() const;

		/**
		 * mesh indices of voxels in extended S matrix, to represent
		 * abutting voxels.
		 */
		const vector< unsigned int >& abutMeshIndex() const;

		/**
		 * Do the calculation as a simple sum onto the target vector.
		 * Later plan a more sophisticated numerical approach than explicit
		 * Euler.
		 */
		void incrementTargets( vector< vector< double > >& y, 
						const vector< double >& v ) const;

		//////////////////////////////////////////////////////////////////
		// Setup functions
		//////////////////////////////////////////////////////////////////
		/// Assigns list of pools which undergo diffusion
		void setDiffTerms( const vector< unsigned int >& diffTerms );

		/// Assigns the localReacPools vector.
		void setLocalReacPools( const vector< unsigned int >& pools );
		/// Assigns the remoteReacPools vector.
		void setRemoteReacPools( const vector< unsigned int >& pools );
		/// Assignes the meshMap.
		void setMeshMap( const vector< VoxelJunction >& meshMap );

		void setSendPools( 
						const vector< unsigned int >& meshIndex,
						const vector< unsigned int >& poolIndex
		);

		void setAbutPools( 
						const vector< unsigned int >& meshIndex,
						const vector< unsigned int >& poolIndex
		);

		void setCompartments( Id myCompt, Id otherCompt );

		//////////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();
	private:
		Id otherCompartment_; /// Id of other compartment
		Id myCompartment_; /// Id of self compartment.
		
		/**
		 * crossTerms identifies cross-compartment reaction RateTerms.
		 * It looks them up from the parent StoichCore's rates_ vector.
		 */
		vector< unsigned int > reacTerms_;

		/**
		 * diffTerms identifies diffusion terms across compartments.
		 * Only a subset of varPools diffuse across compartments. 
		 * So this vector looks up this
		 * subset, and uses the parent StoichCore's diffConsts_ vector
		 * for their diff consts.
		 * diffTerms too apply to the same set of meshIndices as the reacs.
		 */
		vector< unsigned int > diffTerms_;

		/**
		 * diffScale_:
		 * For each meshIndex on the junction, there is a scaling factor
		 * by xa/(h * volume)
		 * For now I assume this is common to all voxels abutting a given
		 * meshIndex, but of course it may be selective for each one.
		 * Deal with that case later.
		 */
		vector< double > diffScale_;

 		/** 
		 * The received vector maps onto meshEntries (voxels) as follows:
 		 * 	The targetMeshIndices_ vector::first = vecIndex / (nR + nD )
 		 * 	The targetMeshIndices_ vector::second = meshIndex of target.
 		 * 	The targetMeshIndices_ vector::diffScale between src and tgt.
		 */
		vector< VoxelJunction > targetMeshIndices_;

		///////////////////////////////////////////////////////////////

		/**
		 * Send vector varies faster by poolIndex, then by 
		 * meshIndex. Likewise Recv vector.
		 *
		 */
		/**
		 * localReacPools_ are local poolIndices of pools which live on the
		 * current solver, but have a cross-solver reaction. So their
		 * values have to be exported and their deltas have to be
		 * imported.
		 * These are indexed by the index of the outgoing # vector, or the
		 * incoming delta vector, modulo total # of pools transmitted.
		 */
		vector< unsigned int > localReacPools_;

		/**
		 * remoteReacPools_ are the special extended local poolIndices of
		 * pools which live on a remote solver, but which participate in a
		 * reaction located on the current solver. Their values are
		 * imported from the remote solver, and their deltas are exported.
		 * These are indexed by the index of the incoming # vector, or the
		 * outgoing delta vector, modulo total # of pools transmitted.
		 */
		vector< unsigned int > remoteReacPools_;

		/**
		 * sendPoolIndex_ are local poolIndices of pools  going out to 
		 * other solver.  We assume that
		 * the same set of pools are sent out by each abutting voxel.
		 * This set includes only diffusive pools.
		 */
		vector< unsigned int > sendPoolIndex_;

		/**
		 * MeshIndices (to lookup S_[meshIndex][poolIndex]) of outgoing
		 * Pools from the core set handled by this solver. The same
		 * indices will be incremented by return messages.
		 * This set of indices applies both to reactions and to diffusion.
		 */
		vector< unsigned int > sendMeshIndex_;

		/**
		 * abutPoolIndex is the local poolIndex of incoming pools, in order,
		 * from the recvVector.
		 */
		vector< unsigned int > abutPoolIndex_;

		/**
		 * abutMeshIndex_:
		 * MeshIndices (to lookup S_[meshIndex][poolIndex]) of incoming
		 * Pools, coming into the extra indices defined for abutments.
		 */
		vector< unsigned int > abutMeshIndex_;

};

extern SrcFinfo1< vector< double > >* junctionPoolDeltaFinfo();
extern SrcFinfo1< vector< double > >* junctionPoolNumFinfo();

#endif // _SOLVER_JUNCTION_H
