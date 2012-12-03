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
		 * mesh indices to which the reac and diff terms apply
		 */
		const vector< unsigned int >& meshIndex() const;

		/**
		 * The map of meshIndex and diffTerm to incoming vector index.
		 */
		const vector< VoxelJunction >& meshMap() const;

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
		void setReacTerms( const vector< unsigned int >& reacTerms,
			const vector< pair< unsigned int, unsigned int > >& poolMap );
		void setDiffTerms( const vector< unsigned int >& diffTerms );
		void setMeshIndex( const vector< unsigned int >& meshIndex,
			const vector< VoxelJunction >& meshMap );

		//////////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();
	private:
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
		 * meshIndex_:
		 * For each crossTerm, we need a vector of meshIndices to which
		 * the term is applied. However, we can safely assume that 
		 * all crossTerms will use the same set of meshIndices as they
		 * all apply to the same junction between compts.
		 */
		vector< unsigned int > meshIndex_;

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
		 * The total number of transmitted datapoints is 
		 * 	(crossTerms_.size() * meshIndex_.size() + 
		 * 	diffTerms_.size() ) * targetmeshIndices_.size().
		 * Given the symmetry of this matrix, we do the following for the
		 * targets: we specify the target molecules in one vector.
		 * We separately specify the target meshIndices for each of the
		 * meshIndices. 
		 * A simple one-to-one map won't work, because a
		 * given reac may have multiple targets, and a given meshIndex
		 * may map onto multiple (or fractional) target meshIndices.
		 */
		/**
 		 * The received vector maps onto Pools by SolverJunction as follows:
 		 * 	The targetMols_ vector::first = vecIndex % (nR)
 		 * 	The targetMols_ vector::second = poolIndex 
 		 */
		vector< pair< unsigned int, unsigned int > > targetMols_;

 		/** 
		 * The received vector maps onto meshEntries (voxels) as follows:
 		 * 	The targetMeshIndices_ vector::first = vecIndex / (nR + nD )
 		 * 	The targetMeshIndices_ vector::second = meshIndex of target.
 		 * 	The targetMeshIndices_ vector::diffScale between src and tgt.
		 */
		vector< VoxelJunction > targetMeshIndices_;
};

extern SrcFinfo1< vector< double > >* updateJunctionFinfo();

#endif // _SOLVER_JUNCTION_H
