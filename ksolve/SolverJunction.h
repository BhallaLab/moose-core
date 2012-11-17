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
		 * Do the calculation as a simple sum onto the target vector.
		 * Later plan a more sophisticated numerical approach than explicit
		 * Euler.
		 */
		void incrementTargets( vector< vector< double > >& y, 
						const vector< double >& v ) const;

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
		 * The total number of transmitted datapoints is 
		 * (crossTerms_.size() + diffTerms_.size() ) * meshIndex_.size().
		 * Given the symmetry of this matrix, we do the following for the
		 * targets: we specify the target molecules in one vector.
		 * We separately specify the target meshIndices for each of the
		 * meshIndices. 
		 * A simple one-to-one map won't work, because a
		 * given reac may have multiple targets, and a given meshIndex
		 * may map onto multiple (or fractional) target meshIndices.
		 * So each of the vectors below has first the index of the source,
		 * and then the index of the target.
		 */
		vector< pair< unsigned int, unsigned int > > targetMols_;
		vector< pair< unsigned int, unsigned int > > targetMeshIndices_;
};

#endif // _SOLVER_JUNCTION_H
