/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This class handles the Gillespie Stochastic Simulation Algorithm
 * using the Stoich class as a base. 
 */

#ifndef _GssaStoich_h
#define _GssaStoich_h
class GssaStoich: public Stoich
{
	public:
		GssaStoich();
		~GssaStoich();

		///////////////////////////////////////////////////
		// Msg Dest function definitions
		///////////////////////////////////////////////////
		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );

		///////////////////////////////////////////////////
		// Field access functions.
		///////////////////////////////////////////////////
		string getMethod() const;
		void setMethod( string method );

		// Overrides the Stoich version.
		void setPath( const Eref& e, const Qinfo* q, string path );

		///////////////////////////////////////////////////
		// Functions used by the GillespieIntegrator
		///////////////////////////////////////////////////
		void rebuildMatrix();
		/**
		 * Used to update rate terms and atot whenever a reaction has
		 * fired. We precompute a set of dependencies from each reaction
		 * to all affected downstream reactions.
		 */
		void updateDependentRates( unsigned int meshIndex,
			const vector< unsigned int >& deps );
		/**
		 * Used to update rate terms and atot whenever the 'n' of a pool
		 * has changed. This may be due to field assignment, alteration
		 * of concInit of a buffered molecule, or through a FuncTerm.
		 */
		void updateDependentRates( unsigned int meshIndex, 
			unsigned int molIndex );
		void updateDependentMathExpn( double t, unsigned int meshIndex,
			const vector< unsigned int >& deps );
		void updateAllRates( unsigned int meshIndex );
		unsigned int pickReac( unsigned int meshIndex, gsl_rng* r );

		static const Cinfo* initCinfo();
	private:

		/**
 		* Inserts reactions that depend on molecules modified by the
 		* specified MathExpn, into the dependency list for the
		* firedReac
 		*/
		void insertMathDepReacs( unsigned int mathDepIndex,
			unsigned int firedReac );
		/**
 		* Fill in dependency list for all MathExpns on reactions.
 		* Note that when a MathExpn updates, it alters a further
 		* molecule, that may be a substrate for another reaction.
 		* So we need to also add further dependent reactions.
 		* In principle we might also cascade to deeper MathExpns. Later.
 		*/
		void fillMathDep();

		/**
 		* Fill in dependency list for all MMEnzs on reactions.
 		* The dependencies of MMenz products are already in the system,
 		* so here we just need to add cases where any reaction product
 		* is the Enz of an MMEnz.
 		*/
		void fillMmEnzDep();

		/// Clean up reac dependency lists: Ensure only unique entries.
		void makeReacDepsUnique();

		///////////////////////////////////////////////////
		// These functions control the updates of state
		// variables by calling the functions in the StoichMatrix.
		///////////////////////////////////////////////////
		// void updateV( );
		// void updateRates( vector< double>* yprime, double dt  );

		// virtual func to handle externally imposed changes in mol N
		// void innerSetMolN( double y, unsigned int i );
		
		///////////////////////////////////////////////////
		// Internal fields.
		///////////////////////////////////////////////////

		/**
		 * Vector of rates of reactions. This is a state vector because
		 * we don't recalculate it each time, only the subset that are
		 * changed by the last reaction.
		 * One vector of v per mesh entry.
		 */
		vector< vector< double > > v_;

		/**
		 * Specifies method to use for calculation. Currently
		 * only G1, but also plan tau-leap and an adaptive
		 * one.
		 */
		 string method_;

		/**
		 * This vector has one entry for each RateTerm. The entry
		 * points to a list of RateTerms that must be updated
		 * whenever the original RateTerm fires.
		 * We use indexing rather than direct pointers to look up the
		 * RateTerms because we also need to look up v_ entries, and
		 * possibly additional entries for the propensity tree.
		 */
		vector< vector< unsigned int > > dependency_; 

		/**
		 * Similar vector, points to SumTots that must be updated
		 * whenever the original RateTerm fires.
		 * Here we're being ambitious: someday it will be a full
		 * Math Expression. For now just SumTot.
		 */
		vector< vector< unsigned int > > dependentMathExpn_; 

		/**
		 * Yet another dependency graph. Here every pool indicates
		 * which reac is dependent on it. Used whenever there is a
		 * forcible change in 'n' or 'conc' of regular or buffered
		 * or function molecules. The case for function molecules also lets
		 * us update whenever a function calculation gives a new
		 * value.
		 */
		vector< vector< unsigned int > > ratesDependentOnPool_; 

		/**
		 * atot is the total propensity of all the reacns in the system
		 * One per mesh entry.
		 */
		vector< double > atot_;

		/**
		 * Here we make a nested structure to handle quick lookup
		 * for which reaction to pick on a given timestep.
		 */
		// PropensityTree propensity_;

		/**
		 * This field is used to avoid recalculation of next time
		 * when the current calculation has been interrupted by a 
		 * checkpoint before time t is reached.
		 */
		vector< double > t_;

		/**
		 * Whenever an external input has invalidated the
		 * stored t_ value and the rates, we need to redo the whole
		 * lot, rather than rely on the stored value.
		 * Not activated yet.
		 */
		bool redoStep_;

		/**
		 * transN_ is the transpose of the N_ (stoichiometry) matrix. 
		 * It is expensive to compute, but once set up gives fast
		 * operations for a number of steps in the algorithm.
		 * Specifically, a row of transN_ has all the molecules
		 * that depend on the specified reacn ( row# ).
		 */
		KinSparseMatrix transN_; 

		/**
		 * meshIndex_[thread][i] has a vector of meshIndices allocated to
		 * each thread.
		 */
		vector< vector< unsigned int > > meshIndex_;
		vector< gsl_rng* > randNumGenerators_;
};
#endif // _GssaStoich_h
