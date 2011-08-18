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
		void updateDependentRates( const vector< unsigned int >& deps );
		void updateDependentMathExpn( 
			const vector< unsigned int >& deps );
		void updateAllRates();
		unsigned int pickReac();
		void innerProcessFunc( Eref e, ProcInfo info );

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
		void updateV( );
		void updateRates( vector< double>* yprime, double dt  );

		// virtual func to handle externally imposed changes in mol N
		void innerSetMolN( double y, unsigned int i );
		
		///////////////////////////////////////////////////
		// Internal fields.
		///////////////////////////////////////////////////

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
		 * atot is the total propensity of all the reacns in the system
		 */
		double atot_;

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
		double t_;

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

};
#endif // _GssaStoich_h
