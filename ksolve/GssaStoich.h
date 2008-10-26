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

#ifndef _gssaStoich_h
#define _gssaStoich_h
class GssaStoich: public Stoich
{
	public:
		GssaStoich();

		///////////////////////////////////////////////////
		// Msg Dest function definitions
		///////////////////////////////////////////////////
		static void reinitFunc( const Conn* c );
		static void integrateFunc( 
			const Conn* c, vector< double >* v, double dt );
		void clear( Eref stoich );

		///////////////////////////////////////////////////
		// Functions used by the GillespieIntegrator
		///////////////////////////////////////////////////
		void rebuildMatrix( Eref stoich, vector< Id >& ret );
		void updateDependentRates( const vector< unsigned int >& deps );
		unsigned int pickReac();
		void innerProcessFunc( Eref e, ProcInfo info );
	private:

		///////////////////////////////////////////////////
		// These functions control the updates of state
		// variables by calling the functions in the StoichMatrix.
		///////////////////////////////////////////////////
		void updateV( );
		void updateRates( vector< double>* yprime, double dt  );
		
		///////////////////////////////////////////////////
		// Internal fields.
		///////////////////////////////////////////////////

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
		double continuationT_;
		// unsigned int continuationReac_;

		/**
		 * transN_ is the transpose of the N_ (stoichiometry) matrix. 
		 * It is expensive to compute, but once set up gives fast
		 * operations for a number of steps in the algorithm.
		 */
		KinSparseMatrix transN_; 
};
#endif // _Stoich_h
