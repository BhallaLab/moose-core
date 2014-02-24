/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class FastMatrixElim: public SparseMatrix< double >
{
	public:
		void makeTestMatrix( const double* test, unsigned int numCompts );

		/** 
		 * Reduces the forward elimination phase into a series of operations
		 * defined by the fops vector. 
		 */
		void buildForwardElim( vector< unsigned int >& diag,
				vector< Triplet< double > >& fops );
		/** 
		 * Reduces the backward substitution phase into a series of 
		 * operations defined by the bops vector, and by the list of
		 * values on the diagonal. 
		 */
		void buildBackwardSub( vector< unsigned int >& diag,
			vector< Triplet< double > >& bops, vector< double >& diagVal );
		/////////////////////////////////////////////////////////////
		// Here we do stuff to set up the Hines ordering of the matrix.
		/////////////////////////////////////////////////////////////
		/**
		 * Takes the tree specification in the form of a list of parent
		 * entries for each tree node, and reorders the matrix into the
		 * twig-first sequence required for fast elimination.
		 * Returns true if it succeeded in doing this; many matrices will
		 * not reorder correctly.
		 */
		bool hinesReorder( const vector< unsigned int >& parentVoxel );

		/**
		 * Reorders rows of the matrix according to the vector 
		 * lookupOldRowFromNew. The vector tells the function which old
		 * row to put in the ith row of the new matrix. Since the
		 * matrix has matching column entries, those get shuffled too.
		 */
		void shuffleRows( 
				const vector< unsigned int >& lookupOldRowFromNew );

		/**
		 * Does the actual computation of the matrix inversion, which is
		 * equivalent to advancing one timestem in Backward Euler.
		 * Static function here to keep namespaces clean.
		 */
		static void advance( vector< double >& y,
			const vector< Triplet< double > >& ops, //has both fops and bops
			const vector< double >& diagVal );
};

void sortByColumn( 
			vector< unsigned int >& col, vector< double >& entry );

// Todo: Maintain an internal vector of the mapping between rows so that
// the output vector can be updated in the right order, and input values
// can be mapped if matrix reassignment happens.
// The input to the parent class should just be a matrix with diameter
// and connectivity info, and then the system spawns out the ops
// vector depending on diffusion constant.
