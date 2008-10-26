/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _KIN_SPARSE_MATRIX_H
#define _KIN_SPARSE_MATRIX_H
class KinSparseMatrix
{
	friend ostream& operator <<( ostream& s, KinSparseMatrix& sm );
#ifdef DO_UNIT_TESTS
	friend void testKinSparseMatrix();
#endif

	public:
		KinSparseMatrix();

		KinSparseMatrix( unsigned int nrows, unsigned int ncolumns )
		{
			setSize( nrows, ncolumns );
		}

		void setSize( unsigned int nrows, unsigned int ncolumns );

		void set( unsigned int row, unsigned int column, int value );

		int get( unsigned int row, unsigned int column );

		unsigned int nRows() {
			return nrows_;
		}

		unsigned int nColumns() {
			return ncolumns_;
		}

		double computeRowRate( 
			unsigned int row, const vector< double >& v
		) const;

		/**
		 * Does a special self-product of the specified row. Output
		 * is the set of nonzero indices in the product
		 * abs( Rij ) * neg( Rjk ) for the specified index i, where
		 * neg( val ) is true only if val < 0.
		 */
		void getGillespieDependence( 
			unsigned int row, vector< unsigned int >& cols
		) const;

		/**
		 * Transposes the matrix, which requires a fair amount of juggling
		 * because of the way it is stored internally.
		 */
		void transpose( KinSparseMatrix& ret ) const;

		/**
		 * Fires a stochastic reaction: It undergoes a single transition
		 * This operation updates the mol concs due to the reacn.
		 */
		void fireReac( unsigned int reacIndex, vector< double >& S ) const;
		

	private:
		unsigned int nrows_; /// Number of molecules in a kinetc system.
		unsigned int ncolumns_; /// Number of reactions.
		vector< int > N_;	/// Non-zero entries in the KinSparseMatrix.

		/* 
		 * Column index of each non-zero entry. 
		 * This matches up entry-by entry with the N_ vector.
		 */
		vector< unsigned int > colIndex_;	

		/*
		 * Start index in the N_ and colIndex_ vectors, of each row.
		 * Additionally stores one last entry in nRows_ + 1, for the end
		 * of the N_ vector.
		 */
		vector< unsigned int > rowStart_;
		static const unsigned int MAX_ROWS;
		static const unsigned int MAX_COLUMNS;
};

#endif // _KIN_SPARSE_MATRIX_H
