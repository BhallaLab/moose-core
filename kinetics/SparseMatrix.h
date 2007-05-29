/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class SparseMatrix
{
	friend ostream& operator <<( ostream& s, SparseMatrix& sm );

	public:
		SparseMatrix();

		SparseMatrix( unsigned int nrows, unsigned int ncolumns )
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

	private:
		unsigned int nrows_;
		unsigned int ncolumns_;
		vector< int > N_;	/// Non-zero entries in the SparseMatrix.

		/* 
		 * Column index of each non-zero entry. 
		 * This matches up entry-by entry with the N_ vector.
		 */
		vector< unsigned int > colIndex_;	

		/// Start index in the N_ and colIndex_ vectors, of each row.
		vector< unsigned int > rowStart_;
		static const unsigned int MAX_ROWS;
		static const unsigned int MAX_COLUMNS;
};
