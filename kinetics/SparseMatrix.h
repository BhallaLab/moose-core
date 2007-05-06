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
		SparseMatrix()
			: nrows_( 0 ), ncolumns_( 0 )
		{
			N_.resize( 16, 0 );
		}

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
		vector< int > N_;
		static const unsigned int MAX_ROWS;
		static const unsigned int MAX_COLUMNS;
};
