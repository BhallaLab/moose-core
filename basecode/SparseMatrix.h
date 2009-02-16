/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SPARSE_MATRIX_H
#define _SPARSE_MATRIX_H

/**
 * Template for specialized SparseMatrix. Used both for the Kinetic
 * solver and for handling certain kinds of messages. Speciality is that
 * it can extract entire rows efficiently, for marching through a 
 * specified row for a matrix multiplication or for traversing messages.
 *
 * Requires that type T have an equality operator ==
 */

extern const unsigned int SM_MAX_ROWS;
extern const unsigned int SM_MAX_COLUMNS;
extern const unsigned int SM_RESERVE;

typedef vector< class T >::const_iterator constTypeIter;
template < class T > class SparseMatrix
{
	public:
		SparseMatrix()
			: nrows_( 0 ), ncolumns_( 0 )
		{
			N_.resize( 0 );
			N_.reserve( SM_RESERVE );
			colIndex_.resize( 0 );
			colIndex_.reserve( SM_RESERVE );
		}

		SparseMatrix( unsigned int nrows, unsigned int ncolumns )
		{
			setSize( nrows, ncolumns );
		}

		/**
		 * Should be called only at the start. Subsequent resizing destroys
		 * the contents.
		 */
		void setSize( unsigned int nrows, unsigned int ncolumns ) {
			if ( nrows < SM_MAX_ROWS && ncolumns < SM_MAX_COLUMNS ) {
				N_.resize( 0 );
				N_.reserve( 2 * nrows );
				nrows_ = nrows;
				ncolumns_ = ncolumns;
				rowStart_.resize( nrows + 1, 0 );
				colIndex_.resize( 0 );
				colIndex_.reserve( 2 * nrows );
			} else {
				cerr << "Error: SparseMatrix::setSize( " <<
				nrows << ", " << ncolumns << ") out of range: ( " <<
				SM_MAX_ROWS << ", " << SM_MAX_COLUMNS << ")\n";
			}
		}

		/**
		 * Assigns and if necessary adds an entry in the matrix. 
		 * This variant does NOT remove any existing entry.
		 */
		void set( unsigned int row, unsigned int column, T value )
		{
			vector< unsigned int >::iterator i;
			vector< unsigned int >::iterator begin = 
				colIndex_.begin() + rowStart_[ row ];
			vector< unsigned int >::iterator end = 
				colIndex_.begin() + rowStart_[ row + 1 ];
		
			if ( begin == end ) { // Entire row was empty.
				unsigned long offset = begin - colIndex_.begin();
				colIndex_.insert( colIndex_.begin() + offset, column );
				N_.insert( N_.begin() + offset, value );
				for ( unsigned int j = row + 1; j <= nrows_; j++ )
					rowStart_[ j ]++;
				return;
			}
		
			if ( column > *( end - 1 ) ) { // add entry at end of row.
				unsigned long offset = end - colIndex_.begin();
				colIndex_.insert( colIndex_.begin() + offset, column );
				N_.insert( N_.begin() + offset, value );
				for ( unsigned int j = row + 1; j <= nrows_; j++ )
					rowStart_[ j ]++;
				return;
			}
			for ( i = begin; i != end; i++ ) {
				if ( *i == column ) { // Found desired entry. By defn it is nonzero.
					N_[ i - colIndex_.begin()] = value;
					return;
				} else if ( *i > column ) { // Desired entry is blank.
					unsigned long offset = i - colIndex_.begin();
					colIndex_.insert( colIndex_.begin() + offset, column );
					N_.insert( N_.begin() + offset, value );
					for ( unsigned int j = row + 1; j <= nrows_; j++ )
						rowStart_[ j ]++;
					return;
				}
			}
		}

		/**
		 * Removes specified entry.
		 */
		void unset( unsigned int row, unsigned int column )
		{
			vector< unsigned int >::iterator i;
			vector< unsigned int >::iterator begin = 
				colIndex_.begin() + rowStart_[ row ];
			vector< unsigned int >::iterator end = 
				colIndex_.begin() + rowStart_[ row + 1 ];
		
			if ( begin == end ) { // Entire row was empty. Ignore
				return;
			}
		
			if ( column > *( end - 1 ) ) { // End of row. Ignore
				return;
			}
			for ( i = begin; i != end; i++ ) {
				if ( *i == column ) { // Found desired entry. Zap it.
					unsigned long offset = i - colIndex_.begin();
					colIndex_.erase( i );
					N_.erase( N_.begin() + offset );
					for ( unsigned int j = row + 1; j <= nrows_; j++ )
						rowStart_[ j ]--;
					return;
				} else if ( *i > column ) { //Desired entry is blank. Ignore
					return;
				}
			}
		}

		/**
		 * Returns the entry identified by row, column. Returns T(0)
		 * if not found
		 */
		T get( unsigned int row, unsigned int column ) const
		{
			assert( row < nrows_ && column < ncolumns_ );
			vector< unsigned int >::const_iterator i;
			vector< unsigned int >::const_iterator begin = 
				colIndex_.begin() + rowStart_[ row ];
			vector< unsigned int >::const_iterator end = 
				colIndex_.begin() + rowStart_[ row + 1 ];
		
			i = find( begin, end, column );
			if ( i == end ) { // most common situation for a sparse Stoich matrix.
				return 0;
			} else {
				return N_[ rowStart_[row] + (i - begin) ];
			}
		}

		/**
		 * Used to get an entire row of entries. 
		 * Returns # entries.
		 * Passes back iterators for the row and for the column index.
		 *
		 * Ideally I should provide a foreach type function so that the
		 * user passes in their operation as a functor, and it is 
		 * applied to the entire row.
		 *
		 */
		unsigned int getRow( unsigned int row, 
			const T** entry, const unsigned int** colIndex ) const
		{
			if ( row >= nrows_ )
				return 0;
			unsigned int rs = rowStart_[row];
			if ( rs >= N_.size() )
				return 0;			
			*entry = &( N_[ rs ] );
			*colIndex = &( colIndex_[rs] );
			return rowStart_[row + 1] - rs;
		}

		/**
		 * This is an unnatural lookup here, across the grain of the
		 * sparse matrix.
		 * Ideally should use copy_if, but the C++ chaps forgot it.
		 */
		unsigned int getColumn( unsigned int col, 
			vector< T >& entry, 
			vector< unsigned int >& rowIndex ) const
		{
			entry.resize( 0 );
			rowIndex.resize( 0 );

			unsigned int row = 0;
			for ( unsigned int i = 0; i < N_.size(); ++i ) {
				if ( col == colIndex_[i] ) {
					entry.push_back( N_[i] );
					while ( rowStart_[ row + 1 ] <= i )
						row++;
					rowIndex.push_back( row );
				}
			}
			return entry.size();
		}

		void rowOperation( unsigned int row, unary_function< T, void>& f )
		{
			assert( row < nrows_ );

			constTypeIter i;
			// vector< T >::const_iterator i;
			unsigned int rs = rowStart_[row];
			vector< unsigned int >::const_iterator j = colIndex_.begin() + rs;
			// vector< T >::const_iterator end = 
			constTypeIter end = 
				N_.begin() + rowStart_[ row + 1 ];

			// for_each 
			for ( i = N_.begin() + rs; i != end; ++i )
				f( *i );
		}

		unsigned int nRows() const {
			return nrows_;
		}

		unsigned int nColumns() const {
			return ncolumns_;
		}

		unsigned int nEntries() const {
			return N_.size();
		}

	private:
		unsigned int nrows_;
		unsigned int ncolumns_;
		vector< T > N_;	/// Non-zero entries in the SparseMatrix.

		/* 
		 * Column index of each non-zero entry. 
		 * This matches up entry-by entry with the N_ vector.
		 */
		vector< unsigned int > colIndex_;	

		/// Start index in the N_ and colIndex_ vectors, of each row.
		vector< unsigned int > rowStart_;
};

#endif // _SPARSE_MATRIX_H
