/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _DATA_DIMENSIONS_H
#define _DATA_DIMENSIONS_H

/**
 * This class manages conversion of integers to arbitrary dimension
 * indices. It assumes rectangular arrays.
 * Dimensions and vectors of indices have the fastest-varying index 
 * at zero, and the slowest-varying index at size-1.
 */
class DataDimensions
{
	public:
		DataDimensions( vector< unsigned int > dims )
			: dims_( dims )
		{;}

		DataDimensions()
			: dims_( 0 )
		{;}

		DataDimensions( unsigned int size )
			: dims_( 1, size )
		{;}

		DataDimensions( unsigned int size0, unsigned int size1 )
			: dims_( 2 )
		{
			dims_[0] = size0;
			dims_[1] = size1;
		}

		unsigned int numDimensions() const 
		{
			return dims_.size();
		}

		/**
		 * Returns the size of the specified dimension
		 */
		 unsigned int sizeOfDim1() const
		 {
		 	if ( dims_.size() > 0 ) return dims_[0];
			return 0;
		 }

		 unsigned int sizeOfDim2() const
		 {
		 	if ( dims_.size() > 1 ) return dims_[1];
			return 0;
		 }

		 unsigned int sizeOfDim( unsigned int dim ) const
		 {
		 	if ( dims_.size() > dim ) return dims_[dim];
			return 0;
		 }

		 /**
		  * Converts unsigned int into vector with index in each dimension
		  */
		 vector< unsigned int > multiDimIndex( unsigned int index ) const
		 {
		 	vector< unsigned int > ret;
		 	for ( unsigned int i = 0; i < dims_.size(); ++i ) {
				ret.push_back( index % dims_[i] );
				index /= dims_[i];
			}
			return ret;
		 }

		 /**
		  * Converts index vector into unsigned int
		  */
		 unsigned int linearIndex( const vector< unsigned int >& index ) const
		 {
			assert( index.size() == dims_.size() );
		 	unsigned int ret = 0;
			for ( int i = dims_.size() - 1; i >= 0; ++i ) {
				assert( index[i] < dims_[i] );
				ret = index[i] + ret * dims_[i];
			}
			return ret;
		 }

	private:
		vector< unsigned int > dims_;
};

#endif	// _DATA_DIMENSIONS_H
