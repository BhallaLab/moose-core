/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ANY_DIM_HANDLER_H
#define _ANY_DIM_HANDLER_H

/**
 * This class manages the data part of Elements. It handles arrays of
 * any dimension.
 */
class AnyDimHandler: public BlockHandler
{
	public:
		AnyDimHandler( const DinfoBase* dinfo, bool isGlobal, 
			const vector< int >& dims );

		AnyDimHandler( const AnyDimHandler* other );

		~AnyDimHandler();

		////////////////////////////////////////////////////////////
		// Information functions
		////////////////////////////////////////////////////////////

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const;

		unsigned int sizeOfDim( unsigned int dim ) const;

		vector< unsigned int > dims() const;

		////////////////////////////////////////////////////////////////
		// load balancing functions
		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////
		// Process function
		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////
		// Data Reallocation functions
		////////////////////////////////////////////////////////////////
		/**
		 * Make a single identity copy, doing appropriate node 
		 * partitioning if toGlobal is false.
		 */
		DataHandler* copy( bool toGlobal, unsigned int n ) const;

		DataHandler* copyUsingNewDinfo( const DinfoBase* dinfo) const;

		DataHandler* addNewDimension( unsigned int size ) const;

		bool resize( unsigned int dimension, unsigned int size );

	private:
		// dims_[0] varies fastest. To index it would be 
		// data[dimN][...][dim0]
		vector< unsigned int > dims_;
};

#endif	// _ANY_DIM_HANDLER_H
