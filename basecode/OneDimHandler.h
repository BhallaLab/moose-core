/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ONE_DIM_HANDLER_H
#define _ONE_DIM_HANDLER_H

/**
 * This class manages the data part of Elements. It handles a one-
 * dimensional array.
 */
class OneDimHandler: public BlockHandler
{
	friend void testOneDimHandler();
	friend void testFieldDataHandler();
	public:

		OneDimHandler( const DinfoBase* dinfo, bool isGlobal, 
			unsigned int size );

		OneDimHandler( const OneDimHandler* other );

		~OneDimHandler();

		////////////////////////////////////////////////////////////
		// Information functions
		////////////////////////////////////////////////////////////

		/// data() is defined in BlockHandler
		/// totalEntries() is defined in BlockHandler
		/// localEntries() is defined in BlockHandler

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const
		{
			return 1;
		}

		unsigned int sizeOfDim( unsigned int dim ) const;

		vector< unsigned int > dims() const;

		/// isDataHere() is defined in BlockHandler
		/// isAllocated() is defined in BlockHandler

		////////////////////////////////////////////////////////////////
		// load balancing functions defined in BlockHandler
		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////
		// Process and foreach functions defined in BlockHandler
		////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////
		// Data Reallocation functions
		////////////////////////////////////////////////////////////////
		/// globalize() is defined in BlockHandler
		/// unGlobalize() is defined in BlockHandler

		/**
		 * Make a single identity copy, doing appropriate node 
		 * partitioning if toGlobal is false.
		 */
		DataHandler* copy( bool toGlobal, unsigned int n ) const;

		DataHandler* copyUsingNewDinfo( const DinfoBase* dinfo) const;

		bool resize( unsigned int dimension, unsigned int size );

		/// assign() is defined in BlockHandler

	private:
};

#endif	// _ONE_DIM_HANDLER_H


