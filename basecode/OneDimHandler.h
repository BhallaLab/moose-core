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
class OneDimHandler: public DataHandler
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

		/// Returns data on specified index
		char* data( DataId index ) const;

		/**
		 * Returns the number of data entries.
		 */
		unsigned int totalEntries() const;

		/**
		 * Returns the number of data entries on local node
		 */
		unsigned int localEntries() const;

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const
		{
			return 1;
		}

		unsigned int sizeOfDim( unsigned int dim ) const;

		vector< unsigned int > dims() const;

		bool isDataHere( DataId index ) const;

		bool isAllocated() const;

		////////////////////////////////////////////////////////////////
		// load balancing functions
		////////////////////////////////////////////////////////////////
		bool innerNodeBalance( unsigned int size,
			unsigned int myNode, unsigned int numNodes );

		////////////////////////////////////////////////////////////////
		// Process function
		////////////////////////////////////////////////////////////////
		/**
		 * calls process on data, using threading info from the ProcInfo
		 */
		void process( const ProcInfo* p, Element* e, FuncId fid ) const;

		////////////////////////////////////////////////////////////////
		// Data Reallocation functions
		////////////////////////////////////////////////////////////////

		void globalize( const char* data, unsigned int size );
		void unGlobalize();

		/**
		 * Make a single identity copy, doing appropriate node 
		 * partitioning if toGlobal is false.
		 */
		DataHandler* copy( bool toGlobal, unsigned int n ) const;

		DataHandler* copyUsingNewDinfo( const DinfoBase* dinfo) const;

		DataHandler* addNewDimension( unsigned int size ) const;

		bool resize( unsigned int dimension, unsigned int size );

		void assign( const char* orig, unsigned int numOrig );

		////////////////////////////////////////////////////////////////
		// Iterator functions
		////////////////////////////////////////////////////////////////

		iterator begin( ThreadId threadNum ) const;

		iterator end( ThreadId threadNum ) const;

		void rolloverIncrement( iterator* i ) const;


	private:
		unsigned int totalEntries_; // Total number of entries on all nodes
		unsigned int start_;	// Starting index of data, used in MPI.
		unsigned int end_;	// Starting index of data, used in MPI.
		char* data_;
// 		unsigned int bitMask_; // Masks out the index.
};

#endif	// _ONE_DIM_HANDLER_H


