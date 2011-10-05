/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _TWO_DIM_HANDLER_H
#define _TWO_DIM_HANDLER_H

/**
 * This class manages the data part of Elements. It handles a one-
 * dimensional array.
 */
class TwoDimHandler: public DataHandler
{
	public:

		TwoDimHandler( const DinfoBase* dinfo, bool isGlobal, 
			unsigned int nx, unsigned int ny );

		TwoDimHandler( const TwoDimHandler* other );

		~TwoDimHandler();

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
			return 2;
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
		// Process and foreach functions: apply ops to all entries.
		////////////////////////////////////////////////////////////////
		/**
		 * calls process on data, using threading info from the ProcInfo
		 */
		void process( const ProcInfo* p, Element* e, FuncId fid ) const;

		void foreach( const OpFunc* f, Element* e, const Qinfo* q,
			const double* arg, unsigned int argIncrement ) const;

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

		/*
		////////////////////////////////////////////////////////////////
		// Iterator functions
		////////////////////////////////////////////////////////////////

		iterator begin( ThreadId threadNum ) const;

		iterator end( ThreadId threadNum ) const;

		void rolloverIncrement( iterator* i ) const;
		*/

	private:
		unsigned int nx_;
		unsigned int ny_;
		unsigned int start_;	// Starting index of data, used in MPI.
		unsigned int end_;	// Starting index of data, used in MPI.
		char* data_;
		/**
		 * Start index for each specified thread. The n+1 index is the 
		 * 'end' of the set for the nth thread. There is an extra index
		 * at the end of threadStart_ for the very end of the list.
		 */
		vector< unsigned int > threadStart_;
};

#endif	// _TWO_DIM_HANDLER_H


