/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZERO_DIM_HANDLER_H
#define _ZERO_DIM_HANDLER_H

/**
 * This class manages the data part of Elements having just one
 * data entry.
 */
class ZeroDimHandler: public DataHandler
{
	public:
		ZeroDimHandler( const DinfoBase* dinfo, bool isGlobal );

		/// Special constructor used in Cinfo::makeCinfoElements
		ZeroDimHandler( const DinfoBase* dinfo, char* data );

		ZeroDimHandler( const ZeroDimHandler* other );

		~ZeroDimHandler();

		////////////////////////////////////////////////////////////
		// Information functions
		////////////////////////////////////////////////////////////

		/// Returns data on specified index
		char* data( DataId index ) const;

		/**
		 * Returns the number of data entries.
		 */
		unsigned int totalEntries() const
		{
			return 1; // Somewhere, on some node, there is an entry.
		}

		/**
		 * Returns the number of data entries on local node
		 */
		unsigned int localEntries() const;

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const
		{
			return 0;
		}

		unsigned int sizeOfDim( unsigned int dim ) const
		{
			return ( dim == 0 );
		}

		vector< unsigned int > dims() const;

		bool isDataHere( DataId index ) const;

		bool isAllocated() const {
			return ( data_ != 0 );
		}

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

		/**
		 * Calls OpFunc f on all data entries, using threading info from 
		 * the Qinfo and the specified argument(s)
		 */
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

		bool resize( unsigned int dimension, unsigned int size );
		
		void assign( const char* orig, unsigned int numOrig );

		////////////////////////////////////////////////////////////////
		// Iterator functions
		////////////////////////////////////////////////////////////////
		/*

		iterator begin( ThreadId threadNum ) const;

		iterator end( ThreadId threadNum ) const;

		void rolloverIncrement( iterator* i ) const;
		*/


	private:
		bool isGlobal_;
		char* data_;
		/// Specifies which thread is allowed to call it
		ThreadId myThread_;
};

#endif // _ZERO_DIM_HANDLER_H
