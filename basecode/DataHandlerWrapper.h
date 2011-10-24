/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _DATA_HANDLER_WRAPPER_H
#define _DATA_HANDLER_WRAPPER_H

/**
 * This class wraps a DataHandler pointer in all respects. It allows the
 * DataHandler to be reused without fear of being deleted. The parent
 * data handler is never modified, though the data contents may be.
 */
class DataHandlerWrapper: public DataHandler
{
	public:
		DataHandlerWrapper( const DataHandler* parentHandler,
			const DataHandler* origHandler );

		~DataHandlerWrapper();

		////////////////////////////////////////////////////////////
		// Information functions
		////////////////////////////////////////////////////////////

		/// Returns data on specified index
		char* data( DataId index ) const;

		/**
		 * Returns the number of data entries on local node
		 */
		unsigned int localEntries() const;

		bool isDataHere( DataId index ) const;

		bool isAllocated() const;

		unsigned int linearIndex( DataId di ) const;

		vector< vector< unsigned int > > pathIndices( DataId di ) const;

		DataId pathDataId( const vector< vector< unsigned int > >& indices )
			const;
		////////////////////////////////////////////////////////////////
		// load balancing functions
		////////////////////////////////////////////////////////////////
		bool innerNodeBalance( unsigned int size,
			unsigned int myNode, unsigned int numNodes );

		bool execThread( ThreadId thread, DataId di ) const;

		////////////////////////////////////////////////////////////////
		// Process function
		////////////////////////////////////////////////////////////////
		/**
		 * calls process on data, using threading info from the ProcInfo
		 */
		void process( const ProcInfo* p, Element* e, FuncId fid ) const;

		void forall( const OpFunc* f, Element* e, const Qinfo* q,
			const double* arg, unsigned int argSize, unsigned int numArgs )
			const;

		unsigned int getAllData( vector< char* >& data ) const;

		////////////////////////////////////////////////////////////////
		// Data Reallocation functions
		////////////////////////////////////////////////////////////////

		void globalize( const char* data, unsigned int size );
		void unGlobalize();

		/**
		 * Make a single identity copy, doing appropriate node 
		 * partitioning if toGlobal is false.
		 */
		DataHandler* copy( unsigned short newParentDepth,
			unsigned short copyRootDepth,
			bool toGlobal, unsigned int n ) const;

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
		const DataHandler* parent_;
};

#endif // _DATA_HANDLER_WRAPPER_H
