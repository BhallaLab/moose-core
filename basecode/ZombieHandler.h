/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_HANDLER_H
#define _ZOMBIE_HANDLER_H

/**
 * This class wraps a DataHandler pointer of a solver, to be used by
 * a Zombie class. It uses zero space per zombie entry. Unlike the generic
 * DataHandlerWrapper, it assumes that the Zombie has a single entry,
 * and the Zombie has an arbitrary number of entries. The parent solver
 * data handler is never modified, though the data contents may be.
 */
class ZombieHandler: public DataHandler
{
	public:
		ZombieHandler( const DataHandler* parentHandler,
			unsigned int start = 0, unsigned int end = 1 );

		~ZombieHandler();

		////////////////////////////////////////////////////////////////
		// Information functions
		////////////////////////////////////////////////////////////////

		/// Returns data on specified index
		char* data( DataId index ) const;

		/**
		 * Returns the number of data entries on local node
		 */
		unsigned int localEntries() const;

		bool isDataHere( DataId index ) const;

		virtual bool isAllocated() const;

		unsigned int linearIndex( DataId di ) const;

		////////////////////////////////////////////////////////////////
		// load balancing functions
		////////////////////////////////////////////////////////////////
		bool innerNodeBalance( unsigned int size,
			unsigned int myNode, unsigned int numNodes );

		bool execThread( ThreadId thread, DataId di ) const;
		////////////////////////////////////////////////////////////////
		// Process and foreach functions
		////////////////////////////////////////////////////////////////
		/**
		 * calls process on data, using threading info from the ProcInfo
		 */
		void process( const ProcInfo* p, Element* e, FuncId fid ) const;

		void foreach( const OpFunc* f, Element* e, const Qinfo* q,
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
		DataHandler* copy( unsigned short copyDepth, bool toGlobal,
			unsigned int n ) const;

		DataHandler* copyUsingNewDinfo( const DinfoBase* dinfo) const;

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
		unsigned int start_;
		unsigned int end_;
};

#endif // _ZOMBIE_HANDLER_H
