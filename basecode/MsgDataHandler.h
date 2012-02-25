/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MSG_DATA_HANDLER_H
#define _MSG_DATA_HANDLER_H

/**
 * This class manages the Element interface for msgs.
 */
class MsgDataHandler: public DataHandler
{
	public:
		MsgDataHandler( const DinfoBase* dinfo, 
			const vector< DimInfo >& dims, unsigned short pathDepth,
			bool isGlobal );

		MsgDataHandler( const MsgDataHandler* other );

		~MsgDataHandler();

		////////////////////////////////////////////////////////////
		// Class-specific functions for managing the mids.
		////////////////////////////////////////////////////////////
		/// Add a MsgId to the list of them stored on this MsgDataHandler
		void addMid( MsgId mid );

		/**
		 * Remove specified MsgId from the list of them stored on 
		 * this MsgDataHandler. Returns true if it was found and removed,
		 * false if not found.
		 */
		bool dropMid( MsgId mid );

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
		// Process and forall functions
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

		bool resize( unsigned int dimension, unsigned int size );

		void assign( const char* orig, unsigned int numOrig );

	protected:
		char* data_;
	private:
		vector< MsgId > mids_;
};

#endif // _MSG_DATA_HANDLER_H
