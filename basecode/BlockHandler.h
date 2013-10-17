/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _BLOCK_HANDLER_H
#define _BLOCK_HANDLER_H

/**
 * This class manages the data part of Elements. It handles a continguous
 * block of data. It is a base class for other classes with
 * different dimensions, but all of which store their data in contiguous
 * blocks.
 * This is also a pure virtual class, because some of the functions needed
 * by DataHandler are not defined here, but in the derived classes.
 */
class BlockHandler: public DataHandler
{
	public:

		BlockHandler( const DinfoBase* dinfo,
			const vector< DimInfo >& dims, 
			unsigned short pathDepth, bool isGlobal );

		BlockHandler( const BlockHandler* other );

		~BlockHandler();

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

		// numDimensions is defined in derived classes.
		// sizeOfDim is defined in derived classes.
		// dims is defined in derived classes.

		bool isDataHere( DataId index ) const;

		bool isAllocated() const;

		unsigned int linearIndex( DataId di ) const;

		////////////////////////////////////////////////////////////////
		// Path management
		////////////////////////////////////////////////////////////////

		/**
		 * Returns vector of array indices present at each level of the
		 * path, for the specified DataId
		 */
		vector< vector< unsigned int > > pathIndices( DataId di ) const;

		/**
		 * Returns DataId for the specified array of indices
		 */
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

		/// copy is defined in derived classes
		/// copyUsingNewDinfo is defined in derived classes
		/// resize is defined in derived classes

		void assign( const char* orig, unsigned int numOrig );

	protected:

		unsigned int start_;	/// Starting index of data, used in MPI.
		unsigned int end_;	/// Starting index of data, used in MPI.

		/// The actual data storage, cast into a char*
		char* data_;

	private:
};

#endif	// _BLOCK_HANDLER_H
