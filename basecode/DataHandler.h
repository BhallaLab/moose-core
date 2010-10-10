/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _DATA_HANDLER_H
#define _DATA_HANDLER_H

/**
 * This class manages the data part of Elements. This includes
 * allocation and freeing, lookup, and decomposition between nodes.
 * It is a virtual base class.
 * It works closely with DataId, whose role is to provide a lookup
 * into the contents of the DataHandler.
 */
class DataHandler
{
	public:
		DataHandler( const DinfoBase* dinfo )
			: dinfo_( dinfo )
		{;}

		virtual ~DataHandler()
		{;}

		/**
		 * Converts handler to its global version, where the same data is
		 * present on all nodes. Ignored if already global.
		 * returns true on success.
		 */
		virtual DataHandler* globalize() const = 0;

		/**
		 * Converts handler to its local version, where the data is 
		 * partitioned between nodes based on the load balancing policy.
		 * Returns true on success.
		 */
		virtual DataHandler* unGlobalize() const = 0;

		/**
		 * Fills in data from other nodes, used when globalizing
		 * and potentially when moving data between nodes.
		 * Data Handler must already have been built and allocated.
		 */
		virtual void assimilateData( const char* data, 
			unsigned int begin, unsigned int end ) = 0;

		/**
		 * Determines how to decompose data among nodes for specified size
		 * Returns true if there is a change from the current configuration
		 * Does NOT touch actual allocation.
		 */
		virtual bool nodeBalance( unsigned int size ) = 0;

		/**
		 * For copy we won't worry about global status. 
		 * Instead define function: globalize above.
		 * Version 1: Just copy as original
		 */
		virtual DataHandler* copy() const = 0;

		/**
		 * Version 2: Copy same dimensions but different # of entries.
		 * The copySize is the total number of targets, 
		 * here we need to figure out
		 * what belongs on the current node.
		 */
		virtual DataHandler* copyExpand( unsigned int copySize ) const = 0;

		/**
		 * Version 3: Add another dimension when doing the copy.
		 * Here too we figure out what is to be on current node for copy.
		 */
		virtual DataHandler* copyToNewDim( unsigned int newDimSize ) const = 0;

		/**
		 * Returns the data on the specified index.
		 * Returns 0 if data not present on current node on specified index
		 */
		virtual char* data( DataId index ) const = 0;

		/**
		 * Goes through all the data resident on the local node, using
		 * threading info from the ProcInfo
		 */
		virtual void process( const ProcInfo* p, Element* e, FuncId fid ) const = 0;

		/**
		 * Returns the number of data entries in the whole object,
		 * not just what is present here on this node. If we have arrays of
		 * type X nested in an array of type Y, then returns total # of X.
		 * If we have a 2-D array of type X, returns total # of X.
		 * Note that if we have a ragged array it still treats it as an
		 * N-dimension array cuboid, and reports the product of all sides
		 * rather than the sum of individual array counts.
		 */
		virtual unsigned int totalEntries() const = 0;

		/**
		 * Returns the number of dimensions of the data.
		 * 0 if there is a single entry.
		 * 1 if it is a 1-D array
		 * 2 if it is a 2-D array or nesting of arrays of X in array of Y.
		 * and so on.
		 */
		virtual unsigned int numDimensions() const = 0;

		/**
		 * Returns the number of data entries at any index.
		 */
		virtual unsigned int sizeOfDim( unsigned int dim ) const = 0;

		/**
		 * Reallocates data. Data not preserved unless same # of dims
		 * Returns 0 if it cannot handle the requested allocation.
		 */
		virtual bool resize( vector< unsigned int > dims );

		 /**
		  * Returns vector of dimensions.
		  */
		 virtual vector< unsigned int > dims() const = 0;

		/**
		 * Returns true if the node decomposition has the data on the
		 * current node
		 */
		virtual bool isDataHere( DataId index ) const = 0;

		/**
		 * True if data has been allocated
		 */
		virtual bool isAllocated() const = 0;

		/**
		 * True if the data is global on all nodes
		 */
		virtual bool isGlobal() const = 0;

		/**
		 * This class handles going through all data objects in turn.
		 */
		class iterator {
			public: 
				iterator( const iterator& other )
					: dh_( other.dh_ ), index_( other.index_ )
				{;}

				iterator( const DataHandler* dh, unsigned int index )
					: dh_( dh ), index_( index )
				{;}

				unsigned int index() const {
					return index_;
				}

				// This does prefix increment.
				iterator operator++() {
					index_ = dh_->nextIndex( index_ );
					return *this;
				}

				// Bizarre C++ convention to tell it to do postfix increment
				iterator operator++( int ) {
					index_ = dh_->nextIndex( index_ );
					return *this;
				}

				bool operator==( const iterator& other ) const
				{
					return ( index_ == other.index_ && dh_ == other.dh_ );
				}

				bool operator!=( const iterator& other ) const
				{
					return ( index_ != other.index_ || dh_ != other.dh_ );
				}

				char* operator* () const
				{
					return dh_->data( index_ );
				}

			private:
				const DataHandler* dh_;
				unsigned int index_;
		};

		virtual iterator begin() const = 0;
		virtual iterator end() const = 0;

		const DinfoBase* dinfo() const
		{
			return dinfo_;
		}

		/**
		 * Assigns block of data, which is a slice of 0 to n dimensions,
		 * in a data handler of n dimensions. The block of data is a 
		 * contiguous block in memory, and contains numEntries objects.
		 * The target data block is at the dimNum dimension, and dimIndex
		 * specifies the slice of data to replace.
		 * Here the numEntries must equal the size of the entire slice of
		 * data being replaced. In principle we could do treadmilling or
		 * truncation to do the assignment, but let's start with something
		 * simple that can be checked.
		 * Note that the numEntries is the full slice, not the sub=part
		 * that may be used on any given node. For example, if we have
		 * 4 nodes and use every 4th location on any given node, the
		 * data block passed in and numEntries nevertheless refer to the
		 * entire data block.
		 * Does not do any memory allocation.
		 * Returns true if assignment OK, which means that numEntries was
		 * correct.
		 */
		virtual bool setDataBlock( 
			const char* data, unsigned int numEntries, 
			unsigned int dimNum, unsigned int dimIndex );


	protected:
		/**
		 * Assigns the data field and indicates how many total entries
		 * are present in the incoming data. Does NOT touch allocation.
		 * If numData > num alloced, then fills in numAlloced.
		 * Else fills in numData.
		virtual void setData( char* data, unsigned int numData ) = 0; 
		 */


		/**
		 * Used to march through the entries in this DataHandler
		 */
		virtual unsigned int nextIndex( unsigned int index ) const = 0;

	private:
		const DinfoBase* dinfo_;
};

#endif	// _DATA_HANDLER_H
