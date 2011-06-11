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
		DataHandler( const DinfoBase* dinfo );

		/**
		 * The respective DataHandler subclasses should also provide
		 * a constructor that takes another DataHandler ptr of the
		 * same type as an argument. This should allocate a copy
		 * of the original data
		 */


		/**
		 * The destructor has to destroy the data contents.
		 */
		virtual ~DataHandler();

		/**
		 * Converts handler to its global version, where the same data is
		 * present on all nodes. Ignored if already global.
		 * Returns a newly allocated DataHandler, the old one remains.
		 */
		virtual DataHandler* globalize() const = 0;

		/**
		 * Converts handler to its local version, where the data is 
		 * partitioned between nodes based on the load balancing policy.
		 * Returns a newly allocated DataHandler, the old one remains.
		 */
		virtual DataHandler* unGlobalize() const = 0;

		/**
		 * Determines how to decompose data among nodes for specified size
		 * Returns true if there is a change from the current configuration
		 * Does NOT touch actual allocation.
		 * This form of the function is just a front-end for the inner
		 * function, as this talks to the Shell object to find node info.
		 */
		bool nodeBalance( unsigned int size );

		/**
		 * Determines how to decompose data among nodes for specified size
		 * Returns true if there is a change from the current configuration
		 * Does NOT touch actual allocation.
		 * This inner function is self-contained and is independent of the
		 * Shell. Each subclass of DataHandler has to supply this.
		 */
		virtual bool innerNodeBalance( unsigned int size, 
			unsigned int myNode, unsigned int numNodes ) = 0;

		/**
		 * Copies to another DataHandler. If the source is global and
		 * the dest is non-global, it does a selective copy of data
		 * contents only for the entries on current node.
		 * Otherwise it copies everthing over.
		 */
		virtual DataHandler* copy( bool toGlobal) const = 0;

		/**
		 * Copies DataHandler dimensions but uses new Dinfo to allocate
		 * contents and handle new data. Useful when making zombie copies.
		 * This does not need the toGlobal flag as the zombies are always
		 * located identically to the original.
		 */
		virtual DataHandler* copyUsingNewDinfo( 
			const DinfoBase* dinfo ) const = 0;

		/**
		 * Version 2: Copy same dimensions but different # of entries.
		 * The copySize is the total number of targets, 
		 * here we need to figure out
		 * what belongs on the current node.
		 */
		virtual DataHandler* copyExpand( 
			unsigned int copySize, bool toGlobal ) const = 0;

		/**
		 * Version 3: Add another dimension when doing the copy.
		 * Here too we figure out what is to be on current node for copy.
		 */
		virtual DataHandler* copyToNewDim( 
			unsigned int newDimSize, bool toGlobal ) const = 0;

		/**
		 * Returns the data on the specified index.
		 * Returns 0 if data not present on current node on specified index
		 */
		virtual char* data( DataId index ) const = 0;

		/**
		 * Returns DataHandler for the parent object. This applies to 
		 * the FieldDataHandler; in all other cases it returns self.
		 */
		virtual const DataHandler* parentDataHandler() const;

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
		 * Returns the actual number of data entries used on the 
		 * object, on current node. Here if we have a ragged array then
		 * it only counts the sum of the individual array counts
		 * So, adding localEntries over all nodes does not necessarily
		 * give totalEntries.
		 */
		virtual unsigned int localEntries() const = 0;

		/**
		 * Returns a single number corresponding to the DataId.
		 * Usually it is just the data part of the DataId, but it gets
		 * interesting for the FieldDataHandler.
		 */
		virtual unsigned int linearIndex( const DataId& d ) const;

		/**
		 * Returns the DataId corresponding to a single index.
		 */
		virtual DataId dataId( unsigned int linearIndex) const;

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
		 * If 'dim' is greater than the number of dimensions, returns zero.
		 */
		virtual unsigned int sizeOfDim( unsigned int dim ) const = 0;

		/**
		 * Reallocates data. Data not preserved unless same # of dims
		 * Returns 0 if it cannot handle the requested allocation.
		 */
		virtual bool resize( vector< unsigned int > dims ) = 0;

		/**
		 * Reallocates field data. This uses the object-specific resize
		 * function, so we don't know what happens to old data.
		 * Most objects dont' care about this. It is used by the
		 * FieldDataHandler.
		 */
		virtual void setFieldArraySize(
			unsigned int objectIndex, unsigned int size );

		/**
		 * Looks up size of field data. Most objects don't have applicable
		 * fields, and return zero. The FieldDataHandlers use it.
		 */
		virtual unsigned int getFieldArraySize( 
			unsigned int objectIndex ) const;

		/**
		 * Access functions for the FieldDimension. Applicable for 
		 * FieldDataHandlers, which typically manage a ragged array of 
		 * field vectors, belonging to each object in the data array.
		 * The FieldDimension provides a consistent range for indexing
		 * into this ragged array, and it must be bigger than any of the
		 * individual object array sizes.
		 * non FieldDataHandlers return 0 as the dimension and ignore
		 * the 'setFieldDimension' call.
		 */
		virtual void setFieldDimension( unsigned int size );
		virtual unsigned int getFieldDimension() const;


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
					: dh_( other.dh_ ), index_( other.index_ ),
					linearIndex_( other.linearIndex_ )
				{;}

				iterator( const DataHandler* dh, const DataId& index,
					unsigned int linearIndex )
					: 
						dh_( dh ), 
						index_( index ),
						linearIndex_( linearIndex )
				{;}

				/**
				 * This is the DataId index of this entry, completely
				 * specifying both the object index and the field index.
				 */
				DataId index() const {
					return index_;
				}
				
				/**
				 * This provides the linear index of this entry.
				 * If one were to iterate through all the entries and
				 * count how many have passed, this would be the linear
				 * index. With the exception of the FieldDataHandler
				 * and other possible ragged array structures. In these
				 * cases the linear index jumps.
				 */
				unsigned int linearIndex() const {
					return linearIndex_;
				}

				// This does prefix increment.
				iterator operator++() {
					dh_->nextIndex( index_, linearIndex_ );
					return *this;
				}

				// Bizarre C++ convention to tell it to do postfix increment
				iterator operator++( int ) {
					dh_->nextIndex( index_, linearIndex_ );
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

				/**
				 * This field manages the current DataId handled in the
				 * iterator.
				 */
				DataId index_;

				/**
				 * This field tracks the current linear index, which is
				 * an unrolled integer index
				 */
				unsigned int linearIndex_;
		};

		/**
		 * Iterator start point for going through all objects in the 
		 * DataHandler.
		 */
		virtual iterator begin() const = 0;

		/**
		 * Iterator end point for going through all objects in the 
		 * DataHandler.
		 */
		virtual iterator end() const = 0;

		const DinfoBase* dinfo() const
		{
			return dinfo_;
		}

		/**
		 * Assigns block of data, which is a slice of 0 to n dimensions,
		 * in a data handler of n dimensions. The block of data is a 
		 * contiguous block in memory, and contains 'numData' objects in
		 * the range starting at 'startIndex'.
		 * The vector form of the function first converts the index
		 * into the linear form.
		 * Returns true if the assignment succeeds. In other words,
		 * numData + startIndex should be less than size.
		 * Does not do any memory allocation.
		 * If the Handler is doing node decomposition, then this function
		 * takes the full data array, and picks out of it only those
		 * fields that belong on current node.
		 * Some Handlers handle ragged arrays. The DataBlock is defined
		 * to be a cuboid, so it is up to the Handler to select the
		 * appropriate subset of the cuboid to use.
		 */
		virtual bool setDataBlock( 
			const char* data, unsigned int numData,
			const vector< unsigned int >& startIndex ) const = 0;

		virtual bool setDataBlock( 
			const char* data, unsigned int numData,
			DataId startIndex ) const = 0;

		/**
		 * Used to march through the entries in this DataHandler
		 */
		virtual void nextIndex( 
			DataId& index, unsigned int& linearIndex ) const = 0;

	private:
		const DinfoBase* dinfo_;
};

#endif	// _DATA_HANDLER_H
