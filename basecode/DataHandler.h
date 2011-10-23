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

typedef struct {
	unsigned int size;
	unsigned short depth;
	bool isRagged;
} DimInfo;

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
		DataHandler( const DinfoBase* dinfo, const vector< DimInfo >& dims,
			unsigned short pathDepth, bool isGlobal );

		/**
		 * The respective DataHandler subclasses should also provide
		 * a constructor that takes another DataHandler ptr of the
		 * same type as an argument. This should allocate a copy
		 * of the original data
		 * This version copies the dinfo, dims, depth and isGlobal fields.
		 */
		 DataHandler( const DataHandler* orig );


		/**
		 * The destructor has to destroy the data contents.
		 */
		virtual ~DataHandler();

/////////////////////////////////////////////////////////////////////////
// Information functions.
/////////////////////////////////////////////////////////////////////////
		const DinfoBase* dinfo() const
		{
			return dinfo_;
		}

		/**
		 * Returns the data on the specified index.
		 * Returns 0 if data not present on current node on specified index
		 */
		virtual char* data( DataId index ) const = 0;

		/**
		 * Returns the data of the parent DataHandler of the specified index
		 * Returns 0 if data not present on current node on specified index,
		 * or if the DataId does not refer to a FieldDataHandler.
		 */
		virtual char* parentData( DataId index ) const;

		/**
		 * Returns DataHandler for the parent object. This applies to 
		 * the FieldDataHandler; in all other cases it returns self.
		 */
		virtual const DataHandler* parentDataHandler() const;

		/**
		 * Returns the number of data entries in the whole object,
		 * not just what is present here on this node. If we have arrays of
		 * type X nested in an array of type Y, then returns total # of X.
		 * If we have a 2-D array of type X, returns total # of X.
		 * Note that if we have a ragged array it still treats it as an
		 * N-dimension array cuboid, and reports the product of all sides
		 * rather than the sum of individual array counts.
		 */
		unsigned int totalEntries() const;

		/**
		 * Returns the actual number of data entries used on the 
		 * object, on current node. Here if we have a ragged array then
		 * it only counts the sum of the individual array counts
		 * So, adding localEntries over all nodes does not necessarily
		 * give totalEntries.
		 */
		virtual unsigned int localEntries() const = 0;

		/**
		 * Returns the number of dimensions of the data.
		 * 0 if there is a single entry.
		 * 1 if it is a 1-D array
		 * 2 if it is a 2-D array or nesting of arrays of X in array of Y.
		 * and so on.
		 */
		unsigned int numDimensions() const;

		/**
		 * Returns the depth of the current DataHandler in the element
		 * tree. Root is zero.
		 */
		unsigned short pathDepth() const;

		/**
		 * Returns the number of data entries at any index.
		 * If 'dim' is greater than the number of dimensions, returns zero.
		 */
		unsigned int sizeOfDim( unsigned int dim ) const;

		 /**
		  * Returns vector of dimensions.
		  * Lowest index is closest to root. 0 is root.
		  * Highest index varies fastest.
		  */
		 const vector< DimInfo >& dims() const;

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
		 * True if the data is global on all nodes.
		 * This is overridden in rare cases, like FieldDataHandler,
		 * which look up their parent.
		 */
		virtual bool isGlobal() const;

		/**
		 * Returns the # of entries in the field on this object, should
		 * it indeed be a FieldDataHandler. Otherwise returns 0.
		 * The index i specifies which parent object to refer to.
		 * If I had 100 synchans, each with assorted synapses as Fields,
		 * the index i would identify the parent synchan.
		 */
		virtual unsigned int getFieldArraySize( unsigned int i ) const;

		/**
		 * Returns the mask for the field portion of the DataId that works
		 * with this DataHandler. In most cases this is zero, in 
		 * FieldDataHandlerBase it is a bitmask.
		 */
		virtual unsigned int fieldMask() const;

		/**
		 * Returns the linearIndex equivalent of the dataId. linearIndex
		 * is the index that would apply to a linear mapping of an
		 * n-dimensional cuboid indexed by the dimensions of the object.
		 * This differs from DataId::value because
		 * ragged arrays use bitmasks for sections of the DataId value,
		 * and the sizes of these bitmasks are typically larger than the
		 * dimension (maxFieldEntries) assigned to the ragged array,
		 * which would be an edge of the cuboid.
		 */
		virtual unsigned int linearIndex( DataId di ) const = 0;

/////////////////////////////////////////////////////////////////////////
// Path management
/////////////////////////////////////////////////////////////////////////

		/**
		 * Returns a vector of array indices for the specified DataId.
		 * At any level of the path we may have multidimensional arrays,
		 * so each level is represented by a vector.
		 * The size of the vector is pathDepth + 1, so that the root
		 * element (which is always a singleton) would be returned as an
		 * empty vector entry at index 0.
		 * If the return pathIndex vector is of size zero, it means that 
		 * there is a * mismatch and the requested DataId does not fit 
		 * in the current DataHandler.
		 * /foo[23]/bar[4][5][6]/zod would return the following
		 * vectors at each level:
		 *	root	foo		bar		zod
		 * 	{} 		{23} 	{4,5,6}	{}
		 */
		virtual vector< vector< unsigned int > > pathIndices( DataId di ) 
			const = 0;

		/**
		 * Returns the DataId for the specified vector of array indices.
		 * If the indices cannot fit in the current DataHandler, then it
		 * returns DataId::bad.
		 */
		virtual DataId pathDataId( 
			const vector< vector < unsigned int > >& indices ) const = 0;

		/**
		 * Moves the pathDepth to the specified level.
		 * If moving up, then it adds single-dimensions for the inserted
		 * levels at the root of the tree.
		 * If moving down, then it removes levels toward the root of the 
		 * three. It requires that the removed levels all have dimension
		 * size of one.
		 * Returns True on success.
		 * Returns False if the removed levels were non-unity.
		 */
		bool changeDepth( unsigned short newDepth );


/////////////////////////////////////////////////////////////////////////
// Load balancing functions
/////////////////////////////////////////////////////////////////////////

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
		 * Deprecated.
		 * Updates maxFieldEntries by checking all array sizes, and then
		 * does the necessary field mask adjustment.
		 * returns the updated maxFieldEntries.
		 * Most DataHandlers ignore this, we use a default that returns 0.
		 */
		virtual unsigned int syncFieldDim();

		/**
		 * True if the specified thread can operate on the specified DataId.
		 * This function combines attributes of IsDataHere with the
		 * threading check. However, it does NOT pass DataId::any.
		 */
		virtual bool execThread( ThreadId thread, DataId di ) const = 0;
/////////////////////////////////////////////////////////////////////////
// Function to go through entire dataset applying specified operations
// in a thread-safe manner.
/////////////////////////////////////////////////////////////////////////
		/**
		 * Goes through all the data resident on the local node, using
		 * threading info from the ProcInfo
		 */
		virtual void process( const ProcInfo* p, Element* e, FuncId fid )
			const = 0;

		/**
		 * This function iterates through all array entries assigned to
		 * this node and this thread, and calls the OpFunc f with the
		 * specified argument. The Qinfo provides the threadNum.
		 * The arg ptr increments on each cycle:
		 * if you don't want it to change, just set argIncrement to 0.
		 */
		virtual void foreach( const OpFunc* f, Element* e, const Qinfo* q,
			const double* arg, unsigned int argSize, unsigned int numArgs )
			const = 0;

		/**
		 * Fills up all the data entries into the provided vector of 
		 * chars. Return number found.
		 */
		virtual unsigned int getAllData( vector< char* >& data ) const = 0;

/////////////////////////////////////////////////////////////////////////
// Data reallocation and copy functions
/////////////////////////////////////////////////////////////////////////

		/**
		 * Sets state of DataHandler to be global, using the incoming data
		 * to fill it in. 
		 * These manipulations are done in-line, that is, the 
		 * DataHandler itself is retained.
		 */
		virtual void globalize( const char* data, unsigned int numEntries )
			= 0;

		/**
		 * Sets state of DataHandler to be local. Discards and frees data
		 * if needed.
		 * These manipulations are done in-line, that is, the 
		 * DataHandler itself is retained.
		 */
		virtual void unGlobalize() = 0;

		/**
		 * Copies self, creating a new DataHandler at the specified depth
		 * in the path tree, and taking only the portion of the tree
		 * starting from copyRootDepth.
		 * So if we copy with n=5 from
		 * /library/something/other[10] to /foo/bar/zod, 
		 * using /library/something as the copyRoot,
		 * we get
		 * /foo/bar/zod/something[5]/other[10]
		 * Here newParentDepth == 3 ( for zod )
		 * copyRootDepth == 2 (for something)
		 * Note that 'other' gets pushed up to depth 5.
		 * If n is 0 or 1 it just makes a duplicate of original, with the
		 * orginal dimensions.
		 * If n > 1 then it adds a dimension and replicates the original
		 * n times.
		 * If the source is global and
		 * the dest is non-global, it does a selective copy of data
		 * contents only for the entries on current node.
		 * Otherwise it copies everything over.
		 * It is the responsibility of the wrapping Shell command to ensure
		 * that the size of the dimensions matches the parent onto which
		 * the copy is being made.
		 *
		 * In due course need to extend so I can copy off a single entry.
		 */
		virtual DataHandler* copy( unsigned short newParentDepth,
			unsigned short copyRootDepth, bool toGlobal,
			unsigned int n ) const =0;

		/**
		 * Copies DataHandler dimensions but uses new Dinfo to allocate
		 * contents and handle new data. Useful when making zombie copies.
		 * This does not need the depth or the toGlobal flag,
		 * as the zombies are always
		 * located identically to the original.
		 */
		virtual DataHandler* copyUsingNewDinfo( 
			const DinfoBase* dinfo ) const = 0;

		/**
		 * Change the number of entries on the specified dimension. 
		 * Return true if OK.
		 * Does NOT change number of dimensions.
		 */
		virtual bool resize( unsigned int dimension, unsigned int numEntries) = 0;

		/**
		 * Copy over the original chunk of data to fill the entire data.
		 * If the size doesn't match, use tiling. 
		 * If the target is not global, use the tiling that would result
		 * in the consolidated data matching the global version.
		 * In other words, if we were to assign a chunk of data to a
		 * non-global target on all nodes, and then globalize the target,
		 * it should have the same contents as if we had assigned the
		 * chunk of data to a global.
		 */
		virtual void assign( const char* orig, unsigned int numOrig ) = 0;

	protected:
		 /**
		  * vector of dimensions.
		  * Lowest index is closest to root. 0 is root.
		  * Highest index varies fastest.
		  */
		vector< DimInfo > dims_;

		/**
		 * Specifies Depth on Element tree. Root is zero
		 */
		unsigned short pathDepth_;
		bool isGlobal_;
		unsigned int totalEntries_;

	private:
		const DinfoBase* dinfo_;
};

#endif	// _DATA_HANDLER_H
