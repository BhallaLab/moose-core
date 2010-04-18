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
		 * Returns the data on the specified index.
		 * Returns 0 if data not present on current node on specified index
		 */
		virtual char* data( DataId index ) const = 0;

		/**
		 * Goes through all the data resident on the local node, using
		 * threading info from the ProcInfo
		 */
		virtual void process( const ProcInfo* p, Element* e ) const = 0;

		/**
		 * Returns the data at one level up of indexing, in the special
		 * case where we have arrays of type X nested in an array of
		 * type Y. This function returns the entry of type Y.
		 * For a synapse on an IntFire, would
		 * return the appropriate IntFire, rather than the synapse.
		 * In other cases returns the data at the first index of DataId.
		 *
		 * Returns 0 if data not found at index.
		 */
		virtual char* data1( DataId index ) const = 0;


		/**
		 * Returns the number of data entries in the whole message,
		 * not just what is present here on this node. If we have arrays of
		 * type X nested in an array of type Y, then returns total # of X.
		 * If we have a 2-D array of type X, returns total # of X.
		 * If we have a vector of vectors of type X, returns total # of X.
		 */
		virtual unsigned int numData() const = 0;

		/**
		 * Returns the number of data entries at index 1.
		 * For regular Elements this is identical to numData.
		 * If we have 2-D arrays or vectors of vectors, this is the 
		 * number of entries at the first index.
		 * If we have arrays of type X nested in an array of type Y, this
		 * is the number of Y.
		 */
		virtual unsigned int numData1() const = 0;

		/**
		 * Returns the number of data entries at index 2, if present.
		 * For regular Elements and 1-D arrays this is always 1.
		 * If we have 2-D arrays or vectors of vectors, this looks up
		 * the first index and returns the # of array entries in it.
		 * If we have arrays of type X nested in an array of type Y, this
		 * looks up Y[ index ] and returns the number of entries of X
		 * within it.
		 */
		 virtual unsigned int numData2( unsigned int index1 ) const = 0;

		/**
		 * Returns the number of dimensions of the data.
		 * 0 if there is a single entry.
		 * 1 if it is a 1-D array
		 * 2 if it is a 2-D array or nesting of arrays of X in array of Y.
		 * and so on.
		 */
		virtual unsigned int numDimensions() const = 0;

		/**
		 * Assigns the # of entries in dimension 1.
		 * Ignores if 0 dimensions.
		 * Does not do any actual allocation: that waits till
		 * the 'allocate' function.
		 */
		virtual void setNumData1( unsigned int size ) = 0;

		/**
		 * Assigns the sizes of all array field entries at once.
		 * Ignore if 1 or 0 dimensions.
		 * The 'sizes' vector must be of length numData1.
		 * In a FieldElement we can assign different array sizes
		 * for each entry in the Element.
		 * In a 2-D array we can do the equivalent.
		 * Note that a single Element may have more than one array field.
		 * However, each FieldElement instance will refer to just one of
		 * these array fields, so there is no ambiguity.
		 */
		virtual void setNumData2( const vector< unsigned int >& sizes ) 
			 = 0;

		/**
		 * Looks up the sizes of all array field entries at once.
		 * Ignored for 0 and 1 dimension Elements. 
		 * Note that a single Element may have more than one array field.
		 * However, each FieldElement instance will refer to just one of
		 * these array fields, so there is no ambiguity.
		 */
		virtual void getNumData2( vector< unsigned int >& sizes ) const
			= 0;

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
		 * Allocates the data.
		 */
		virtual void allocate() = 0;

		/**
		 * True if the data is global on all nodes
		 */
		virtual bool isGlobal() const = 0;

		typedef unsigned int iterator; // the ++i and i++ operators are already known.

		virtual iterator begin() const = 0;
		virtual iterator end() const = 0;

	protected:
		const DinfoBase* dinfo() const
		{
			return dinfo_;
		}

		/**
		 * Used to iterate over indices managed by DataHandler
		 * to find those on the current node. A foreach would be nice.
		class iterator {
			public:
				iterator( unsigned int start, const DataHandler *d )
					: i( 0 )
				{;}

				iterator ++operator() {
					return ++i;
				}

				iterator 
			
			private:
				unsigned int i;
		};
		 */

	private:
		const DinfoBase* dinfo_;
};

#endif	// _DATA_HANDLER_H
