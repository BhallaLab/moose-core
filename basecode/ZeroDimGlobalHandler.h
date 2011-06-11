/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZERO_DIM_GLOBAL_HANDLER_H
#define _ZERO_DIM_GLOBAL_HANDLER_H

/**
 * This class manages the data part of Elements having just one
 * data entry. This variant repeats the same data on all nodes,
 * for things like Shell and Clock, and sometimes prototypes.
 */
class ZeroDimGlobalHandler: public DataHandler
{
	public:
		ZeroDimGlobalHandler( const DinfoBase* dinfo );

		/// Special constructor used in Cinfo::makeCinfoElements
		ZeroDimGlobalHandler( const DinfoBase* dinfo, char* data );

		ZeroDimGlobalHandler( const ZeroDimGlobalHandler* other );

		~ZeroDimGlobalHandler();

		DataHandler* globalize() const;
		DataHandler* unGlobalize() const;

		bool innerNodeBalance( unsigned int size,
			unsigned int myNode, unsigned int numNodes );

		/**
		 * Make a single copy
		 */
		DataHandler* copy( bool toGlobal ) const;

		/**
		 * Make a single copy with same dimensions, using a different Dinfo
		 */
		DataHandler* copyUsingNewDinfo( const DinfoBase* dinfo ) const;

		DataHandler* copyExpand( unsigned int copySize, bool toGlobal )
			const;

		DataHandler* copyToNewDim( unsigned int newDimSize, bool toGlobal )
			const;

		/**
		 * Returns the data on the specified index.
		 */
		char* data( DataId index ) const {
			return data_;
		}

		/**
		 * calls process on data, using threading info from the ProcInfo
		 */
		void process( const ProcInfo* p, Element* e, FuncId fid ) const;

		/**
		 * Returns the number of data entries.
		 */
		unsigned int totalEntries() const {
			return 1;
		}

		/**
		 * Returns the number of data entries on local node.
		 */
		unsigned int localEntries() const {
			return 1;
		}

		/**
		 * Returns a number corresponding to DataId. Since this DataHandler
		 * does not permit any index more than zero, we return zero.
		 */
		unsigned int linearIndex( const DataId& d ) const {
			return 0;
		}

		/**
		 * Returns the DataId corresponding to the specified linear index.
		 * Again, we return zero since that is the only legal value.
		 */
		DataId dataId( unsigned int linearIndex) const {
			return DataId( 0 );
		}

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const {
			return 0;
		}

		unsigned int sizeOfDim( unsigned int dim ) const
		{
			return ( dim == 0 );
		}

		bool resize( vector< unsigned int > dims );

		vector< unsigned int > dims() const;

		/**
		 * Returns true always: it is a global.
		 */
		bool isDataHere( DataId index ) const {
			return 1;
		}

		bool isAllocated() const;

		bool isGlobal() const
		{
			return 1;
		}

		iterator begin() const;

		iterator end() const;

		/**
		 * Assigns a block of data at the specified location.
		 * Here the numData has to be 1 and the startIndex
		 * has to be 0. Returns true if all this is OK.
		 */
		bool setDataBlock( const char* data, unsigned int numData,
			const vector< unsigned int >& startIndex ) const;
		bool setDataBlock( const char* data, unsigned int numData,
			DataId startIndex ) const;

		void nextIndex( DataId& index, unsigned int& linearIndex ) const;

	protected:
		char* data_;
	private:
};

#endif // _ZERO_DIM_GLOBAL_HANDLER_H
