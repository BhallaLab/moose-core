/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ONE_DIM_GLOBAL_HANDLER_H
#define _ONE_DIM_GLOBAL_HANDLER_H

/**
 * This class manages the data part of Elements. It handles a one-
 * dimensional array.
 */
class OneDimGlobalHandler: public DataHandler
{
	public:
		OneDimGlobalHandler( const DinfoBase* dinfo );
		OneDimGlobalHandler( const OneDimGlobalHandler* other );

		~OneDimGlobalHandler();

		DataHandler* globalize() const;

		DataHandler* unGlobalize() const;

		bool innerNodeBalance( unsigned int size,
			unsigned int myNode, unsigned int numNodes );

		DataHandler* copy( bool toGlobal ) const;

		/**
		 * Make a single copy with same dimensions, using a different Dinfo
		 */
		DataHandler* copyUsingNewDinfo( const DinfoBase* dinfo ) const;

		DataHandler* copyExpand( unsigned int copySize, bool toGlobal ) const;

		DataHandler* copyToNewDim( unsigned int newDimSize, bool toGlobal ) const;

		/**
		 * Returns the data on the specified index.
		 */
		char* data( DataId index ) const;

		/**
		 * calls process on data, using threading info from the ProcInfo,
		 * and internal info about node decomposition.
		 */
		void process( const ProcInfo* p, Element* e, FuncId fid ) const;

		/**
		 * Returns the number of data entries.
		 */
		unsigned int totalEntries() const;

		/**
		 * Returns the number of data entries on local node.
		 */
		unsigned int localEntries() const;

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const {
			return 1;
		}

		unsigned int sizeOfDim( unsigned int dim ) const;

		bool resize( vector< unsigned int > dims );

		/**
		 * Returns dimensions of this data.
		 */
		vector< unsigned int > dims() const;

		/**
		 * Returns true if the node decomposition has the data on the
		 * current node
		 */
		bool isDataHere( DataId index ) const;

		bool isAllocated() const;

		bool isGlobal() const;

		iterator begin() const {
			return iterator( this, 0, 0 );
		}

		iterator end() const {
			return iterator( this, numData_, numData_ );
		}

		/**
		 * Assigns a block of data at the specified location.
		 * Returns true if all OK. No allocation.
		 */
		bool setDataBlock( const char* data, unsigned int numData,
			const vector< unsigned int >& startIndex ) const;
		bool setDataBlock( const char* data, unsigned int numData,
			DataId startIndex ) const;

		void nextIndex( DataId& index, unsigned int& linearIndex ) const;

	protected:
		char* data_;
		unsigned int numData_;	// Number of data entries in the whole array
};

#endif	// _ONE_DIM_GLOBAL_HANDLER_H
