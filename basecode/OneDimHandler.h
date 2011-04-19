/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ONE_DIM_HANDLER_H
#define _ONE_DIM_HANDLER_H

/**
 * This class manages the data part of Elements. It handles a one-
 * dimensional array.
 */
class OneDimHandler: public OneDimGlobalHandler
{
	friend void testOneDimHandler();
	friend void testFieldDataHandler();
	public:
		OneDimHandler( const DinfoBase* dinfo );
		OneDimHandler( const OneDimHandler* other );

//		~OneDimHandler();

		DataHandler* globalize() const;

		DataHandler* unGlobalize() const;

		bool innerNodeBalance( unsigned int size, 
			unsigned int myNode, unsigned int numNodes );

		DataHandler* copy() const;

		/**
		 * Make a single copy with same dimensions, using a different Dinfo
		 */
		DataHandler* copyUsingNewDinfo( const DinfoBase* dinfo ) const;

		DataHandler* copyExpand( unsigned int copySize ) const;

		DataHandler* copyToNewDim( unsigned int newDimSize ) const;

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
		 * Returns the number of data entries on local node,
		 * overriding inherited version.
		 */
		unsigned int localEntries() const;

		bool resize( vector< unsigned int > dims );

		/**
		 * Returns true if the node decomposition has the data on the
		 * current node
		 */
		bool isDataHere( DataId index ) const;

		bool isAllocated() const;

		bool isGlobal() const
		{
			return 0;
		}

		iterator begin() const {
			return iterator( this, start_, start_ );
		}

		iterator end() const {
			return iterator( this, end_, end_ );
		}

		/**
		 * Assigns a block of data at the specified location.
		 * Returns true if all OK.
		 */
		bool setDataBlock( const char* data, unsigned int numData,
			const vector< unsigned int >& startIndex ) const;
		bool setDataBlock( const char* data, unsigned int numData,
			DataId startIndex ) const;

	protected:

	private:
		unsigned int start_;	// Starting index of data, used in MPI.
		unsigned int end_;	// Starting index of data, used in MPI.
};

#endif	// _ONE_DIM_HANDLER_H
