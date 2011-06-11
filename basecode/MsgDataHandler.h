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
		MsgDataHandler( const DinfoBase* dinfo );

		MsgDataHandler( const MsgDataHandler* other );

		~MsgDataHandler();

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

		DataHandler* copyExpand( unsigned int copySize, bool toGlobal ) const;

		DataHandler* copyToNewDim( unsigned int newDimSize, bool toGlobal ) const;

		/**
		 * Returns the data on the specified index.
		 */
		char* data( DataId index ) const;

		/**
		 * calls process on data, using threading info from the ProcInfo
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

		// Inherited. Returns data part of DataId. 
		// unsigned int linearIndex( const DataId& d ) const;

		/**
		 * Returns the DataId corresponding to the specified linear index.
		 * Again, inherited. Returns the linearIndex as the data part.
		 */
		// DataId dataId( unsigned int linearIndex) const;

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const;

		unsigned int sizeOfDim( unsigned int dim ) const;

		bool resize( vector< unsigned int > dims );

		vector< unsigned int > dims() const;

		/**
		 * Returns true always: it is a global.
		 */
		bool isDataHere( DataId index ) const;

		/// Return true: msgs are always allocated.
		virtual bool isAllocated() const;

		bool isGlobal() const;

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

#endif // _MSG_DATA_HANDLER_H
