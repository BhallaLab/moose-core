/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZERO_DIM_HANDLER_H
#define _ZERO_DIM_HANDLER_H

/**
 * This class manages the data part of Elements having just one
 * data entry.
 */
class ZeroDimHandler: public ZeroDimGlobalHandler
{
	public:
		ZeroDimHandler( const DinfoBase* dinfo );

		ZeroDimHandler( const ZeroDimHandler* other );

		~ZeroDimHandler();

		DataHandler* globalize() const;

		DataHandler* unGlobalize() const;

		/**
		 * Determines how to decompose data among nodes for specified size
		 * Returns true if there is a change from the current configuration
		 */
		bool innerNodeBalance( unsigned int size,
			unsigned int myNode, unsigned int numNodes );

		/**
		 * For copy we won't worry about global status. 
		 * Instead define function: globalize above.
		 * Version 1: Just copy as original
		 */
		DataHandler* copy( bool toGlobal ) const;

		/**
		 * Make a single copy with same dimensions, using a different Dinfo
		 */
		DataHandler* copyUsingNewDinfo( const DinfoBase* dinfo ) const;

		/**
		 * Version 2: Copy same dimensions but different # of entries.
		 * The copySize is the total number of targets, 
		 * here we need to figure out
		 * what belongs on the current node.
		 */
		DataHandler* copyExpand( unsigned int copySize, bool toGlobal ) const;

		/**
		 * Version 3: Add another dimension when doing the copy.
		 * Here too we figure out what is to be on current node for copy.
		 */
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
		 * Returns the number of data entries on local node.
		 * One only if current node is zero.
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
	private:
};

#endif // _ZERO_DIM_HANDLER_H
