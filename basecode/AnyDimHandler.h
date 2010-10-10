/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ANY_DIM_HANDLER_H
#define _ANY_DIM_HANDLER_H

/**
 * This class manages the data part of Elements. It handles arrays of
 * any dimension.
 */
class AnyDimHandler: public AnyDimGlobalHandler
{
	public:
		AnyDimHandler( const DinfoBase* dinfo );
		AnyDimHandler( const AnyDimHandler* other );
		~AnyDimHandler();

		/**
		 * Converts handler to its global version, where the same data is
		 * present on all nodes. Ignored if already global.
		 * returns true on success.
		 */
		DataHandler* globalize() const;

		/**
		 * Converts handler to its local version, where the data is 
		 * partitioned between nodes based on the load balancing policy.
		 * This is basically a matter of figuring out data range and
		 * deleting other stuff.
		 * Returns true on success.
		 */
		DataHandler* unGlobalize() const;

		/**
		 * Incorporates provided data into already allocated space
		 */
		void assimilateData( const char* data, 
			unsigned int begin, unsigned int end );

		/**
		 * Determines how to decompose data among nodes for specified size
		 * Returns true if there is a change from the current configuration
		 */
		bool nodeBalance( unsigned int size );

		/**
		 * For copy we won't worry about global status. 
		 * Instead define function: globalize above.
		 * Version 1: Just copy as original
		 */
		DataHandler* copy() const;

		/**
		 * Version 2: Copy same dimensions but different # of entries.
		 * The copySize is the total number of targets, 
		 * here we need to figure out
		 * what belongs on the current node.
		 */
		DataHandler* copyExpand( unsigned int copySize ) const;

		/**
		 * Version 3: Add another dimension when doing the copy.
		 * Here too we figure out what is to be on current node for copy.
		 */
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
		 * Reallocates data. Data not preserved unless same # of dims
		 */
		bool resize( vector< unsigned int > dims );

		/**
		 * Returns true if the node decomposition has the data on the
		 * current node
		 */
		bool isDataHere( DataId index ) const;

		/**
		 * Returns true if data is allocated.
		 */
		bool isAllocated() const;

		/**
		 * Returns true if data is global. Not so here.
		 */
		bool isGlobal() const;

		/**
		 * Iterator to start of data
		 */
		iterator begin() const;

		/**
		 * Iterator to start of data
		 */
		iterator end() const;

		/**
		 * Assigns a block of data at the specified dimension and index in
		 * that dimension. Returns true if all OK. No allocation.
		 */
		bool setDataBlock( const char* data, unsigned int numEntries, 
			unsigned int dimNum, unsigned int dimIndex );

	protected:

		unsigned int nextIndex( unsigned int index ) const;

	private:
		unsigned int start_;	// Starting index of data, used in MPI.
		unsigned int end_;	// Starting index of data, used in MPI.
};

#endif	// _ANY_DIM_HANDLER_H
