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
class AnyDimHandler: public DataHandler
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
		DataHandler* globalize();

		/**
		 * Converts handler to its local version, where the data is 
		 * partitioned between nodes based on the load balancing policy.
		 * This is basically a matter of figuring out data range and
		 * deleting other stuff.
		 * Returns true on success.
		 */
		DataHandler* unGlobalize();

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
		 * Returns the total number of data entries.
		 */
		unsigned int totalEntries() const;

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const;

		/**
		 * Returns the number of data entries at any index.
		 */
		unsigned int sizeOfDim( unsigned int dim ) const;

		/**
		 * Reallocates data. Data not preserved unless same # of dims
		 */
		bool resize( vector< unsigned int > dims );

		 /**
		  * Converts unsigned int into vector with index in each dimension
		  */
		 vector< unsigned int > multiDimIndex( unsigned int index ) const;

		 /**
		  * Converts index vector into unsigned int. If there are indices
		  * outside the dimension of the current data, then it returns 0?
		  */
		 unsigned int linearIndex( const vector< unsigned int >& index ) const;

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

	protected:
		unsigned int nextIndex( unsigned int index ) const;
		unsigned int G

	private:
		char* data_;
		unsigned int size_;	// Number of data entries in the whole array
		unsigned int start_;	// Starting index of data, used in MPI.
		unsigned int end_;	// Starting index of data, used in MPI.
		DataDimensions dims_;
};

#endif	// _ANY_DIM_HANDLER_H
