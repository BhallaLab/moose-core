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
		ZeroDimGlobalHandler( const DinfoBase* dinfo )
			: DataHandler( dinfo ), data_( 0 )
		{;}

		~ZeroDimGlobalHandler();

		/**
		 * Make 'n' copies, doing appropriate node partitioning if
		 * toGlobal is false.
		 */
		DataHandler* copy( unsigned int n, bool toGlobal ) const;

		/**
		 * calls process on data, using threading info from the ProcInfo
		 */
		void process( const ProcInfo* p, Element* e, FuncId fid ) const;

		/**
		 * Returns the data on the specified index.
		 */
		char* data( DataId index ) const {
			return data_;
		}

		/**
		 * Returns the data at one level up of indexing.
		 * Here there isn't any.
		 */
		char* data1( DataId index ) const {
			return data_;
		}

		/**
		 * Returns the number of data entries.
		 */
		unsigned int numData() const {
			return 1;
		}

		/**
		 * Returns the number of data entries at index 1.
		 */
		unsigned int numData1() const {
			return 1;
		}

		/**
		 * Returns the number of data entries at index 2, if present.
		 * For regular Elements and 1-D arrays this is always 1.
		 */
		 unsigned int numData2( unsigned int index1 ) const {
		 	return 1;
		 }

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const {
			return 0;
		}

		/**
		 * Assign # of entries in dimension 1. 
		 * Ignore here
		 */
		void setNumData1( unsigned int size ) {
			;
		}
		/**
		 * Assigns the sizes of all array field entries at once.
		 * This is ignored for regular Elements.
		 */
		void setNumData2( unsigned int start,
			const vector< unsigned int >& sizes ) {
			;
		}


		/**
		 * Looks up the sizes of all array field entries at once.
		 * Ignored in this case, as there are none.
		 * Returns the first index on this node, irrelevant here.
		 */
		unsigned int getNumData2( vector< unsigned int >& sizes ) const {
			return 0;
		}

		/**
		 * Returns true always: it is a global.
		 */
		bool isDataHere( DataId index ) const;

		virtual bool isAllocated() const;

		void allocate();

		bool isGlobal() const
		{
			return 1;
		}

		iterator begin() const;

		iterator end() const;

		/**
		 * This is relevant only for the 2 D cases like
		 * FieldDataHandlers.
		 */
		unsigned int startDim2index() const {
			return 0;
		}

		void setData( char* data, unsigned int numData ) {
			data_ = data;
		}

	private:
		char* data_;
};

#endif // _ZERO_DIM_GLOBAL_HANDLER_H
