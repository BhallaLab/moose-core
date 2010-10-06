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

		DataHandler* globalize();
		DataHandler* unGlobalize();
		void assimilateData( const char* data,
			unsigned int begin, unsigned int end );

		bool nodeBalance( unsigned int size );

		/**
		 * Make a single copy
		 */
		DataHandler* copy() const;

		DataHandler* copyExpand( unsigned int copySize ) const;

		DataHandler* copyToNewDim( unsigned int newDimSize ) const;

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
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const {
			return 0;
		}

		unsigned int sizeOfDim( unsigned int dim ) const
		{
			return ( dim == 0 );
		}

		bool resize( vector< unsigned int > dims )
		{
			return 0;
		}

		vector< unsigned int > multiDimIndex( unsigned int index ) const;

		unsigned int linearIndex(
		 	const vector< unsigned int >& index ) const;

		/**
		 * Returns true always: it is a global.
		 */
		bool isDataHere( DataId index ) const {
			return 1;
		}

		virtual bool isAllocated() const;

		bool isGlobal() const
		{
			return 1;
		}

		iterator begin() const;

		iterator end() const;

	protected:
		void setData( char* data, unsigned int numData ) {
			data_ = data;
		}

		unsigned int nextIndex( unsigned int index ) const {
			return 1;
		}

	private:
		char* data_;
};

#endif // _ZERO_DIM_GLOBAL_HANDLER_H
