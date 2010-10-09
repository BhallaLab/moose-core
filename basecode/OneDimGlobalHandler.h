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

		void assimilateData( const char* data,
			unsigned int begin, unsigned int end );

		virtual bool nodeBalance( unsigned int size ) {
			return 0; // No change for the globals.
		}

		DataHandler* copy() const;

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
		 * Returns the number of data entries.
		 */
		unsigned int totalEntries() const {
			return size_;
		}

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
		bool isDataHere( DataId index ) const {
			return 1;
		}

		bool isAllocated() const {
			return ( data_ != 0 );
		}

		bool isGlobal() const
		{
			return 1;
		}

		iterator begin() const {
			return iterator( this, 0 );
		}

		iterator end() const {
			return iterator( this, size_ );
		}

	protected:
		void setData( char* data, unsigned int numData );
		unsigned int nextIndex( unsigned int index ) const {
			return index + 1;
		}
		char* data_;
		unsigned int size_;	// Number of data entries in the whole array
};

#endif	// _ONE_DIM_GLOBAL_HANDLER_H
