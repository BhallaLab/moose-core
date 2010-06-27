/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _DATA_HANDLER_WRAPPER_H
#define _DATA_HANDLER_WRAPPER_H

/**
 * This class wraps a DataHandler pointer in all respects. It allows the
 * DataHandler to be reused without fear of being deleted. The parent
 * data handler is never modified, though the data contents may be.
 */
class DataHandlerWrapper: public DataHandler
{
	public:
		DataHandlerWrapper( const DataHandler* parentHandler );

		~DataHandlerWrapper();

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
		char* data( DataId index ) const;

		/**
		 * Returns the data at one level up of indexing.
		 * Here there isn't any.
		 */
		char* data1( DataId index ) const;

		/**
		 * Returns the number of data entries.
		 */
		unsigned int numData() const;

		/**
		 * Returns the number of data entries at index 1.
		 */
		unsigned int numData1() const;

		/**
		 * Returns the number of data entries at index 2, if present.
		 * For regular Elements and 1-D arrays this is always 1.
		 */
		 unsigned int numData2( unsigned int index1 ) const;

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const;

		/**
		 * Assign # of entries in dimension 1. 
		 * Ignore here
		 */
		void setNumData1( unsigned int size );
		/**
		 * Assigns the sizes of all array field entries at once.
		 * This is ignored as the parent is readonly.
		 */
		void setNumData2( unsigned int start,
			const vector< unsigned int >& sizes );

		/**
		 * Looks up the sizes of all array field entries at once.
		 * Ignored in this case, as there are none.
		 * Returns the first index on this node, irrelevant here.
		 */
		unsigned int getNumData2( vector< unsigned int >& sizes ) const;

		/**
		 * Returns true always: it is a global.
		 */
		bool isDataHere( DataId index ) const;

		virtual bool isAllocated() const;

		void allocate();

		bool isGlobal() const;

		iterator begin() const;

		iterator end() const;

		/**
		 * This is relevant only for the 2 D cases like
		 * FieldDataHandlers.
		 */
		unsigned int startDim2index() const;

		void setData( char* data, unsigned int numData );

	private:
		const DataHandler* parent_;
};

#endif // _DATA_HANDLER_WRAPPER_H
