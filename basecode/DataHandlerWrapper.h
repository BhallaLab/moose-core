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

		DataHandler* globalize();
		DataHandler* unGlobalize();
		void assimilateData( const char* data,
			unsigned int begin, unsigned int end );
		bool nodeBalance( unsigned int size );

		/**
		 * Make a single identity copy, doing appropriate node 
		 * partitioning if toGlobal is false.
		 */
		DataHandler* copy() const;

		DataHandler* copyExpand( unsigned int copySize ) const;

		DataHandler* copyToNewDim( unsigned int newDimSize ) const;

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
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const;

		unsigned int sizeOfDim( unsigned int dim ) const;

		bool resize( vector< unsigned int > dims );

		vector< unsigned int > multiDimIndex( unsigned int index )     const;
		unsigned int linearIndex( 
			const vector< unsigned int >& index ) const;

		bool isDataHere( DataId index ) const;

		virtual bool isAllocated() const;

		bool isGlobal() const;

		iterator begin() const;

		iterator end() const;

	protected:
		void setData( char* data, unsigned int numData );
		unsigned int nextIndex( unsigned int index ) const;

	private:
		const DataHandler* parent_;
};

#endif // _DATA_HANDLER_WRAPPER_H
