/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _FIELD_DATA_HANDLER_BASE_H
#define _FIELD_DATA_HANDLER_BASE_H

/**
 * This is a base class for FieldDataHandlers, providing generic functions
 */
class FieldDataHandlerBase: public DataHandler
{
	public:
		FieldDataHandlerBase( const DinfoBase* dinfo,
			const DataHandler* parentDataHandler );

		~FieldDataHandlerBase();

		DataHandler* globalize() const;
		DataHandler* unGlobalize() const;

		bool innerNodeBalance( unsigned int size,
			unsigned int myNode, unsigned int numNodes );

		/// We don't implement the copy() func, left to the derived class
		// DataHandler* copy() const

		// These copy functions both return 0. Don't apply to Fields.
		DataHandler* copyExpand( unsigned int copySize ) const;
		DataHandler* copyToNewDim( unsigned int copySize ) const;

		// Process doesn't do anything, left to the parent DataHandler.
		void process( const ProcInfo* p, Element* e, FuncId fid ) const;

		/**
		 * Looks up and returns data pointer of field.
		 */
		char* data( DataId index ) const;

		//////////////////////////////////////////////////////////////////
		// Utility functions managed by derived class
		//////////////////////////////////////////////////////////////////
		/**
		 * Return number of fields on Parent object pa located at
		 * the specified location
		 */
		virtual unsigned int getNumField( const char* pa ) const = 0;

		/**
		 * Set number of fields on Parent object located at
		 * the specified data location
		 */
		virtual void setNumField( char* pa, unsigned int num ) = 0;

		/**
		 * Looks up field entry on specified parent data object, 
		 * with specified integer index for the field.
		 * Returns field entry cast to char.
		 */
		virtual char* lookupField( char* pa, unsigned int index ) 
			const = 0;

		/**
		 * Returns the number of field entries.
		 * If parent is global the return value is also global.
		 * If parent is local then it returns # on current node.
		 */
		unsigned int totalEntries() const;

		/**
		 * Returns the number of field entries on local node.
		 */
		unsigned int localEntries() const;

		/**
		 * Returns a single number corresponding to the DataId.
		 * Overrides the default behaviour as we need to take into account
		 * the dimensions for the field part of the DataId
		 */
		unsigned int linearIndex( const DataId& d ) const;

		/**
		 * Returns the DataId corresponding to a single index.
		 * Overrides the default behaviour as we need to take into account
		 * the dimensions for the field part of the DataId
		 */
		DataId dataId( unsigned int linearIndex) const;

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const;

		/**
		 * Returns size of specified dimension. Note that dimension 0 
		 * is the size of the field.
		 */
		unsigned int sizeOfDim( unsigned int dim ) const;


		/**
		 * Assigns size for first (data) dimension. This usually will not
		 * be called here, but by the parent data Element.
		 */
		bool resize( vector< unsigned int > dims );

		/**
		 * Returns the dimensions of this. The Field dimension is on 
		 * index 0.
		 */
		vector< unsigned int > dims() const;

		/**
		 * Assigns the size of the field array on the specified object.
		 * 
		 */
		void setFieldArraySize( 
			unsigned int objectIndex, unsigned int size );

		/**
		 * Looks up the size of the field array on the specified object
		 */
		unsigned int getFieldArraySize( unsigned int objectIndex ) const;

		/**
		 * Looks up the biggest field array size on the current node
		 * Implemented in derived classes.
		 */
		unsigned int biggestFieldArraySize() const;

		/**
		 * This func gets the FieldArraySize from all nodes and updates
		 * Deprecated.
		unsigned int syncFieldArraySize();
		 */

		/**
		 * Assigns the fieldDimension. Checks that it is bigger than the
		 * biggest size on this node.
		 */
		void setFieldDimension( unsigned int size );
		unsigned int getFieldDimension() const;

		/**
		 * Returns true if the node decomposition has the data on the
		 * current node
		 */
		bool isDataHere( DataId index ) const;

		bool isAllocated() const;

		bool isGlobal() const;

		/////////////////////////////////////////////////////////////////
		// Iterators
		/////////////////////////////////////////////////////////////////

		/**
		 * Starting point for iterating over all Fields on all Objects on
		 * this node.
		 */
		iterator begin() const;

		/**
		 * This is 1+(last valid field entry) on the last valid data entry
		 * on the parent data handler, expressed as a single int.
		 */
		iterator end() const;

		/**
		 * Advances the iteration by one place. Note that due to the
		 * ragged array and skipping zero arrays, both the DataId index
		 * and the linear index may jump forward.
		 */
		void nextIndex( DataId& index, unsigned int& linearIndex ) const;

		/////////////////////////////////////////////////////////////////
		// Data access
		/////////////////////////////////////////////////////////////////

		/**
		 * Returns the data handler of the parent object that contains the
		 * Field array.
		 */
		const DataHandler* parentDataHandler() const;

		/**
		 * Assigns a data block.
		 * Note that due to the possible ragged arrays here, the incoming
		 * data block may have sections that are skipped over.
		 */
		bool setDataBlock( const char* data, unsigned int numData,
			DataId startIndex ) const;

		
		/**
		 * Assigns a block of data at the specified location.
		 * Returns true if all OK. No allocation.
		 */
		bool setDataBlock( const char* data, unsigned int numData,
			const vector< unsigned int >& startIndex ) const;
		
	private:
		/**
		 * Pointer to the data handler of the parent object.
		 */
		const DataHandler* parentDataHandler_;

		/**
		 * This keeps track of the max # of fieldElements assigned. It is
		 * analogous to the reserve size of a vector, but does not incur
		 * any extra overhead in memory. This determines how the indexing
		 * happens.
		 */
		unsigned int fieldDimension_;

};


#endif	// _FIELD_DATA_HANDLER_BASE_H

