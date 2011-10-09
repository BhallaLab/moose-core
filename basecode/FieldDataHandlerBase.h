/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
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


		////////////////////////////////////////////////////////////
		// Information functions
		////////////////////////////////////////////////////////////

		/// Returns data on specified index
		char* data( DataId index ) const;

		/**
		 * Returns the number of data entries.
		 */
		unsigned int totalEntries() const;

		/**
		 * Returns the number of data entries on local node
		 */
		unsigned int localEntries() const;

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const;

		unsigned int sizeOfDim( unsigned int dim ) const;

		vector< unsigned int > dims() const;

		bool isDataHere( DataId index ) const;

		bool isAllocated() const;

		unsigned int getFieldArraySize( unsigned int i ) const;
		unsigned int getFieldArraySize( DataId di ) const;

		/**
		 * Returns the size of the largest field entry array on any object
		 * on all nodes handled by this DataHandler. Equivalent to 
		 * sizeOfDim( 0 ).
		 */
		unsigned int getMaxFieldEntries() const;

		/// Returns the bitmask for the field index
 		unsigned int fieldMask() const;

		/// Returns the number of bits in the field index.
		unsigned int numFieldBits() const;
		////////////////////////////////////////////////////////////////
		// Special field access funcs
		////////////////////////////////////////////////////////////////

		/**
		 * Looks up field entry on specified parent data object, 
		 * with specified integer index for the field.
		 * Returns field entry cast to char.
		 */
		virtual char* lookupField( char* pa, unsigned int index ) 
			const = 0;

		/**
		 * Return number of fields on Parent object pa
		 */
		virtual unsigned int getNumField( const char* pa ) const = 0;

		/**
		 * Set number of fields on Parent object pa
		 */
		virtual void setNumField( char* pa, unsigned int num ) = 0;

		/**
		 * Assigns the size of the field array on the specified object.
		 * 
		 */
		void setFieldArraySize( DataId di, unsigned int size );

		/**
		 * Looks up the size of the field array on the specified object
		unsigned int getFieldArraySize( unsigned int objectIndex ) const;
		 */

		/**
		 * Looks up the biggest field array size on the current node
		 * Implemented in derived classes.
		 */
		unsigned int biggestFieldArraySize() const;


		////////////////////////////////////////////////////////////////
		// load balancing functions
		////////////////////////////////////////////////////////////////
		bool innerNodeBalance( unsigned int size,
			unsigned int myNode, unsigned int numNodes );

		////////////////////////////////////////////////////////////////
		// Process function
		////////////////////////////////////////////////////////////////
		/**
		 * calls process on data, using threading info from the ProcInfo
		 */
		void process( const ProcInfo* p, Element* e, FuncId fid ) const;

		void foreach( const OpFunc* f, Element* e, const Qinfo* q,
			const double* arg, 
			unsigned int argSize, unsigned int numArgs ) const;

		unsigned int getAllData( vector< char* >& data ) const;

		////////////////////////////////////////////////////////////////
		// Data Reallocation functions
		////////////////////////////////////////////////////////////////

		void globalize( const char* data, unsigned int size );
		void unGlobalize();


		// We do not implement these copy() funcs here, left to the
		// derived templated class.
		//DataHandler* copy( bool toGlobal, unsigned int n ) const;
		// DataHandler* copyUsingNewDinfo( const DinfoBase* dinfo) const;

		bool resize( unsigned int dimension, unsigned int size );

		// Handled by derived templated FieldDataHandler classes.
		// void assign( const char* orig, unsigned int numOrig );


		/**
		 * This function updates the mask_ and the numFieldBits_ in
		 * coordination with maxFieldEntries.
		 */
		void setMaxFieldEntries( unsigned int num );

		/////////////////////////////////////////////////////////////////
		// Data access
		/////////////////////////////////////////////////////////////////

		/**
		 * Returns the data handler of the parent object that contains the
		 * Field array.
		 */
		const DataHandler* parentDataHandler() const;

		/**
		 * Assign parent data handler. Used only by 
		 * Shell::innerCopyElements().
		 */
		void assignParentDataHandler( const DataHandler* parent );
		
	private:

		/**
		 * This is the size of the largest array of fields on all the 
		 * parent objects on this node.
		 */
		unsigned int maxFieldEntries_;

		/**
		 * Pointer to the data handler of the parent object.
		 */
		const DataHandler* parentDataHandler_;

		/**
		 * Bitmask for field part of DataId
		 */
		unsigned int mask_;

		/**
		 * bits used by the last portion of the DataId to specify last
		 * dimension.
		 */
		unsigned int numFieldBits_;
};

#endif	// _FIELD_DATA_HANDLER_BASE_H

