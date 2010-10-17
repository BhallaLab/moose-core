/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _FIELD_DATA_HANDLER_H
#define _FIELD_DATA_HANDLER_H

/**
 * This class manages access to array fields X in an array of objects Y.
 * Examples are synapses and clock ticks.
 * Replaces FieldElement.h
 * It is templated by the field type, the parent type and a lookup function
 * that extracts the field from the parent.
 */

template< class Parent, class Field > class FieldDataHandler: public DataHandler
{
	public:
		FieldDataHandler( const DinfoBase* dinfo,
			const DataHandler* parentDataHandler,
			Field* ( Parent::*lookupField )( unsigned int ),
			unsigned int ( Parent::*getNumField )() const,
			void ( Parent::*setNumField )( unsigned int num ) )
			: DataHandler( dinfo ),
				parentDataHandler_( parentDataHandler ),
				lookupField_( lookupField ),
				getNumField_( getNumField ),
				setNumField_( setNumField ),
				size_( 1 ),
				start_( 0 )

		{;}

		~FieldDataHandler()
		{;} // Don't delete data because the parent Element should do so.

		DataHandler* globalize() const
		{
			return 0;
		}

		DataHandler* unGlobalize() const
		{
			return 0;
		}

		bool nodeBalance( unsigned int size )
		{
			return 0;
		}


		/**
		 * This really won't work, as it is just a hook to the parent
		 * Data Handler. Need the duplicated Parent for this.
		 *
		 * If n is 1, just duplicates everything. No problem.
		 * if n > 1, then operation is nasty.
		 * Scales up the data dimension from 0 to 1 if original had 1 entry,
		 * and assigns n to the new dimension. This is a problem on multi
		 * nodes as the original would have been sitting on node 0.
		 * Scales up data dimension from 1 to 2 if original had an array.
		 * 2nd dimension is now n. For multinodes does a hack by scaling
		 * up all entries by n, rather than doing a clean repartitioning.
		 */
		DataHandler* copy() const
		{
			FieldDataHandler< Parent, Field >* ret =
				new FieldDataHandler< Parent, Field >( *this );
			return ret;
		}

		DataHandler* copyExpand( unsigned int copySize ) const
		{
			return 0;
		}

		DataHandler* copyToNewDim( unsigned int newDimSize ) const
		{
			return 0;
		}

		void process( const ProcInfo* p, Element* e, FuncId fid ) const 
		{
			; // Fields don't do independent process?
		}

		/**
		 * Returns the data on the specified index.
		 */
		char* data( DataId index ) const
		{
			char* pa = parentDataHandler_->data( index );
			if ( pa ) {
				Field* s = ( ( reinterpret_cast< Parent* >( pa ) )->*lookupField_ )( index.field() );
				return reinterpret_cast< char* >( s );
			}
			return 0;
		}

		/**
		 * Returns the number of field entries.
		 * If parent is global the return value is also global.
		 * If parent is local then it returns # on current node.
		 */
		unsigned int totalEntries() const {
			unsigned int ret = 0;
			for ( DataHandler::iterator i = parentDataHandler_->begin();
				i != parentDataHandler_->end(); ++i ) {
				ret += 
				( ( reinterpret_cast< Parent* >( *i ) )->*getNumField_ )();
			}


			/*
			unsigned int size = parentDataHandler_->numData1();
			unsigned int start = 
				 ( size * Shell::myNode() ) / Shell::numNodes();
			unsigned int end = 
				 ( size * ( 1 + Shell::myNode() ) ) / Shell::numNodes();

			for ( unsigned int i = start; i < end; ++i ) {
				char* pa = parentDataHandler_->data1( i );
				ret += ( ( reinterpret_cast< Parent* >( pa ) )->*getNumField_ )();
			}
			*/
			return ret;
		}

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const {
			// Should refine to include local dimensions.
			// For now assume 1 dim.
			return parentDataHandler_->numDimensions() + 1;
		}

		unsigned int sizeOfDim( unsigned int dim ) const
		{
			if ( dim > 0 )
				return parentDataHandler_->sizeOfDim( dim - 1 );
			return size_;
		}


		/**
		 * Assigns size for first (data) dimension. This usually will not
		 * be called here, but by the parent data Element.
		 */
		bool resize( vector< unsigned int > dims )
		{
			cout << Shell::myNode() << ": FieldDataHandler::setNumData1: Error: Cannot resize from Field\n";
			return 0;
		}

		vector< unsigned int > dims() const
		{
			vector< unsigned int > ret( parentDataHandler_->dims() );
			ret.insert( ret.begin(), size_ );
			return ret;
		}

		/**
		 * Assigns the sizes of all array field entries at once.
		void setNumData2( unsigned int start, 
			const vector< unsigned int >& sizes ) {
			assert ( sizes.size() == parentDataHandler_->numData() );
			for ( DataHandler::iterator i = parentDataHandler_->begin();
				i != parentDataHandler_->end(); ++i ) {
				char* pa = parentDataHandler_->data1( i );
				( ( reinterpret_cast< Parent* >( pa ) )->*setNumField_ )( sizes[i] );
			}
			start_ = start;
		}
		 */


		/**
		 * Returns true if the node decomposition has the data on the
		 * current node
		 */
		bool isDataHere( DataId index ) const {
			return parentDataHandler_->isDataHere( index );
		}

		bool isAllocated() const {
			return parentDataHandler_->isAllocated();
		}

		bool isGlobal() const
		{
			return parentDataHandler_->isGlobal();
		}

		/**
		 * This seems funny, but remember than begin() refers to the
		 * data part on the index.
		 */
		iterator begin() const {
			// Assume we start on 0. Easy to extend to starting at another
			// index.
			iterator ret( this, parentDataHandler_->begin().index() * size_ );
			return ret;
		}

		/**
		 * This is the last valid field entry on the last valid data entry
		 * on the parent data handler, expressed as a single int.
		 * Index should perhaps be a long.
		 */
		iterator end() const {
			unsigned int paIndex = parentDataHandler_->end().index();
			char* pa = parentDataHandler_->data( paIndex );
			assert( pa );
			unsigned int numHere = ( ( reinterpret_cast< Parent* >( pa ) )->*getNumField_ )();
			iterator ret( this, paIndex * size_ + numHere );
			return ret;
		}

		const DataHandler* parentDataHandler() const {
			return parentDataHandler_;
		}

		bool setDataBlock( const char* data, unsigned int numData,
			DataId did ) const 
		{
			if ( parentDataHandler_->isDataHere( did.data() ) ) {
				char* temp = parentDataHandler_->data( did.data() );
				assert( temp );
				Parent* pa = reinterpret_cast< Parent* >( temp );

				unsigned int numField = ( pa->*getNumField_ )();
				unsigned int max = numData;
				if ( did.field() + numData > numField  )
					max = numField - did.field();
				for ( unsigned int i = 0; i < max; ++i ) {
					Field* f = ( pa->*lookupField_ )( did.field() + i );
					*f = *reinterpret_cast< const Field* >( 
						data + i * dinfo()->size() );
				}
			}
			return 1;
		}

		/**
		 * Assigns a block of data at the specified location.
		 * Returns true if all OK. No allocation.
		 */
		bool setDataBlock( const char* data, unsigned int numData,
			const vector< unsigned int >& startIndex ) const
		{
			if ( startIndex.size() == 0 )
				return setDataBlock( data, numData, 0 );
			unsigned int fieldIndex = startIndex[0];
			if ( startIndex.size() == 1 )
				return setDataBlock( data, numData, DataId( 0, fieldIndex ) );
			vector< unsigned int > temp = startIndex;
			temp.pop_back(); // Get rid of fieldIndex.
			DataDimensions dd( parentDataHandler_->dims() );
			unsigned int paIndex = dd.linearIndex( temp );
			return setDataBlock( data, numData, DataId( paIndex, fieldIndex ) );
		}

		unsigned int nextIndex( unsigned int index ) const {
			unsigned int paIndex = index / size_;
			unsigned int fieldIndex = index % size_;
			char* pa = parentDataHandler_->data( paIndex );
			assert( pa );
			unsigned int numHere = ( ( reinterpret_cast< Parent* >( pa ) )->*getNumField_ )();
			if ( ( fieldIndex + 1 ) >= numHere ) {
				return (paIndex + 1) * size_;
			} else {
				return index + 1;
			}
		}

	private:
		const DataHandler* parentDataHandler_;
		Field* ( Parent::*lookupField_ )( unsigned int );
		unsigned int ( Parent::*getNumField_ )() const;
		void ( Parent::*setNumField_ )( unsigned int num );
		unsigned int size_;
		/**
		 * This keeps track of the max # of fieldElements assigned. It is
		 * analogous to the reserve size of a vector, but does not incur
		 * any extra overhead in memory. This determines how the indexing
		 * happens. As this gets rescaled every time new allocation happens
		 */
		unsigned int fieldSize_;
		unsigned int start_;
};

#endif	// _FIELD_DATA_HANDLER_H

