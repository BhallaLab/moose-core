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

template< class Field, class Parent, Field* ( Parent::*Lookup )( unsigned int ) > class FieldDataHandler: public DataHandler
{
	public:
		FieldDataHandler( const DinfoBase* dinfo,
			unsigned int ( Parent::*getNumField )() const,
			void ( Parent::*setNumField )( unsigned int num ))
			: DataHandler( dinfo ),
				getNumField_( getNumField ),
				setNumField_( setNumField )

		{;}

		~FieldDataHandler()
		{;}

		/**
		 * Returns the data on the specified index.
		 */
		char* data( DataId index ) const
		{
			if ( isDataHere( index ) ) {
				Field* s = ( ( reinterpret_cast< Parent* >( 
					data_ + ( index.data() - start_ ) * 
						dinfo()->size() ) )->*Lookup )( index.field() );
				return reinterpret_cast< char* >( s );
			}
			return 0;
		}

		/**
		 * Returns the data at one level up of indexing. In this case it
		 * returns the parent of the field.
		 */
		char* data1( DataId index ) const
		{
			assert( index.data() < size_ );
			if ( isDataHere( index ) )
			{
				return data_ + ( index.data() - start_ ) * dinfo()->size();
			}
		}

		/**
		 * Returns the number of field entries.
		 * This runs into trouble on multinodes.
		 * I'll just return # on local node.
		 */
		unsigned int numData() const {
			unsigned int ret = 0;
			char* endData = data_ + ( end_ - start_ ) * dinfo()->size();
			for ( char* data = data_; 
					data < endData; data += dinfo()->size() )
				ret += ( ( reinterpret_cast< Parent* >( data ) )->*getNumField_ )();
			return ret;
		}

		/**
		 * Returns the number of data entries in the whole object.
		 */
		unsigned int numData1() const {
			return size_;
		}

		/**
		 * Returns the number of field entries on the data entry indicated
		 * by index1, if present.
		 * e.g., return the # of synapses on a given IntFire
		 */
		unsigned int numData2( unsigned int index1 ) const
		{
			if ( index1 >= start_ && index1 < end_ ) {
				return ( ( reinterpret_cast< Parent* >(
					data_ + ( index1 - start_ ) * dinfo()->size() ) )->*getNumField_ )();
			}
			return 0;
		}

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const {
			return 2;
		}

		/**
		 * Assigns size for first (data) dimension. This usually will not
		 * be called here, but by the parent data Element.
		 */
		void setNumData1( unsigned int size ) {
		/*
			size_ = size;
			unsigned int start = 
				( size_ * Shell::myNode() ) / Shell::numNodes();
			unsigned int end = 
				( size_ * ( 1 + Shell::myNode() ) ) / Shell::numNodes();
			if ( data_ && start == start && end == end_ ) // already done
				return;
			if ( data_ )
				dinfo()->destroyData
			data_ = reinterpret_cast< char* >(
				dinfo()->allocData( end - start ) );
			start_ = start;
			end_ = end;
			*/
		}

		/**
		 * Assigns the sizes of all array field entries at once.
		 * oops, this differs from what I had done for other subclasses.
		 */
		void setNumData2( const vector< unsigned int >& sizes ) {
			char* endData = data_ + size_ * dinfo()->size();
			assert( sizes.size() == size_ );
			vector< unsigned int >::const_iterator i = sizes.begin();
			for ( char* data = data_; 
				data < endData; 
				data += dinfo()->size() )
			( ( reinterpret_cast< Parent* >( data ) )->*setNumField_ )( *i++ );
		}

		/**
		 * Looks up the sizes of all array field entries at once. Returns
		 * all ones for regular Elements. 
		 */
		void getNumData2( vector< unsigned int >& sizes ) const
		{
			sizes.resize( 0 );
			char* endData = data_ + size_ * dinfo().size();

			for ( char* data = data_; 
				data < endData; 
				data += dinfo().size() )
				sizes.push_back( ( ( reinterpret_cast< Parent* >( data ) )->*getNumField_ )() );
		}

		/**
		 * Returns true if the node decomposition has the data on the
		 * current node
		 */
		bool isDataHere( DataId index ) const {
			return ( index.data() >= start_ && index.data() < end_ );
		}

		bool isAllocated() const {
			return (data_ != 0 );
		}

		/**
		 * Again, this should really be done at the parent Element, not
		 * here.
		 */
		void allocate() {
			if ( data_ )
				dinfo()->destroyData( data_ );
			data_ = reinterpret_cast< char* >(
				dinfo()->allocData( end_ - start_ ) );
		}

	private:
		char* data_;
		unsigned int size_;	// Number of data entries in the whole array
		unsigned int start_;	// Starting index of data, used in MPI.
		unsigned int end_;	// Starting index of data, used in MPI.
		unsigned int ( Parent::*getNumField_ )() const;
		void ( Parent::*setNumField_ )( unsigned int num );
};

#endif	// _FIELD_DATA_HANDLER_H

