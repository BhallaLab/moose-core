/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _FIELD_ELEMENT_H
#define _FIELD_ELEMENT_H

// Template in the field lookup function for speed.
// Provide the less-used functions for getting and setting num of
// fields, as function pointers.

/**
 * This template is for an array field which itself should be accessed as
 * an Element. Examples are synapses and clock ticks.
 */
template< class Field, class Parent, Field* ( Parent::*Lookup )( unsigned int ) > class FieldElement: public Element
{
	public:
		FieldElement( const Cinfo* c, const Element* other,
			unsigned int ( Parent::*getNumField )() const,
			void ( Parent::*setNumField )( unsigned int num )
		)
			: Element( c, other ), 
				getNumField_( getNumField ),
				setNumField_( setNumField )
		{;}

		void process( const ProcInfo* p ) // Don't do anything.
		{;}

		/**
		 * Return the field.
		 */
		char* data( DataId index )
		{
			assert( index.data() < numData_ );
			Field* s = ( ( reinterpret_cast< Parent* >( d_ + index.data() * dataSize_ ) )->*Lookup )( index.field() );
			return reinterpret_cast< char* >( s );
		}

		/**
		 * Return the parent data
		 */
		char* data1( DataId index )
		{
			assert( index.data() < numData_ );
			return d_ + index.data() * dataSize_;
		}

		/**
		 * Return the # of field entries.
		 */
		unsigned int numData() const
		{
			unsigned int ret = 0;
			char* endData = d_ + numData_ * dataSize_;
			for ( char* data = d_; data< endData; data += dataSize_ )
				ret += ( ( reinterpret_cast< Parent* >( data ) )->*getNumField_ )();
			return ret;
		}

		/**
		 * Return the # of synapses on a given IntFire
		 */
		unsigned int numData2( unsigned int index1 ) const
		{
			assert( index1 < numData_ );
			return ( ( reinterpret_cast< Parent* >( d_ + index1 * dataSize_ ) )->*getNumField_ )();
		}

		/**
		 * This Element has 2 dimensions.
		 */
		unsigned int numDimensions() const
		{
			return 2;
		}

		void setArraySizes( const vector< unsigned int >& sizes )
		{
		   char* endData = d_ + numData_ * dataSize_;
			assert( sizes.size() == numData_ );
			vector< unsigned int >::const_iterator i = sizes.begin();
			for ( char* data = d_; data < endData; data += dataSize_ )
			( reinterpret_cast< Parent* >( data ) )->setNumField_( *i++ );
		}

		void getArraySizes( vector< unsigned int >& sizes ) const
		{
			sizes.resize( 0 );
			char* endData = d_ + numData_ * dataSize_;

			for ( char* data = d_; data < endData; data += dataSize_ )
				sizes.push_back( ( reinterpret_cast< Parent* >( data ) )->getNumField_() );
		}
	private:
		unsigned int ( Parent::*getNumField_ )() const;
		void ( Parent::*setNumField_ )( unsigned int num );
};

#endif // _FIELD_ELEMENT_H
