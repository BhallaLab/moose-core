/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _SERIALIZER_H
#define _SERIALIZER_H

#define MAX_SERIALIZER_SIZE 100000

/**
 * Class to handle data serialization. This was originally implemented
 * as a a set of independent functions, 
 * but C++ does not allow partial template specialization
 * of functions which we need here to handle vector data transfer.
 */
template< class T > class Serializer
{
	public:
		Serializer()
			: data_( 0 )
		{;}

		Serializer( const void* data )
			: data_( data )
		{;}
		/**
 		* Converts serialized data into defined types.
 		* This function has to be specialised for many Ftypes like
 		* vectors and strings. This default works only for simple
 		* types.
 		*/
		static const void* unserialize( T& value, const void* data ) {
			value = *static_cast< const T* >( data );
			return static_cast< const char* >( data ) + sizeof( T );
		}

		/**
 		* Converts defined types into serialized data.
 		* Puts the value into the data.
 		* This function has to be specialised for many Ftypes like
 		* vectors and strings. This default works only for simple
 		* types.
 		*/
		static void* serialize( void* data, const T& value )
		{
			T* temp = static_cast< T* >( data );
			*temp = value;
			return temp + 1;
		}

		/**
 		* Gets the serialized size of the data type. Usually a simple
 		* sizeof, but when handling complex types it will need to be 
		* specialized.
 		*/
		static size_t serialSize( const T& value )
		{
			return sizeof( T );
		}
	private:
		void* data_;
};

/**
 * Specialization to handle strings.
 */
template<> class Serializer< string >
{
	public:
		/*
		Serializer()
			: data_( 0 )
		{;}
		*/
		/**
 		* Converts serialized data into string
 		*/
		static const void* unserialize( string& value, const void* data ) {
			const char* temp = static_cast< const char* >( data );
			value = temp;
			return static_cast< const void* >( temp + value.length() + 1 );
		}

		/**
 		* Converts defined types into serialized data.
 		*/
		static void* serialize( void* data, const string& value )
		{
			char* c = static_cast< char* >( data );
			strcpy( c, value.c_str() );
			return c + value.length() + 1;
		}

		/**
 		* Gets the serialized size of the data type.
 		*/
		static size_t serialSize( const string& value )
		{
			return value.length() + 1;
		}
	private:
		void* data_;
};

/**
 * Specialization to handle vectors of strings.
 */
template<> class Serializer< vector< string > >
{
	public:
		/*
		Serializer( void* data )
			: data_( data )
		{;}
		*/
		/**
 		* Converts serialized data into a vector of strings.
 		*/
		static const void* unserialize( vector< string >& value, 
			const void* data ) {
			unsigned int size = *static_cast< const unsigned int* >( data);
			assert( size < MAX_SERIALIZER_SIZE );
			const char* temp = static_cast< const char* >( data ) + 
				sizeof( unsigned int );
			value.resize( size );
			vector< string >::iterator i;
			for ( i = value.begin(); i != value.end(); i++ ) {
				*i = temp;
				temp += i->length() + 1;
			}
			return static_cast< const void* >( temp );
		}

		/**
 		* Converts vector of strings into serialized data.
 		* Stores the size as the first entry.
		* Returns a pointer to the next location in 'data' memory.
 		*/
		static void* serialize( void* data, const vector< string >& value )
		{
			assert( value.size() < MAX_SERIALIZER_SIZE );
			*static_cast< unsigned int* >( data ) = value.size();
		
			char* temp = static_cast< char* >( data ) + sizeof( unsigned int );
			vector< string >::const_iterator i;
			for ( i = value.begin(); i != value.end(); i++ ) {
				strcpy( temp, i->c_str() );
				temp += i->length() + 1;
			}
			return static_cast< void* >( temp );
		}

		/**
 		* Gets the serialized size of the data type.
 		*/
		static size_t serialSize( const vector< string >& value )
		{
			unsigned int ret = sizeof( unsigned int );
			vector< string >::const_iterator i;
			for ( i = value.begin(); i != value.end(); i++ ) {
				ret += i->length() + 1;
			}
			return ret;
		}
	private:
		void* data_;
};


/**
 * Here is a partial specialization of Serializer to help it handle
 * vectors.
 */
// template< class T > class Serializer< T * >
template< class T > class Serializer< vector< T > >
//template< class T > Serializer< vector< T > >
{
	public:
		/*
		Serializer( void* data )
			: data_( data )
		{;}
		*/
		/**
 		* Converts serialized data into defined types.
 		* Returns the data pointer incremented to the next field.
 		*/
		static const void* unserialize( vector< T >& value,
			const void* data ) {
			// Now why would you want to transfer such a big vector?
			// Here the first entry is the vector size, stored as an
			// unsigned int.
			unsigned int size = *static_cast< const unsigned int* >( data);
			assert( size < MAX_SERIALIZER_SIZE );
			value.resize( size );
			const char* temp = static_cast< const char* >( data );
			memcpy( &( value[0] ), temp + sizeof( unsigned int ), 
				size * sizeof( T ) );
			return static_cast< const void* >( 
				temp + sizeof( unsigned int ) + size * sizeof( T ) );
		}

		/**
 		* Converts defined types into serialized data.
 		* Returns data pointer incremented to end of this field.
 		*/
		static void* serialize( void* data, const vector< T >& value )
		{
			size_t vs = value.size();
			assert( vs < MAX_SERIALIZER_SIZE );
			*static_cast< unsigned int* >( data ) = vs;
			char* temp = static_cast< char* >( data );
			memcpy( temp + sizeof( unsigned int ), &( value[0] ),
				sizeof( T ) * vs );

			return static_cast< void* >( 
				temp + sizeof( unsigned int ) + vs * sizeof( T ) );
		}

		/**
 		* Gets the serialized size of the data type.
 		*/
		static size_t serialSize( const vector< T >& value )
		{
			return sizeof( unsigned int ) + sizeof( T ) * value.size();
		}
	private:
		void* data_;
};

#endif // _SERIALIZER_H
