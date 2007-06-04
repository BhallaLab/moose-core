/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "moose.h"
// #include "DerivedFtype.h"

//////////////////////////////////////////////////////////////////
// Some template specializations for the serialization in the
// incomingFunc and outgoingFunc.
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// Strings.
//////////////////////////////////////////////////////////////////
/**
 * Converts serialized data into a string.
 * Here the string was serialized as a null-terminated char array.
 */
template<> const void* unserialize< string >( string& s, const void* data )
{
	const char* temp = static_cast< const char* >( data );
	s = temp;
	return temp + s.length() + 1;
}


/**
 * Converts string into serialized data.
 * Puts the value into the data as a null-terminated string.
 */
template<> void* serialize< string >( void* data, const string& value )
{
	char* c = static_cast< char* >( data );
	strcpy( c, value.c_str() );
	return c + value.length() + 1;
}

/**
 * Gets the serialized size of the string.
 */
template<> size_t serialSize< string >( const string& value )
{
	return value.length() + 1;
}

//////////////////////////////////////////////////////////////////
// vector of unsigned int. First entry is set to size.
//////////////////////////////////////////////////////////////////
/**
 * Converts serialized data into a vector. Now how do I do vectors
 * of arbitrary objects?
 */
template<> const void* unserialize< vector< unsigned int > >( 
				vector< unsigned int >& value, const void* data )
{
	// Now why would you want to transfer such a big vector?
	static const unsigned int MAX_SIZE = 1000000;
	// Here the first entry is the vector size, stored as an
	// unsigned int.
	const unsigned int* temp = static_cast< const unsigned int* >( data );
	unsigned int size = temp[0];
	assert( size < MAX_SIZE );
	value.resize( size );
	temp++;
	vector< unsigned int >::iterator i;
	for ( i = value.begin(); i != value.end(); i++ )
		*i = *temp++;

	return temp;
}


/**
 * Converts vector into serialized data.
 * Stores the size as the first entry.
 * We assume enough space in data for all this!
 */
template<> void* serialize< vector< unsigned int > >( 
		void* data, const vector< unsigned int >& value )
{
	// Now why would you want to transfer such a big vector?
	static const unsigned int MAX_SIZE = 1000000;
	assert( value.size() < MAX_SIZE );
	unsigned int* temp = static_cast< unsigned int* >( data );
	*temp++ = value.size();
	vector< unsigned int >::const_iterator i;
	for ( i = value.begin(); i != value.end(); i++ )
		*temp++ = *i;
	return temp;
}

/**
 * Gets the serialized size of the vector.
 */
template<> size_t serialSize< vector< unsigned int > >(
				const vector< unsigned int >& value )
{
	return ( value.size() + 1 ) * sizeof( unsigned int );
}

//////////////////////////////////////////////////////////////////
// vector of double. First entry is set to size.
//////////////////////////////////////////////////////////////////
/**
 * Converts serialized data into a vector. Now how do I do vectors
 * of arbitrary objects?
 */
template<> const void* unserialize< vector< double > >( 
				vector< double >& value, const void* data )
{
	// Now why would you want to transfer such a big vector?
	static const unsigned int MAX_SIZE = 1000000;
	// Here the first entry is the vector size, stored as an
	// unsigned int.
	const unsigned int* itemp = 
			static_cast< const unsigned int* >( data );
	unsigned int size = *itemp;
	assert( size < MAX_SIZE );
	const double* temp = static_cast< const double* >( 
					static_cast< const void* >( itemp + 1 ) );
	value.resize( size );
	vector< double >::iterator i;
	for ( i = value.begin(); i != value.end(); i++ )
		*i = *temp++;

	return temp;
}


/**
 * Converts vector into serialized data.
 * Stores the size as the first entry.
 * We assume enough space in data for all this!
 */
template<> void* serialize< vector< double > >( 
		void* data, const vector< double >& value )
{
	// Now why would you want to transfer such a big vector?
	static const double MAX_SIZE = 1000000;
	assert( value.size() < MAX_SIZE );
	unsigned int* itemp = static_cast< unsigned int* >( data );
	*itemp = value.size();

	double* temp = static_cast< double* >( 
					static_cast< void* >( itemp + 1 ) );
	vector< double >::const_iterator i;
	for ( i = value.begin(); i != value.end(); i++ )
		*temp++ = *i;

	return temp;
}

/**
 * Gets the serialized size of the vector.
 */
template<> size_t serialSize< vector< double > >(
				const vector< double >& value )
{
	return sizeof( unsigned int ) + value.size() * sizeof( double );
}

//////////////////////////////////////////////////////////////////
// vector of strings. First entry is set to size, rest are
// null-separated chars.
//////////////////////////////////////////////////////////////////
/**
 * Converts serialized data into a vector. Now how do I do vectors
 * of arbitrary objects?
 */
template<> const void* unserialize< vector< string > >( 
				vector< string >& value, const void* data )
{
	// Now why would you want to transfer such a big vector?
	static const unsigned int MAX_SIZE = 100000;
	// Here the first entry is the vector size, stored as an
	// unsigned int.
	unsigned int size = *static_cast< const unsigned int* >( data );
	assert( size < MAX_SIZE );
	const char* temp = static_cast< const char* >( data ) + 
			sizeof( unsigned int );
	value.resize( size );
	vector< string >::iterator i;
	for ( i = value.begin(); i != value.end(); i++ ) {
		*i = temp;
		temp += i->length() + 1;
	}

	return temp;
}


/**
 * Converts vector into serialized data.
 * Stores the size as the first entry.
 * We assume enough space in data for all this!
 */
template<> void* serialize< vector< string > >( 
		void* data, const vector< string >& value )
{
	// Now why would you want to transfer such a big vector?
	static const double MAX_SIZE = 100000;
	assert( value.size() < MAX_SIZE );
	*static_cast< unsigned int* >( data ) = value.size();

	char* temp = static_cast< char* >( data ) + sizeof( unsigned int );
	vector< string >::const_iterator i;
	for ( i = value.begin(); i != value.end(); i++ ) {
		strcpy( temp, i->c_str() );
		temp += i->length() + 1;
	}
	return temp;
}

/**
 * Gets the serialized size of the vector.
 */
template<> size_t serialSize< vector< string > >(
				const vector< string >& value )
{
	unsigned int ret = sizeof( unsigned int );
	vector< string >::const_iterator i;
	for ( i = value.begin(); i != value.end(); i++ ) {
		ret += i->length() + 1;
	}
	return ret;
}

//////////////////////////////////////////////////////////////////
//  This is here as a fallback in case we do not have the 
//  parallel code to use.
//////////////////////////////////////////////////////////////////
#ifndef USE_MPI
void* getParBuf( const Conn& c, unsigned int size )
{
	return 0;
}
void* getAsyncParBuf( const Conn& c, unsigned int size )
{
	return 0;
}
#endif

