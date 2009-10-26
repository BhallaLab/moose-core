/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _CONV_H
#define _CONV_H

/**
 * This set of templates defines converters. The conversions are from
 * string to object, object to string, and 
 * binary buffer to object, object to binary buffer.
 * Many classes convert through a single template. Strings and other things
 * with additional data need special converters.
 * The key point is that the serialized form (buffer or output string)
 * must have a complete independent copy of the data. So if a pointer or
 * reference type is converted, then there must be a complete copy made.
 */

template< class T > class Conv
{
	public:
		static unsigned int size( const T& value ) {
			return sizeof( T );
		}

		// Will need to specialize for variable size and pointer-containing
		// D.
		// Later: Need to replace char buf with some managed, expandable
		// buffer.
		static unsigned int val2buf( char* buf, const T& val ) {
			*reinterpret_cast< T* >( buf ) = val;
			return sizeof( T );
		}

		static unsigned int buf2val( T& val, const char* buf ) {
			val = *reinterpret_cast< const T* >( buf );
			return sizeof( T );
		}

		/**
		 * Puts pointer to T into val. Returns new position of the buffer
		 * Usually does it simply by pointing T into the correct location
		 * in buf. No actual data transfer or allocation.
		 * This has a problem if we need to allocate something new to
		 * get it into T. For example, converting a char* into a string.
		static const char* buf2val( T** val, const char* buf );
		/// Other option is to return a reference
		 */
		/**
		 *
		static char* val2buf( char* buf, const T* val );
		*/

		static void str2val( T& val, const string& s ) {
			istringstream is( s );
			is >> val;
		}

		static void val2str( string& s, const T& val ) {
			ostringstream os( s );
			os << val;
		}
};

template<> class Conv< string >
{
	public:
		static unsigned int size( const string& value ) {
			return value.length() + 1;
		}

		static unsigned int val2buf( char* buf, const string& val ) {
			strcpy( buf, val.c_str() );
			return val.length() + 1;
		}

		static unsigned int buf2val( string& val, const char* buf ) {
			val = buf; // assume terminating \0.
			return val.length() + 1;
		}

		static void str2val( string& val, const string& s ) {
			val = s;
		}

		static void val2str( string& s, const string& val ) {
			s = val;
		}
};

#endif // _DINFO_H
