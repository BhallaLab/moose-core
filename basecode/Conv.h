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
 *
 * Conv is used only during while the memory for the argument is stable.
 * You can't create a Conv object from a variable, and free the variable
 * before using the Conv.
 */

template< class T > class Conv
{
	public:
		Conv( const char* buf )
		{
			// Can make this more compact for things smaller than a ptr.
			// Worth trying as a default for speed.
			val_ = buf;
		}

		Conv( const T& arg )
		{
			val_ = reinterpret_cast< const char* >( &arg );
		}

		unsigned int size() const
		{
			return sizeof( T );
		}

		const T operator*() const {
			return *reinterpret_cast< const T* >( val_ );
		}

		// Will need to specialize for variable size and pointer-containing
		// D.
		unsigned int val2buf( char* buf ) {
			*reinterpret_cast< T* >( buf ) = *reinterpret_cast< const T* >( val_);
			// Or I could do a memcpy. Worth trying to see speed implication
			return sizeof( T );
		}

		static void str2val( T& val, const string& s ) {
			istringstream is( s );
			is >> val;
		}

		static void val2str( string& s, const T& val ) {
			ostringstream os( s );
			os << val;
		}
	private:
		const char* val_;
};

template<> class Conv< string >
{
	public:
		Conv( const char* buf )
			: val_( buf )
		{;}

		Conv( const string& arg )
			: val_( arg )
		{;}

		/**
		 * This is the size used in the serialized form, as a char*
		 * Note that we do some ugly stuff to get alignment on 8-byte
		 * boundaries.
		 * We need to have strlen + 1 as a minimum.
		 */
		unsigned int size() const
		{
			static const unsigned int ALIGNMENT = 8;
			return ALIGNMENT * ( 1 + ( val_.length() / ALIGNMENT ) );
		}

		const string& operator*() const {
			return val_;
		}

		unsigned int val2buf( char* buf ) {
			strcpy( buf, val_.c_str() );
			return size();
		}

		static void str2val( string& val, const string& s ) {
			val = s;
		}

		static void val2str( string& s, const string& val ) {
			s = val;
		}
	private:
		string val_;
};

#endif // _DINFO_H
