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

static const unsigned int UnreasonablyLargeArray = 1000000;
template< class T > class Conv
{
	public:
		/**
		 * Constructs a conv object from a binary buffer, which is presumed
		 * filled by a complementary conversion at the sender.
		 */
		Conv( const double* buf )
		{
			// By default the inner representation is just another char*
			// pointer.
			// Can make this more compact for things smaller than a ptr.
			// Worth trying as a default for speed.
			val_ = buf;
		}

		/**
		 * Constructs a conv object from a reference to the original
		 * object of type T. By default the conv object just records the
		 * pointer of the object.
		 */ 
		Conv( const T& arg )
		{
			val_ = reinterpret_cast< const double* >( &arg );
		}

		const double* ptr() const
		{
			return val_;
		}

		/**
		 * Size, in units of sizeof(double). So a double would be 1,
		 * something with 1 chars would be 1 as well, but something
		 * with 9 chars would be 2.
		 */
		unsigned int size() const
		{
			return 1 + ( sizeof( T ) - 1 ) / sizeof( double );
		}

		/**
		 * Returns the value of the converted object
		 */
		const T operator*() const {
			if ( val_ == 0 ) {
				return T();
			}
			return *reinterpret_cast< const T* >( val_ );
		}

		/**
		 * Converts data contents into double* buf. Buf must be allocated
		 * ahead of time.
		 * Needs to be specialized for variable size and pointer-containing
		 * types T.
		 */
		unsigned int val2buf( double* buf ) const {
			*reinterpret_cast< T* >( buf ) = 
				*reinterpret_cast< const T* >( val_);
			// Or I could do a memcpy. Worth trying to see speed implication
			return sizeof( T );
		}

		/**
		 * Default conversion from string just puts the string
		 * into the char* pointer.
		 */
		static void str2val( T& val, const string& s ) {
			istringstream is( s );
			is >> val;
		}

		/**
		 * Default conversion into string just puts the char* representation
		 * into the string. Arguably a bad way to do it.
		 */
		static void val2str( string& s, const T& val ) {
			stringstream ss;
			ss << val;
			s = ss.str();
			// ostringstream os( s );
			// os << val;
		}

		static string rttiType() {
			if ( typeid( T ) == typeid( int ) )
				return "int";
			if ( typeid( T ) == typeid( short ) )
				return "short";
			if ( typeid( T ) == typeid( long ) )
				return "long";
			if ( typeid( T ) == typeid( unsigned int ) )
				return "unsigned int";
			if ( typeid( T ) == typeid( unsigned long ) )
				return "unsigned long";
			if ( typeid( T ) == typeid( float ) )
				return "float";
			if ( typeid( T ) == typeid( double ) )
				return "double";
			if ( typeid( T ) == typeid( Id ) )
				return "Id";
			if ( typeid( T ) == typeid( ObjId ) )
				return "ObjId";
			if ( typeid( T ) == typeid( DataId ) )
				return "DataId";
			return "bad";
		}

	private:
		const double* val_;
};

/**
 * This stores the data as the equivalent of a char* string, terminated with
 * the usual \0, but allocates it as a double[].
 */
template<> class Conv< string >
{
	public:
		Conv( const char* buf )
		{
			unsigned int len = 0;
			if ( buf != 0 ) {
				len = strlen( buf );
			}
			val_ = new double[ 1 + len/ sizeof( double ) ];
			memcpy( val_, buf, len + 1 );
		}

		Conv( const double* dbuf )
		{
			const char* buf = reinterpret_cast< const char* >( dbuf );
			unsigned int len = 0;
			if ( dbuf != 0 ) {
				len = strlen( buf );
			}
			val_ = new double[ 1 + len/ sizeof( double ) ];
			memcpy( val_, buf, len + 1 );
		}

		Conv( const string& arg )
		{
			val_ = new double[ 1 + arg.length()/ sizeof( double ) ];
			memcpy( val_, arg.c_str(), arg.length() + 1 );
		}

		~Conv()
		{
			delete[] val_;
		}


		const double* ptr() const
		{
			return val_;
		}

		/**
		 * This is the size used in the serialized form, as a double*
		 * Note that we do some ugly stuff to get alignment on 8-byte
		 * boundaries.
		 * We need to have strlen + 1 as a minimum.
		 */
		unsigned int size() const
		{
			return ( 1 + strlen( reinterpret_cast< char* >( val_ ) ) / sizeof( double ) );
		}

		const string operator*() const {
			return reinterpret_cast< const char* >( val_ );
		}

		unsigned int val2buf( double* dbuf ) const {
			unsigned int len = 
				strlen( reinterpret_cast< const char* >(val_) );
			memcpy( dbuf, val_, len + 1 );
			return 1 + len / sizeof( double );
		}

		static void str2val( string& val, const string& s ) {
			val = s;
		}

		static void val2str( string& s, const string& val ) {
			s = val;
		}

		static string rttiType() {
			return "string";
		}
	private:
		double* val_;
};

/**
 * The template specialization of Conv< bool > sets up alignment on
 * word boundaries by storing the bool as a double. 
 */
template<> class Conv< double >
{
	public:
		/// Constructor assumes that the buffer points to a double
		Conv( const double* buf )
		{
			assert( buf );
			val_ = *buf;
		}

		/// Constructor uses implicit conversion of bool to int.
		Conv( const double& arg )
			: val_( arg )
		{;}

		/**
		 * This is the size used in the serialized form.
		 */
		unsigned int size() const
		{
			return 1;
		}

		const double* ptr() const
		{
			return &val_;
		}

		const double operator*() const {
			return val_;
		}

		unsigned int val2buf( double* buf ) const {
			*buf = val_;
			return 1;
		}

		static void str2val( double& val, const string& s ) {
			val = atof( s.c_str() );
		}

		static void val2str( string& s, const double& val ) {
			stringstream ss;
			ss << val;
			s = ss.str();
		}

		static string rttiType() {
			return "double";
		}
	private:
		double val_;
};

/**
 * The template specialization of Conv< unsigned int > sets up alignment on
 * word boundaries by storing the data as a double. 
 */
template<> class Conv< unsigned int >
{
	public:
		/// Constructor assumes that the buffer points to a double
		Conv( const double* buf )
		{
			assert( buf );
			val_ = *buf;
		}

		/// Constructor uses implicit conversion of int to double.
		Conv( const unsigned int& arg )
			: val_( arg )
		{;}

		/**
		 * This is the size used in the serialized form.
		 */
		unsigned int size() const
		{
			return 1;
		}

		const double* ptr() const
		{
			return &val_;
		}

		const unsigned int operator*() const {
			return val_;
		}

		unsigned int val2buf( double* buf ) const {
			*buf = val_;
			return 1;
		}

		static void str2val( unsigned int& val, const string& s ) {
			val = atoi( s.c_str() );
		}

		static void val2str( string& s, const unsigned int& val ) {
			stringstream ss;
			ss << val;
			s = ss.str();
		}

		static string rttiType() {
			return "unsigned int";
		}
	private:
		double val_;
};

/**
 * The template specialization of Conv< float > sets up alignment on
 * word boundaries by storing the data as a double. 
 */
template<> class Conv< float >
{
	public:
		/// Constructor assumes that the buffer points to a double
		Conv( const double* buf )
		{
			assert( buf );
			val_ = *buf;
		}

		/// Constructor uses implicit conversion of int to double.
		Conv( const float& arg )
			: val_( arg )
		{;}

		/**
		 * This is the size used in the serialized form.
		 */
		unsigned int size() const
		{
			return 1;
		}

		const double* ptr() const
		{
			return &val_;
		}

		const float operator*() const {
			return val_;
		}

		unsigned int val2buf( double* buf ) const {
			*buf = val_;
			return 1;
		}

		static void str2val( float& val, const string& s ) {
			val = atoi( s.c_str() );
		}

		static void val2str( string& s, const float& val ) {
			stringstream ss;
			ss << val;
			s = ss.str();
		}

		static string rttiType() {
			return "float";
		}
	private:
		double val_;
};

/**
 * The template specialization of Conv< int > sets up alignment on
 * word boundaries by storing the data as a double. 
 */
template<> class Conv< int >
{
	public:
		/// Constructor assumes that the buffer points to a double
		Conv( const double* buf )
		{
			assert( buf );
			val_ = *buf;
		}

		/// Constructor uses implicit conversion of int to double.
		Conv( const int& arg )
			: val_( arg )
		{;}

		/**
		 * This is the size used in the serialized form.
		 */
		unsigned int size() const
		{
			return 1;
		}

		const double* ptr() const
		{
			return &val_;
		}

		const int operator*() const {
			return val_;
		}

		unsigned int val2buf( double* buf ) const {
			*buf = val_;
			return 1;
		}

		static void str2val( int& val, const string& s ) {
			val = atoi( s.c_str() );
		}

		static void val2str( string& s, const int& val ) {
			stringstream ss;
			ss << val;
			s = ss.str();
		}

		static string rttiType() {
			return "int";
		}
	private:
		double val_;
};

/**
 * The template specialization of Conv< unsigned short > sets up alignment on
 * word boundaries by storing the data as a double. 
 */
template<> class Conv< unsigned short >
{
	public:
		/// Constructor assumes that the buffer points to a double
		Conv( const double* buf )
		{
			assert( buf );
			val_ = *buf;
		}

		/// Constructor uses implicit conversion of int to double.
		Conv( const unsigned short& arg )
			: val_( arg )
		{;}

		/**
		 * This is the size used in the serialized form.
		 */
		unsigned int size() const
		{
			return 1;
		}

		const double* ptr() const
		{
			return &val_;
		}

		const unsigned short operator*() const {
			return val_;
		}

		unsigned int val2buf( double* buf ) const {
			*buf = val_;
			return 1;
		}

		static void str2val( unsigned short& val, const string& s ) {
			val = atoi( s.c_str() );
		}

		static void val2str( string& s, const unsigned short& val ) {
			stringstream ss;
			ss << val;
			s = ss.str();
		}

		static string rttiType() {
			return "unsigned short";
		}
	private:
		double val_;
};

/**
 * The template specialization of Conv< short > sets up alignment on
 * word boundaries by storing the data as a double. 
 */
template<> class Conv< short >
{
	public:
		/// Constructor assumes that the buffer points to a double
		Conv( const double* buf )
		{
			assert( buf );
			val_ = *buf;
		}

		/// Constructor uses implicit conversion of int to double.
		Conv( const short& arg )
			: val_( arg )
		{;}

		/**
		 * This is the size used in the serialized form.
		 */
		unsigned int size() const
		{
			return 1;
		}

		const double* ptr() const
		{
			return &val_;
		}

		const short operator*() const {
			return val_;
		}

		unsigned int val2buf( double* buf ) const {
			*buf = val_;
			return 1;
		}

		static void str2val( short& val, const string& s ) {
			val = atoi( s.c_str() );
		}

		static void val2str( string& s, const short& val ) {
			stringstream ss;
			ss << val;
			s = ss.str();
		}

		static string rttiType() {
			return "short";
		}
	private:
		double val_;
};

/**
 * The template specialization of Conv< bool > sets up alignment on
 * word boundaries by storing the bool as a double. 
 */
template<> class Conv< bool >
{
	public:
		Conv( const double* buf )
		{
			assert( buf );
			val_ = *buf;
		}

		/// Constructor uses implicit conversion of bool to double.
		Conv( const bool& arg )
			: val_( arg )
		{;}

		/**
		 * This is the size used in the serialized form.
		 */
		unsigned int size() const
		{
			return 1;
		}

		const double* ptr() const
		{
			return &val_;
		}

		const bool operator*() const {
			return ( val_ > 0.5 );
		}

		unsigned int val2buf( double* buf ) const {
			*buf = val_;
			return 1;
		}

		static void str2val( bool& val, const string& s ) {
			if ( s == "0" || s == "false" || s == "False" )
				val = 0;
			else
				val = 1;
		}

		static void val2str( string& s, const bool& val ) {
			if ( val > 0.5 )
				s = "1";
			else
				s = "0";
		}

		static string rttiType() {
			return "bool";
		}
	private:
		double val_;
};

template<> class Conv< PrepackedBuffer >
{
	public:
		Conv( const double* buf )
			: val_( buf )
		{;}

		Conv( const PrepackedBuffer& arg )
			: val_( arg )
		{;}

		/**
		 * This is the size of the double* converted array
		 */
		unsigned int size() const
		{
			return val_.size();
		}

		const PrepackedBuffer& operator*() const {
			return val_;
		}

		const double* ptr() const
		{
			return val_.ptr();
		}

		/**
		 * Converts data contents into char* buf. Buf must be allocated
		 * ahead of time.
		 * Needs to be specialized for variable size and pointer-containing
		 * types T.
		 * returns size of newly filled buffer.
		 */
		unsigned int val2buf( double* buf ) const {
			return val_.conv2buf( buf );
		}

		static void str2val( PrepackedBuffer& val, const string& s ) {
			; // Doesn't work.
		}

		static void val2str( string& s, const PrepackedBuffer& val ) {
			; // Doesn't work.
		}

		static string rttiType() {
			return "bad";
		}
	private:
		PrepackedBuffer val_;
};

/**
 * Trying to do a partial specialization.
 * This works with anything that has a uniform size.
 * Assume strings are the only exception.
 * The first double in the vector vec_ holds the # of data entries that
 * follow it. This excludes the space for the size entry itself.
 */
template< class T > class Conv< vector< T > >
{
	public:
		Conv( const double* buf )
		{
			unsigned int numEntries = *buf;
			unsigned int numDoubles = 
				1 + ( sizeof( T ) * numEntries - 1 ) / sizeof( double );

			vec_.insert( vec_.end(), buf, buf + numDoubles + 1 );
		}

		Conv( const vector< T >& arg )
		{
			if ( arg.size() == 0 ) {
				vec_.resize( 1 );
				vec_[0] = 0;
				return;
			}

			unsigned int entrySize = sizeof( T );
			unsigned int numDoubles = 
				1 + ( entrySize * arg.size() - 1 ) / sizeof( double );
			// One extra for the size field.
			vec_.resize( 1 + numDoubles );
			vec_[0] = arg.size();
			memcpy( &vec_[1], &arg[0], arg.size() * entrySize );
		}

		/** 
		 * Returns ptr to entire array
		 * We rely on the constructors to ensure size is at least 1.
		 */
		const double* ptr() const
		{
			return &vec_[0];
		}

		/** 
		 * Size of returned array in doubles.
		 */
		unsigned int size() const
		{
			return vec_.size();
		}

		const vector< T > operator*() const {
			if ( vec_.size() > 0 ) {
				unsigned int numEntries = vec_[0];
				vector< T > ret( numEntries );
				memcpy( &ret[0], &vec_[1], sizeof( T ) * numEntries );
				return ret;
			}
			return vector< T >();
		}

		unsigned int val2buf( double* buf ) const {
			memcpy( buf, &vec_[0], vec_.size() * sizeof( double ) );
			return vec_.size();
		}

		static void str2val( vector< T >& val, const string& s ) {
			cout << "Specialized Conv< vector< T > >::str2val not done\n";
		}

		static void val2str( string& s, const vector< T >& val ) {
			cout << "Specialized Conv< vector< T > >::val2str not done\n";
		}
		static string rttiType() {
			string ret = "vector<" + Conv< T >::rttiType() + ">";
			return ret;
		}
	private:
		vector< double > vec_;
};

/// Here entry 0 is the # of entries, and entry 1 is the total size of
/// the Conv, including two places for # entry 0 and entry 1.
template<> class Conv< vector< string > >
{
	public:
		Conv( const double* buf )
		{
			// unsigned int numEntries = buf[0];
			unsigned int size = buf[1];
			vec_.resize( 0 );
			vec_.insert( vec_.end(), buf, buf + size );
		}

		Conv( const vector< string >& arg )
		{
			if ( arg.size() == 0 ) {
				vec_.resize( 0 );
				return;
			}
			unsigned int totNumChars = 0;
			for ( unsigned int i = 0; i < arg.size(); ++i ) {
				totNumChars += arg[i].length() + 1;
			}
			unsigned int charsInDoubles = 
				1 + ( totNumChars - 1 )/sizeof( double );

			vec_.resize( 2 + charsInDoubles );
			vec_[0] = arg.size();
			vec_[1] = vec_.size();
			char* ptr = reinterpret_cast< char*> (&vec_[ 2 ] );
			for (unsigned int i = 0; i < arg.size(); ++i ) {
				strcpy( ptr, arg[i].c_str() );
				ptr += arg[i].length() + 1;
			}
		}

		/** 
		 * Returns ptr to entire array
		 */
		const double* ptr() const
		{
			if ( vec_.size() > 0 )
				return &vec_[0];
			return 0;
		}

		/** 
		 * Size of returned array in doubles.
		 */
		unsigned int size() const
		{
			return vec_.size();
		}

		const vector< string > operator*() const {
			if ( vec_.size() > 1 ) {
				unsigned int numEntries = vec_[0];
				vector< string > ret( numEntries );
				const char* ptr = reinterpret_cast< const char* > (&vec_[ 2 ] );
				for (unsigned int i = 0; i < numEntries; ++i ) {
					ret[i] = ptr;
					ptr += ret[i].length() + 1;
				}
				return ret;
			}
			return vector< string >();
		}

		unsigned int val2buf( double* buf ) const {
			if ( vec_.size() > 0 ) {
				memcpy( buf, &vec_[0], vec_.size() * sizeof( double ) );
				return vec_.size();
			}
			return 0;
		}

		static void str2val( vector< string >& val, const string& s ) {
			cout << "Specialized Conv< vector< T > >::str2val not done\n";
		}

		static void val2str( string& s, const vector< string >& val ) {
			cout << "Specialized Conv< vector< T > >::val2str not done\n";
		}
		static string rttiType() {
			string ret = "vector<" + Conv< string >::rttiType() + ">";
			return ret;
		}
	private:
		vector< double > vec_;
};

#endif // _CONV_H
