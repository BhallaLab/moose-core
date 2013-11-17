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
		 * Size, in units of sizeof(double). So a double would be 1,
		 * something with 1 chars would be 1 as well, but something
		 * with 9 chars would be 2.
		 */
		static unsigned int size( const T& val )
		{
			return 1 + ( sizeof( T ) - 1 ) / sizeof( double );
		}


		static const T& buf2val( const double** buf ) {
			T* ret = *reinterpret_cast< const T* >( *buf );
			*buf += size( *buf );
			return *ret;
		}

		/**
		 * Converts data contents into double* buf. Buf must be allocated
		 * ahead of time. Returns size of value.
		 * Needs to be specialized for variable size and pointer-containing
		 * types T.
		 */
		static void val2buf( const T& val, double** buf ) {
			**reinterpret_cast< T >( **buf ) = val;
			*buf += size( val );
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
                        if ( typeid( T ) == typeid( char ))
                                return "char";
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
			return typeid( T ).name(); // this is not portable but may be more useful than "bad"
		}

	private:
};

/**
 * This stores the data as the equivalent of a char* string, terminated with
 * the usual \0, but allocates it as a double[].
 */
template<> class Conv< string >
{
	public:
		/**
		 * This is the size used in the serialized form, as a double*
		 * Note that we do some ugly stuff to get alignment on 8-byte
		 * boundaries.
		 * We need to have strlen + 1 as a minimum.
		 */
		static unsigned int size( const string& val )
		{
			return ( 1 + val.length() ) / sizeof( double );
		}

		static const string& buf2val( const double** buf ) {
			static string ret;
			ret = reinterpret_cast< const char* >( *buf );
			*buf += size( ret );
			return ret;
		}

		/**
		 * Converts data contents into double* buf. Buf must be allocated
		 * ahead of time. Returns size of value.
		 * Needs to be specialized for variable size and pointer-containing
		 * types T.
		 */
		static void val2buf( const string& val, double** buf ) {
			char* temp = reinterpret_cast< char* >( *buf );
			strcpy( temp, val.c_str() );
			*buf += val.length();
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
};

/**
 * The template specialization of Conv< unsigned int > sets up alignment on
 * word boundaries by storing the data as a double. 
 */
template<> class Conv< double >
{
	public:
		/**
		 * This is the size used in the serialized form.
		 */
		static unsigned int size( double val )
		{
			return 1;
		}

		static const double buf2val( const double** buf ) {
			double ret = **buf;
			(*buf)++;
			return ret;
		}
		static void val2buf( float val, double** buf ) {
			**buf = val;
			(*buf)++; 
		}

		static void str2val( float val, const string& s ) {
			val = atof( s.c_str() );
		}

		static void val2str( string& s, float val ) {
			stringstream ss;
			ss << val;
			s = ss.str();
		}

		static string rttiType() {
			return "double";
		}
	private:
};

/**
 * The template specialization of Conv< unsigned int > sets up alignment on
 * word boundaries by storing the data as a double. 
 */
template<> class Conv< float >
{
	public:
		/**
		 * This is the size used in the serialized form.
		 */
		static unsigned int size( float val )
		{
			return 1;
		}

		static const float buf2val( const double** buf ) {
			float ret = **buf;
			(*buf)++;
			return ret;
		}
		static void val2buf( float val, double** buf ) {
			**buf = val;
			(*buf)++; 
		}

		static void str2val( float val, const string& s ) {
			val = atof( s.c_str() );
		}

		static void val2str( string& s, float val ) {
			stringstream ss;
			ss << val;
			s = ss.str();
		}

		static string rttiType() {
			return "float";
		}
	private:
};


/**
 * The template specialization of Conv< unsigned int > sets up alignment on
 * word boundaries by storing the data as a double. 
 */
template<> class Conv< int >
{
	public:
		/**
		 * This is the size used in the serialized form.
		 */
		static unsigned int size( int val )
		{
			return 1;
		}

		static const int buf2val( const double** buf ) {
			int ret = **buf;
			(*buf)++;
			return ret;
		}
		static void val2buf( int val, double** buf ) {
			**buf = val;
			(*buf)++; 
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
};

template<> class Conv< unsigned short >
{
	public:
		/**
		 * This is the size used in the serialized form.
		 */
		static unsigned int size( unsigned short val )
		{
			return 1;
		}

		static const unsigned short buf2val( const double** buf ) {
			unsigned short ret = **buf;
			(*buf)++;
			return ret;
		}
		static void val2buf( unsigned short val, double** buf ) {
			**buf = val;
			(*buf)++; 
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
};

template<> class Conv< short >
{
	public:
		/**
		 * This is the size used in the serialized form.
		 */
		static unsigned int size( short val )
		{
			return 1;
		}

		static const short buf2val( const double** buf ) {
			short ret = **buf;
			(*buf)++;
			return ret;
		}
		static void val2buf( short val, double** buf ) {
			**buf = val;
			(*buf)++; 
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
};

template<> class Conv< bool >
{
	public:
		/**
		 * This is the size used in the serialized form.
		 */
		static unsigned int size( bool val )
		{
			return 1;
		}

		static const bool buf2val( const double** buf ) {
			bool ret = (**buf > 0.5);
			(*buf)++;
			return ret;
		}
		static void val2buf( bool val, double** buf ) {
			**buf = val;
			(*buf)++; 
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
};

/**
 * The template specialization of Conv< Id > sets up alignment on
 * word boundaries by storing the Id as a double. It also deals with 
 * the string conversion issues.
 */

template<> class Conv< Id >
{
	public:
		/**
		 * This is the size used in the serialized form.
		 */
		static unsigned int size( Id val )
		{
			return 1;
		}

		static const Id buf2val( const double** buf ) {
			Id ret( **buf );
			(*buf)++;
			return ret;
		}
		static void val2buf( Id id, double** buf ) {
			**buf = id.value();
			(*buf)++; 
		}

		static void str2val( Id& val, const string& s ) {
			Id temp( s ); // converts the path
			val = temp;
		}

		static void val2str( string& s, const Id& val ) {
			s = val.path();
		}

		static string rttiType() {
			return "Id";
		}
	private:
};

/**
 * Still more specialized partial specialization.
 * This works with any field that has a uniform size.
 * Assume strings are the only exception.
 * The first double in the vector vec_ holds the # of data entries that
 * follow it. This excludes the space for the size entry itself.
 * This can be a ragged array: the number of entries in each vector need
 * not be the same.
 * The order of entries is:
 * 0		:  Dimension of left index, can be zero. As double.
 * 1:numDim	:  Dimensions of right index. Any and all can be zero. Double.
 * numDim+1 to total: Data, in condensed format.
 */
template< class T > class Conv< vector< vector< T > > >
{
	public:
		static unsigned int size( const vector< vector < T > > & val)
		{
			unsigned int ret = 0;
			for ( unsigned int i = 0; i < val.size(); ++i )
				ret += val[i].size();
			if ( ret > 0 )
				ret *= Conv< T >::size( val[0][0] );
			else  {
				T temp;
				ret *= Conv< T >::size( temp );
			}
			return ret;
		}
		static const vector< vector< T > >& buf2val( const double** buf )
		{
			static vector< vector< T > > ret;
			ret.clear();
			unsigned int numEntries = **buf++; // first entry is vec size
			unsigned int numDoubles = 1 + numEntries;
			for ( unsigned int i = 0; i < numEntries; ++i ) {
				unsigned int rowSize = **buf;
				(*buf)++;
				for ( unsigned int j = 0; j < rowSize; ++j )
					ret[i].push_back( Conv< T >::buf2val( buf ) );
			}
			return ret;
		}

		static void val2buf( const vector< vector< T > >& val, double**buf )
		{
			double* temp = *buf;
			*temp++ = val.size();
			for( int i = 0; i < val.size(); ++i ) {
				*temp++ = val[i].size();
				for ( int j = 0; j < val[i].size(); ++j ) {
					Conv< T >::val2buf( val[i][j], &temp );
				}
			}
		}

		static void str2val( vector< vector< T > >& val, const string& s ) {
			cout << "Specialized Conv< vector< vector< T > > >::str2val not done\n";
		}

		static void val2str( string& s, const vector< vector< T > >& val ) {
			cout << "Specialized Conv< vector< vector< T > > >::val2str not done\n";
		}
		static string rttiType() {
			string ret = "vector< vector<" + Conv< T >::rttiType() + "> >";
			return ret;
		}
	private:
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
