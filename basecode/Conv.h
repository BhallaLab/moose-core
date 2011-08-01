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

template<> class Conv< string >
{
	public:
		Conv( const char* buf )
		{
			if ( buf == 0 )
				val_ = "";
			else
				val_ = buf;
		}
		Conv( const double* dbuf )
		{
			const char* buf = reinterpret_cast< const char* >( dbuf );
			if ( buf == 0 )
				val_ = "";
			else
				val_ = buf;
		}

		Conv( const string& arg )
			: val_( arg )
		{;}

		const double* ptr() const
		{
			return reinterpret_cast< const double* >( val_.c_str() );
		}

		/**
		 * This is the size used in the serialized form, as a double*
		 * Note that we do some ugly stuff to get alignment on 8-byte
		 * boundaries.
		 * We need to have strlen + 1 as a minimum.
		 */
		unsigned int size() const
		{
			
			return ( 1 + ( val_.length() / sizeof( double ) ) );
		}

		const string& operator*() const {
			return val_;
		}

		unsigned int val2buf( double* dbuf ) const {
			char* buf = reinterpret_cast< char* >( dbuf );
			strcpy( buf, val_.c_str() );
			return size();
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
		string val_;
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
		 * This is the size of the char* converted array
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
			return reinterpret_cast< const double* >( &val_ );
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
 * The buffer here starts with the # of entries in the vector
 * Then it has the actual entries.
 */
template< class T > class Conv< vector< T > >
{
	public:
		Conv( const double* buf )
		{
			unsigned int numEntries = 
				*reinterpret_cast< const unsigned int* >( buf );
			buf += sizeof( unsigned int );
			assert( numEntries < UnreasonablyLargeArray );
			val_.resize( numEntries );
			size_ = sizeof( unsigned int );
			for ( unsigned int i = 0; i < numEntries; ++i ) {
				Conv< T > arg( buf );
				val_[ i ] = *arg;
				unsigned int temp = arg.size();
				buf += temp;
				size_ += temp;
			}
		}

		Conv( const vector< T >& arg )
			: val_( arg )
		{
			size_ = sizeof( unsigned int );
			for ( unsigned int i = 0; i < val_.size(); ++i ) {
				Conv< T > temp( val_[ i ] );
				size_ += temp.size();
			}
		}

		const double* ptr() const
		{
			return 0;
		}

		unsigned int size() const
		{
			return size_;
		}

		const vector< T >& operator*() const {
			return val_;
		}

		unsigned int val2buf( double* buf ) const {
			*( reinterpret_cast< unsigned int* >( buf ) ) = val_.size();
			buf += 1;
			for ( unsigned int i = 0; i < val_.size(); ++i ) {
				Conv< T > temp( val_[ i ] );
				buf += temp.val2buf( buf );
			}
			return size_;
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
		unsigned int size_;
		vector< T > val_;
};

#endif // _CONV_H
