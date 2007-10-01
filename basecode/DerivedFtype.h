/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _DERIVED_FTYPE_H
#define _DERIVED_FTYPE_H

class Ftype0: public Ftype
{
		public:
			unsigned int nValues() const {
				return 0;
			}
			
			bool isSameType( const Ftype* other ) const {
				return ( dynamic_cast< const Ftype0* >( other ) != 0 );
			}

			static bool isA ( const Ftype* other ) {
				return ( dynamic_cast< const Ftype0* >( other ) != 0 );
			}

			size_t size() const
			{
				return 0;
			}

			static const Ftype* global() {
				static Ftype* ret = new Ftype0();
				return ret;
			}

			RecvFunc recvFunc() const {
				return 0;
			}

			RecvFunc trigFunc() const {
				return 0;
			}

			virtual std::string getTemplateParameters() const
			{
				return "none";
			}
			
			///////////////////////////////////////////////////////
			// Here we define the functions for serializing data
			// for parallel messaging.
			///////////////////////////////////////////////////////
			/*
			static const void* incomingFunc(
				const Element* e, const void* data, unsigned int index )
			{
				send0( e, index );
				return data;
			}
			
			void appendIncomingFunc( vector< IncomingFunc >& vec )
					const {
				vec.push_back( &incomingFunc );
			}

			static void outgoingFunc( const Conn& c ) {
				// here the getParBuf just sticks in the id of the 
				// message. No data is sent.
				getParBuf( c, 0 ); 
			}

			void appendOutgoingFunc( vector< RecvFunc >& vec ) const {
				vec.push_back( &outgoingFunc );
			}
			*/

			/**
			 * This function extracts the value for this field from
			 * the data, and executes the function call for its
			 * target Conn. It returns the data pointer set to the
			 * next field. Here we don't have any arguments so the function
			 * just executes the target function.
			 */
			static const void* incomingFunc(
				const Conn& c, const void* data, RecvFunc rf )
			{
				rf( c );
				return data;
			}

			/**
			 * This function inserts data into the outgoing buffer.
			 * This variant is used when the data is synchronous: sent
			 * every clock step, so that the sequence is fixed.
			 * For the Ftype0, this function is pretty useless: it 
			 * amounts to a function executed without args every dt.
			 * We define it here for compiler satisfaction.
			 */
			static void outgoingSync( const Conn& c ) {
				;
			}

			/**
			 * This variant is used for asynchronous data, where data
			 * is sent in at unpredictable stages of the simulation. 
			 * As there is no additonal data, it just inserts the
			 * index of the conn into the data buffer by the getAsyncParBuf
			 * function. 
			 */
			static void outgoingAsync( const Conn& c ) {
				getAsyncParBuf( c, 0 ); 
			}

			/// Returns the statically defined incoming func
			IncomingFunc inFunc() const {
				return this->incomingFunc;
			}
			/*
			void inFunc( vector< IncomingFunc >& ret ) const {
				ret.push_back( this->incomingFunc );
			}
			*/

			/// Returns the statically defined outgoingSync function
			void syncFunc( vector< RecvFunc >& ret ) const {
				ret.push_back( RFCAST( this->outgoingSync ) );
			}

			/// Returns the statically defined outgoingAsync function
			void asyncFunc( vector< RecvFunc >& ret ) const {
				ret.push_back( RFCAST( this->outgoingAsync ) );
			}
};


/**
 * This function has to be specialized for each Ftype
 * that we wish to be able to convert. Otherwise it
 * reports failure.
 * These functions are defined in strconv.cpp
 */
template< class T >bool val2str( T v, string& s );
template<> bool val2str< string >( string v, string& ret);
template<> bool val2str< int >( int v, string& ret);
template<> bool val2str< unsigned int >( unsigned int v, string& ret);
template<> bool val2str< double >( double v, string& ret);
template<> bool val2str< Id >( Id v, string& ret);
template<> bool val2str< bool >( bool v, string& ret);
template<> bool val2str< vector< string > >(
	vector< string > v, string& ret);
template<> bool val2str< const Ftype* >(
	const Ftype* f, string& ret );

template<> bool val2str< ProcInfo >( ProcInfo v, string& ret);

template< class T >bool val2str( T v, string& s ) {
	s = "";
	return 0;
}

/**
 * This function has to be specialized for each Ftype
 * that we wish to be able to convert. Otherwise it
 * reports failure.
 */
template< class T > bool str2val( const string& s, T& v );
template<> bool str2val< string >( const string& v, string& ret);
template<> bool str2val< int >( const string& s, int& ret );
template<> bool str2val< unsigned int >( 
	const string& s, unsigned int& ret );
template<> bool str2val< double >( const string& s, double& ret );
template<> bool str2val< Id >( const string& s, Id& ret );
template<> bool str2val< bool >( const string& s, bool& ret );
template<> bool str2val< vector< string > >(
	const string& s, vector< string >& ret );
template<> bool str2val< const Ftype* >( 
	const string& s, const Ftype* &ret );
template<> bool str2val< ProcInfo >( const string& s, ProcInfo& ret );

template< class T > bool str2val( const string& s, T& v ) {
	cerr << "This is the default str2val.\n";
	return 0;
}

/**
 * Converts serialized data into defined types.
 * This function has to be specialised for many Ftypes like
 * vectors and strings. This default works only for simple
 * types.
 * Returns the data pointer incremented to the next field.
 */
template< class T > const void* unserialize( T& value, const void* data )
{
	const T* temp = static_cast< const T* >( data );
	value = *temp;
	return static_cast< const void* >( temp + 1 );
}

/**
 * Converts defined types into serialized data.
 * Puts the value into the data.
 * This function has to be specialised for many Ftypes like
 * vectors and strings. This default works only for simple
 * types.
 * Returns data pointer incremented to end of this field.
 */
template< class T > void* serialize( void* data, const T& value )
{
	T* temp = static_cast< T* >( data );
	*temp = value;
	return temp + 1;
}

/**
 * Gets the serialized size of the data type. Usually a simple sizeof,
 * but when handling complex types it will need to be specialized.
 */
template< class T > size_t serialSize( const T& value )
{
	return sizeof( T );
}


/**
 * The Ftype1 is the most used Ftype as it handles values.
 * This is still a virtual base class as it lacks the
 * functions for recvFunc and trigFunc. Those are specialized for
 * Value, Array, and Nest derived Ftypes.
 * It also introduces the get function that needs to be specialized
 * depending on the Finfo type.
 */
template < class T > class Ftype1: public Ftype
{
		public:
			unsigned int nValues() const {
				return 1;
			}
			
			bool isSameType( const Ftype* other ) const {
				return ( dynamic_cast< const Ftype1< T >* >( other ) != 0 );
			}
			
			static bool isA( const Ftype* other ) {
				return ( dynamic_cast< const Ftype1< T >* >( other ) != 0 );
			}

			size_t size() const
			{
				return sizeof( T );
			}

			void* create( const unsigned int num ) const
			{
				if ( num > 1 )
						return new T[ num ];
				else
						return new T;
			}

			/**
			 * Copies the data pointed to by orig into a new pointer.
			 * the size 'num' must be the size of orig, and will be
			 * the size of the return data.
			 * I wonder if there is a better way to do the array
			 * copy. Usually array ops can use economies of scale,
			 * but here the constructor followed by assignment is
			 * actually slower than the single constructor with
			 * initializer.
			 */
			void* copy( const void* orig, const unsigned int num ) const
			{
				if ( num > 1 ) {
					T* ret = new T[ num ];
					const T* optr = static_cast< const T* >( orig );
					for ( unsigned int i = 0; i < num; i++ )
						ret[ i ] = optr[ i ];
					return ret;
				} else {
					return new T( *static_cast< const T* >( orig ) );
				}
			}
			
			void* copyIntoArray( const void* orig, const unsigned int num, unsigned int numCopies ) const
			{
				if ( num > 1 ) {
					//not done
					T* ret = new T[ num ];
					const T* optr = static_cast< const T* >( orig );
					for ( unsigned int i = 0; i < num; i++ )
						ret[ i ] = optr[ i ];
					return ret;
				} else {
					T* ret = new T[ numCopies ];
					for ( unsigned int i = 0; i < numCopies; i++ )
						ret[ i ] = *( new T( *static_cast< const T* >( orig ) ) );
					return ret;
				}
			}

			void destroy( void* data, bool isArray ) const
			{
				if ( isArray )
					delete[] static_cast< T* >( data );
				else
					delete static_cast< T* >( data );
			}

			virtual bool get( const Element* e, const Finfo* f, T& v )
			const 
			{
					return 0;
			}

			/**
			 * This variant works for most kinds of Finfo,
			 * so we make a default version.
			 * In a few cases we need to specialize it because
			 * we need extra info from the Finfo.
			 * This function makes the strong assumption that
			 * we will NOT look up any dynamic finfo, and will
			 * NOT use the conn index in any way. This would
			 * fail because the conn index given is a dummy one.
			 * We do a few assert tests in downstream functions
			 * for any such attempts
			 * to search for a Finfo based on the index.
			 */
			virtual bool set( Element* e, const Finfo* f, T v ) const {
				void (*set)( const Conn&, T v ) =
					reinterpret_cast< void (*)( const Conn&, T ) >(
									f->recvFunc() );
				Conn c( e, MAXUINT );
				set( c, v );
				return 1;
			}

			static const Ftype* global() {
				static Ftype* ret = new Ftype1< T >();
				return ret;
			}

			RecvFunc recvFunc() const {
				return 0;
			}

			RecvFunc trigFunc() const {
				return 0;
			}

			/**
			 * This is a virtual function that gets the value,
			 * converts it to a string, and puts this into the
			 * string location specified.
			 * Returns true on success.
			 */
			bool strGet( const Element* e, const Finfo* f, string& s )
				const
			{
				T val;
				if ( this->get( e, f, val ) )
						return val2str( val, s );
				return 0;
			}

			/**
			 * This is a virtual function that takes a string,
			 * converts it to a value, and assigns it to a field.
			 * Returns true on success.
			 */
			bool strSet( Element* e, const Finfo* f, const string& s )
					const
			{
				T val;
				if ( str2val( s, val ) )
					return this->set( e, f, val );
				return 0;
			}

			/**
			 * This virtual function returns a void* to an allocated
			 * T instance of the converted string. If the conversion
			 * fails it returns 0.
			 */
			void* strToIndexPtr( const string& s ) const {
				T ret;
				if ( str2val( s, ret ) ) {
					return new T( ret );
				}
				return 0;
			}

			virtual std::string getTemplateParameters() const
			{
				return Ftype::full_type(std::string(typeid(T).name()));
			}
			
			///////////////////////////////////////////////////////
			// Here we define the functions for serializing data
			// for parallel messaging.
			// There are some specializations needed when T is a
			// vector or a string.
			///////////////////////////////////////////////////////

			/**
			 * This function extracts the value for this field from
			 * the data, and executes the function call for its
			 * target Conn. It returns the data pointer set to the
			 * next field.
			 */
			static const void* incomingFunc(
				const Conn& c, const void* data, RecvFunc rf )
			{
				T ret;
				data = unserialize< T >( ret, data );
				( reinterpret_cast< void (*)( const Conn&, T ) >( rf ) )(
					c, ret );
				return data;
			}

			/**
			 * This function inserts data into the outgoing buffer.
			 * This variant is used when the data is synchronous: sent
			 * every clock step, so that the sequence is fixed.
			 */
			static void outgoingSync( const Conn& c, T value ) {
				void* data = getParBuf( c, serialSize< T >( value ) ); 
				serialize< T >( data, value );
			}

			/**
			 * This variant is used for asynchronous data, where data
			 * is sent in at unpredictable stages of the simulation. It
			 * therefore adds additional data to identify the message
			 * source
			 */
			static void outgoingAsync( const Conn& c, T value ) {
				void* data = getAsyncParBuf( c, serialSize< T >( value ) ); 
				serialize< T >( data, value );
			}

			/// Returns the statically defined incoming func
			IncomingFunc inFunc() const {
				return this->incomingFunc;
			}
			/*
			void inFunc( vector< IncomingFunc >& ret ) const {
				ret.push_back( this->incomingFunc );
			}
			*/

			/// Returns the statically defined outgoingSync function
			void syncFunc( vector< RecvFunc >& ret ) const {
				ret.push_back( RFCAST( this->outgoingSync ) );
			}

			/// Returns the statically defined outgoingAsync function
			void asyncFunc( vector< RecvFunc >& ret ) const {
				ret.push_back( RFCAST( this->outgoingAsync ) );
			}
};

#endif // _DERIVED_FTYPE_H
