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
};


/**
 * This function has to be specialized for each Ftype
 * that we wish to be able to convert. Otherwise it
 * reports failure.
 */
template< class T >bool val2str( T v, string& s ) {
	s = "";
	return 0;
}

/**
 * This function has to be specialized for each Ftype
 * that we wish to be able to convert. Otherwise it
 * reports failure.
 */
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
			 * the data, and returns the data pointer set to the
			 * next field.
			 */
			static const void* incomingFunc(
				const Element* e, const void* data, unsigned int index )
			{
				T ret;
				data = unserialize< T >( ret, data );
				send1< T >( e, index, ret );
				return data;
			}
			
			void appendIncomingFunc( vector< IncomingFunc >& vec )
					const {
				vec.push_back( &incomingFunc );
			}

			static void outgoingFunc( const Conn& c, T value ) {
				void* data = getParBuf( c, serialSize< T >( value ) ); 
				serialize< T >( data, value );
			}

			void appendOutgoingFunc( vector< RecvFunc >& vec ) const {
				vec.push_back( RFCAST( &outgoingFunc ) );
			}

};

#endif // _DERIVED_FTYPE_H
