/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _FTYPE1_H
#define _FTYPE1_H


// Forward declaration of send1 needed for the proxyFunc.
template < class T > void send1( Eref e, Slot src, T v );

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
			Ftype1()
				: Ftype( "ftype1" )
			{
				addSyncFunc( RFCAST( &( Ftype1< T >::syncFunc ) ) );
				addAsyncFunc( RFCAST( &( Ftype1< T >::asyncFunc ) ) );
				addProxyFunc( RFCAST( &( Ftype1< T >::proxyFunc ) ) );
			}

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
			
			void* copyIntoArray( const void* orig, const unsigned int num, const unsigned int numCopies ) const
			{
				if ( num > 1 ) {
					//not done....num*numCopies
					T* ret = new T[ num ];
					const T* optr = static_cast< const T* >( orig );
					for ( unsigned int i = 0; i < num; i++ )
						ret[ i ] = optr[ i ];
					return ret;
				} else {
					T* ret = new T[ numCopies ];
					for ( unsigned int i = 0; i < numCopies; i++ ){
						ret[ i ] = *( new T( *static_cast< const T* >( orig ) ) );
					}
					return ret;
				}
			}
			
			void destroy( void* data, bool isArray ) const
			{
				if ( isArray ){
					T* del = static_cast< T* >( data );
					delete[] del;
				}
				else{
					T* del = static_cast< T* >( data );
					delete del;
				}
			}

			virtual bool get( Eref, const Finfo* f, T& v )
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
			 *
			 * Jul 2008: I think actually this is OK with specific conn
			 * indices, but does not handle AnyIndex
			 */
			virtual bool set( Eref e, const Finfo* f, T v ) const {
				void (*set)( const Conn*, T v ) =
					reinterpret_cast< void (*)( const Conn*, T ) >(
									f->recvFunc() );
				SetConn c( e );
				set( &c, v );
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
			bool strGet( Eref e, const Finfo* f, string& s )
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
			bool strSet( Eref e, const Finfo* f, const string& s )
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
            	static std::string s = Ftype::full_type(typeid(T));
				return s;
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
			// template <class MYCHAR> int lr_pair_type<MYCHAR>::id_string;
			static void proxyFunc(
				const Conn* c, const void* data, Slot slot )
			{
				T v;
				// Serializer< T > s( data );
				Serializer< T >::unserialize( v, data );
				send1< T >( c->target(), slot, v );
			}

			/**
			 * This function inserts data into the outgoing buffer.
			 * This variant is used when the data is synchronous: sent
			 * every clock step, so that the sequence is fixed.
			 */
			static void syncFunc( const Conn* c, T value ) {
				void* data = getParBuf( c, Serializer< T >::serialSize( value ) ); 
				// Serializer< T > s( data );
				Serializer< T >::serialize( data, value );
			}

			/**
			 * This variant is used for asynchronous data, where data
			 * is sent in at unpredictable stages of the simulation. It
			 * therefore adds additional data to identify the message
			 * source
			 */
			static void asyncFunc( const Conn* c, T value ) {
				void* data = getAsyncParBuf( c, Serializer< T >::serialSize( value ) );
				// Serializer< T > s( data );
				Serializer< T >::serialize( data, value );
			}
};

#endif // _FTYPE1_H
