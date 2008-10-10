/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _FTYPE2_H
#define _FTYPE2_H

// Forward declaration of send2 needed for proxyFunc.
template < class T1, class T2 > void send2( Eref e, Slot src, T1 v1, T2 v2);
/**
 * The Ftype2 handles 2-argument functions.
 */
template < class T1, class T2 > class Ftype2: public Ftype
{
		public:
			Ftype2()
				: Ftype( "ftype2" )
			{
				addSyncFunc( RFCAST( &( Ftype2< T1, T2 >::syncFunc ) ) );
				addAsyncFunc( RFCAST( &( Ftype2< T1, T2 >::asyncFunc ) ) );
				addProxyFunc( RFCAST( &( Ftype2< T1, T2 >::proxyFunc ) ) );
			}

			unsigned int nValues() const {
				return 2;
			}
			
			bool isSameType( const Ftype* other ) const {
				return ( dynamic_cast< const Ftype2< T1, T2 >* >( other ) != 0 );
			}
			
			static bool isA( const Ftype* other ) {
				return ( dynamic_cast< const Ftype2< T1, T2 >* >( other ) != 0 );
			}

			size_t size() const
			{
				return sizeof( T1 ) + sizeof( T2 );
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
			virtual bool set(
				Eref e, const Finfo* f, T1 v1, T2 v2 ) const {

				void (*set)( const Conn*, T1 v1, T2 v2 ) =
					reinterpret_cast< 
						void (*)( const Conn*, T1 v1, T2 v2 )
					>(
									f->recvFunc()
					);
				SetConn c( e );
				set( &c, v1, v2 );
				return 1;
			}

			/**
			 * This is a virtual function that takes a string,
			 * converts it to two values, and assigns it to a field.
			 * Returns true on success.
			 * It will run into trouble if the contents are strings
			 * with spaces or commas.
			 */
			bool strSet( Eref e, const Finfo* f, const string& s )
					const
			{
				string::size_type pos = s.find_first_of( ", 	" );
				if ( pos == string::npos )
						return 0;
				if ( pos < 1 )
						return 0;
				string s1 = s.substr( 0, pos );
				pos = s.find_last_of( ", 	" );
				if ( pos >=  s.length() - 1 )
						return 0;
				
				string s2 = s.substr( pos );
				pos = s2.find_first_not_of( ", 	" );
				s2 = s2.substr( pos );
				
				T1 val1;
				if ( str2val( s1, val1 ) ) {
					T2 val2;
					if ( str2val( s2, val2 ) )
						return this->set( e, f, val1, val2 );
				}

				return 0;
			}

			static const Ftype* global() {
				static Ftype* ret = new Ftype2< T1, T2 >();

				return ret;
			}

			RecvFunc recvFunc() const {
				return 0;
			}

			RecvFunc trigFunc() const {
				return 0;
			}

          	std::string getTemplateParameters() const
			{
            	static std::string s = Ftype::full_type(typeid(T1))+","+Ftype::full_type(typeid(T2));
				return s;
			}
			
			///////////////////////////////////////////////////////
			// Here we define the functions for handling 
			// messages of this type for parallel messaging.
			///////////////////////////////////////////////////////
			/**
			 * This function extracts the value for this field from
			 * the data, and executes the function call for its
			 * target Conn. It returns the data pointer set to the
			 * next field.
			 */
			static void proxyFunc(
				const Conn* c, const void* data, Slot slot )
			{
				T1 v1;
				T2 v2;
				data = Serializer< T1 >::unserialize( v1, data );
				data = Serializer< T2 >::unserialize( v2, data );
				send2< T1, T2 >( c->target(), slot , v1, v2 );
			}

			/**
			 * This function inserts data into the outgoing buffer.
			 * This variant is used when the data is synchronous: sent
			 * every clock step, so that the sequence is fixed.
			 */
			static void syncFunc( const Conn* c, T1 v1, T2 v2 ) {
				unsigned int size1 = Serializer< T1 >::serialSize( v1 );
				unsigned int size2 = Serializer< T2 >::serialSize( v2 );
				void* data = getParBuf( c, size1 + size2 ); 
				data = Serializer< T1 >::serialize( data, v1 );
				Serializer< T2 >::serialize( data, v2 );
			}

			/**
			 * This variant is used for asynchronous data, where data
			 * is sent in at unpredictable stages of the simulation. It
			 * therefore adds additional data to identify the message
			 * source
			 */
			static void asyncFunc( const Conn* c, T1 v1, T2 v2 ) {
				unsigned int size1 = Serializer< T1 >::serialSize( v1 );
				unsigned int size2 = Serializer< T2 >::serialSize( v2 );
				// unsigned int size2 = serialSize< T2 >( v2 );
				void* data = getAsyncParBuf( c, size1 + size2 ); 
				data = Serializer< T1 >::serialize( data, v1 );
				Serializer< T2 >::serialize( data, v2 );
			}

			
			/*
			/// Returns the statically defined incoming func
			IncomingFunc inFunc() const {
				return this->incomingFunc;
			}
			*/
			/*
			void inFunc( vector< IncomingFunc >& ret ) const {
				ret.push_back( this->incomingFunc );
			}
			*/

			/*
			/// Returns the statically defined outgoingSync function
			void syncFunc( vector< RecvFunc >& ret ) const {
				ret.push_back( RFCAST( this->outgoingSync ) );
			}

			/// Returns the statically defined outgoingAsync function
			void asyncFunc( vector< RecvFunc >& ret ) const {
				ret.push_back( RFCAST( this->outgoingAsync ) );
			}
			*/
};

#endif // _FTYPE2_H
