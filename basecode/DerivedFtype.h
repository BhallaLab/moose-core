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
};

#endif // _DERIVED_FTYPE_H
