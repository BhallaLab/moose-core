/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _FTYPE3_H
#define _FTYPE3_H

/**
 * The Ftype3 handles 3-argument functions.
 */
template < class T1, class T2, class T3 > class Ftype3: public Ftype
{
		public:
			unsigned int nValues() const {
				return 3;
			}
			
			bool isSameType( const Ftype* other ) const {
				return ( dynamic_cast< const Ftype3< T1, T2, T3 >* >( other ) != 0 );
			}
			
			static bool isA( const Ftype* other ) {
				return ( dynamic_cast< const Ftype3< T1, T2, T3 >* >( other ) != 0 );
			}

			size_t size() const
			{
				return sizeof( T1 ) + sizeof( T2 ) + sizeof( T3 );
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
				Element* e, const Finfo* f, T1 v1, T2 v2, T3 v3 ) const
			{

				void (*set)( const Conn&, T1 v1, T2 v2, T3 v3 ) =
					reinterpret_cast< 
						void (*)( const Conn&, T1 v1, T2 v2, T3 v3 )
					>(
									f->recvFunc()
					);
				Conn c( e, MAXUINT );
				set( c, v1, v2, v3 );
				return 1;
			}

			static const Ftype* global() {
				static Ftype* ret = new Ftype3< T1, T2, T3 >();
				return ret;
			}

			RecvFunc recvFunc() const {
				return 0;
			}

			RecvFunc trigFunc() const {
				return 0;
			}
};

#endif // _FTYPE3_H
