/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _ARRAY_FTYPE_H
#define _ARRAY_FTYPE_H


template < class T > class ArrayFtype1: public Ftype1<T>
{
		public:

			/**
			 * The arrayRecvFunc here uses the DynamicFinfo
			 * to keep track of the original setFunc of the ArrayFinfo,
			 * and to hold the index.
			 */
			static void arrayRecvFunc( const Conn& c, T v )
			{
				const DynamicFinfo* f = getDF( c );
				void (*rf)( const Conn& c, T v, unsigned int i ) =
					reinterpret_cast<
					void (*)( const Conn&, T, unsigned int ) > (
									f->innerSetFunc() );
				rf( c, v, f->arrayIndex() );
			}

			/**
			 * This is the recvFunc for triggering outgoing messages
			 * with the array value.
			 * This function refers to one of the dynamicFinfos
			 * on the target object to figure out what has to be
			 * done for the return message.
			 * The DynamicFinfo provides three things:
			 * - Lookup from Conn.
			 * - The index of the array entry.
			 * - The GetFunc for the array type.
			 * - The message response handling for later adds.
			 */
			static void arrayTrigFunc( const Conn& c )
			{
				const DynamicFinfo* f = getDF( c );
				T (*getArray)( const Element*, unsigned int i ) =
					reinterpret_cast< 
					T (*)( const Element*, unsigned int i ) >
					( f->innerGetFunc() );

				Element* e = c.targetElement();
				send1< T >( e, f->srcIndex(), 
						getArray( e, f->arrayIndex() ) );
			}

			RecvFunc recvFunc() const {
				return reinterpret_cast< RecvFunc >(
						&arrayRecvFunc );
			}

			RecvFunc trigFunc() const {
				return &arrayTrigFunc;
			}

			static const Ftype* global() {
				static Ftype* ret = new ArrayFtype1< T >();
				return ret;
			}

			/**
			 * This may only be called from a DynamicFinfo
			 */
			bool get( const Element* e, const Finfo* f, T& v ) const {
				const DynamicFinfo* df =
						dynamic_cast< const DynamicFinfo* >( f );
				assert( df != 0 );
				T ( *g )( const Element*, unsigned int ) =
					reinterpret_cast<
					T (*)( const Element*, unsigned int ) >(
							df->innerGetFunc()
					);
				v = g( e, df->arrayIndex() );
				return 1;
			}

			/**
			 * This may only be called from a DynamicFinfo.
			 * It specializes the generic version in the parent Ftype1
			 */
			bool set( Element* e, const Finfo* f, T v ) const {
				const DynamicFinfo* df =
						dynamic_cast< const DynamicFinfo* >( f );
				assert( df != 0 );

				void (*set)( const Conn&, T v, unsigned int ) =
					reinterpret_cast<
					void (*)( const Conn&, T, unsigned int ) >(
						df->innerSetFunc()
					);
				Conn c( e, 0 );
				set( c, v, df->arrayIndex() );
				return 1;
			}
};

#endif // _ARRAY_FTYPE_H
