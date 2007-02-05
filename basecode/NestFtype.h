/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _NEST_FTYPE_H
#define _NEST_FTYPE_H


template < class T > class NestFtype1: public Ftype1<T>
{
		public:

			/**
			 * The nestRecvFunc here uses the DynamicFinfo
			 * to keep track of the original setFunc of the
			 * eventual target of all the nesting,
			 * and to hold all the lookup info for the nesting.
			 * \todo Still issues with handling nested synapses.
			 */
			static void nestRecvFunc( const Conn& c, T v )
			{
				const DynamicFinfo* f = getDF( c );

				// This magic function returns a temporary element e 
				// with the appropriate data from indirection.
				void* data = f->traverseIndirection( 
								c.targetElement()->data() );
				// Element* e = f->traverseIndirection( c );
				SimpleElement* e( "temp", 0, 0, data );
				Conn temp( e, 0 );
				void (*rf)( const Conn& c, T v) =
					reinterpret_cast< void (*)( const Conn&, T ) > (
									f->innerSetFunc() );
				rf( temp, v );
			}
			

			/**
			 * This is the recvFunc for triggering outgoing messages
			 * with the nested value.
			 * This only works when the nest target Finfo provides
			 * a getFunc. This is something the DynamicFinfo knows,
			 * so it also knows when to refuse such a message
			 * request in DynamicFinfo::respondToAdd. So if things
			 * get here the getFunc should definitely exist.
			 * This function refers to one of the dynamicFinfos
			 * on the target object to figure out what has to be
			 * done for the return message.
			 * The DynamicFinfo provides three things:
			 * - Lookup from Conn.
			 * - The indirection info
			 * - The GetFunc for the nested type.
			 * - The message response handling for later adds.
			 */
			static void nestTrigFunc( const Conn& c, T v )
			{
				const DynamicFinfo* f = getDF( c );
				GetFunc gf = f->innerGetFunc();
				assert( gf );

				void* data = f->traverseIndirection(
								c.targetElement()->data() );

				SimpleElement se ( "temp", 0, 0, data );
				T (*getFunc)( const Element* ) =
					reinterpret_cast< 
					T (*)( const Element*) >( gf );

				Element* e2 = c.targetElement();
				send1< T >( e2, f->srcIndex(), getFunc( se ) );
			}

			RecvFunc recvFunc() const {
				return this->nestRecvFunc();
			}

			RecvFunc trigFunc() const {
				return this->nestTrigFunc();
			}

			static const Ftype* global() {
				static Ftype* ret = new NestFtype1< T >();
				return ret;
			}

			/**
			 * This may only be called from a DynamicFinfo
			 */
			bool get( const Element* e, const Finfo* f, T& v ) const {
				const DynamicFinfo* df =
						dynamic_cast< const DynamicFinfo* >( f );
				assert( df != 0 );
				T ( *g )( const Element* ) =
					reinterpret_cast< T (*)( const Element* ) >(
							df->innerGetFunc()
					);
				SimpleElement se( "temp",
					df->traverseIndirection( e->data() ) );
				v = g( se );
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

				void (*set)( const Conn&, T v ) =
					reinterpret_cast< void (*)( const Conn&, T ) >(
						f->innerSetFunc()
					);
				SimpleElement se( "temp",
					df->traverseIndirection( e->data() ) );
				Conn c( se, 0 );
				set( c, v );
				return 1;
			}
};

#endif // _NEST_FTYPE_H
