/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _VALUE_FTYPE_H
#define _VALUE_FTYPE_H

template < class T > class ValueFtype1: public Ftype1<T>
{
		public:

			/**
			 * This is the recvFunc for triggering outgoing messages
			 * with the field value.
			 * This function refers to one of the dynamicFinfos
			 * on the target object to figure out what has to be
			 * done for the return message.
			 * The DynamicFinfo provides three things:
			 * - Lookup from Conn.
			 * - The recvFunc deposited by the requesting message.
			 * - A pointer back to the original Finfo. This gives
			 *   us access to its getFunc so we can extract the value.
			 */
			static void valueTrigFunc( const Conn* c ) {
				const DynamicFinfo* f = getDF( c );
				T ( *getValue )( Eref ) =
					reinterpret_cast< T (*)( Eref ) > (
							f->innerGetFunc()
					);
				Eref e = c->target();
				///\todo Hack here to be fixed by getting a proper slot
				// send1<T>( e, Slot( f->srcIndex(), 0 ), getValue( e ) );
				sendBack1< T >( c, 
					Slot( c->targetMsg(), 0 ), getValue( e ) );
			}

			/**
			 * This should never be requested by anybody. The
			 * recvFunc for ValueFinfo is passed in by the user, so
			 * this function is never used.
			 */
			RecvFunc recvFunc() const {
				assert( 0 );
				return 0;
			}

			RecvFunc trigFunc() const {
				return valueTrigFunc;
			}

			static const Ftype* global() {
				static Ftype* ret = new ValueFtype1< T >();
				return ret;
			}

			/**
			 * This gets the field value into the typed value v.
			 * It returns true on success.
			 * This could use either a DynamicFinfo already set up
			 * to do messaging, or it might use the original Value
			 * Finfo, as the second argument.
			 */
			bool get( Eref e, const Finfo* f, T& v ) const {
				GetFunc gf = 0;
				const DynamicFinfo* df =
						dynamic_cast< const DynamicFinfo* >( f );
				if ( df ) {
					gf = df->innerGetFunc();
				} else {
					const ValueFinfo* vf =
						dynamic_cast< const ValueFinfo* >( f );
					if ( vf ) {
						gf = vf->innerGetFunc();
					}
				}
				assert ( gf != 0 );
				T (*g )( Eref ) = reinterpret_cast< T (*)( Eref ) >( gf );
				v = g( e );
				return 1;
			}

			const Ftype* baseFtype() const {
				return Ftype1< T >::global();
			}
};

#endif // _VALUE_FTYPE_H
