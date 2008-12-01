/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _LOOKUP_FTYPE_H
#define _LOOKUP_FTYPE_H

/**
 * This class manages the type-specific aspects of lookup fields.
 * A lookup field may be an array
 * or a map, or anything else in which the desired value is referred
 * to by an index which could be any type. The most common cases are
 * lookup by an integral index, and lookup by a string name.
 *
 * The first type T1 refers to the type of the object being looked up.
 * The second type T2 refers to the type of the index.
 *
 * There are several use cases:
 *  Sending messages to or from a specific entry:
 *  	Here it first creates a DynamicFinfo that refers to the
 *  	specific entry, and this DynamicFinfo manages the messages.
 *  	The Dynamic Finfo has to store the index T2 as an allocated
 *  	pointer, in a void*. The job of our Ftype here is to do the
 *  	correct typecasting back.
 *  	The message for 'set' is a simple Ftype1<T1> message and 
 *  	assigns the value at the specified index.
 *  	The messages for 'get' are an incoming trigger message with 
 *  	no arguments. This tells the DynamicFinfo to send out a 
 *  	regular Ftype1<T1> message holding the field value at the
 *  	specific entry.
 *  Set and get onto a specific entry:
 *  	Again, we first make a DynamicFinfo with the indexing info
 *  	and use it as a handle for the set/get calls
 *  Messages including indexing information:
 *  	Here the DynamicFinfo is needed purely to manage the MsgSrc
 *  	and MsgDest, as it uses the index info in the message call.
 *  	The message for 'set' is Ftype2< T1, T2 > where T1 is the
 *  	value and T2 is the index.
 *  	The messages for 'get' are an incoming trigger of Ftype1<T2>
 *  	for the index, and an outgoing Ftype1<T1> with the field value.
 *  	Here we do not need to create a DynamicFinfo, and if one
 *  	exists, it just refers to the Finfo's lookup functions.
 *  	As these lookup functions work with indexing, the base
 *  lookupSet and lookupGet which provide their own index:
 *  	This time, we don't need a DynamicFinfo. These lookup functions
 *  	provide the index along with the value.
 *
 */
template < class T1, class T2 > class LookupFtype: public Ftype1< T1 >
{
		public:
			/**
			 * The LookupRecvFunc here uses the DynamicFinfo
			 * to keep track of the original setFunc of the ArrayFinfo,
			 * and to hold the lookup data. Has to be in a void*.
			 */
			static void lookupRecvFunc( const Conn* c, T1 v )
			{
				const DynamicFinfo* f = getDF( c );
				assert ( f != 0 );
				const T2* index = static_cast< const T2* >(
								f->generalIndex() );

				void (*rf)( const Conn* c, T1 v, const T2& index ) =
					reinterpret_cast<
					void (*)( const Conn*, T1, const T2& ) > (
									f->recvFunc() );
				rf( c, v, *index );
			}

			/**
			 * This is the recvFunc for triggering outgoing messages
			 * with the looked-up value.
			 * You probably want to also consider using the alternate
			 * SharedMessage provided by the LookupFtype, where it
			 * takes the index in an incoming trigger message, and
			 * sends out the value in a return message.
			 * This function refers to one of the dynamicFinfos
			 * on the target object to figure out what has to be
			 * done for the return message.
			 * The DynamicFinfo provides four things:
			 * - Lookup from Conn.
			 * - The index of the lookup entry.
			 * - The GetFunc for the array type.
			 * - The message response handling for later adds.
			 */
			static void lookupTrigFunc( const Conn* c )
			{
				const DynamicFinfo* f = getDF( c );
				T1 (*getLookup)( Eref, const T2& index ) =
					reinterpret_cast< 
					T1 (*)( Eref, const T2& ) >
					( f->innerGetFunc() );

				Eref e = c->target();
				const T2* index = static_cast< const T2* >(
								f->generalIndex() );
				///\todo Fix hack involving Slot
				send1< T1 >( e, Slot( f->msg(), 0 ), 
					getLookup( e, *index ) );
			}

			RecvFunc recvFunc() const {
				return reinterpret_cast< RecvFunc >(
						&lookupRecvFunc );
			}

			RecvFunc trigFunc() const {
				return &lookupTrigFunc;
			}

			static const Ftype* global() {
				static Ftype* ret = new LookupFtype< T1, T2 >();
				return ret;
			}

    virtual std::string getTemplateParameters() const
    {
        static std::string s = Ftype::full_type(typeid(T1))+","+Ftype::full_type(typeid(T2));
        return s;
    }
			/**
			 * 'get' gets a value. It requires that the indexing info
			 * be available through the Finfo, which must be a 
			 * DynamicFinfo. Returns true on success.
			 */
			bool get( Eref e, const Finfo* f, T1& v ) const {
				const DynamicFinfo* df =
						dynamic_cast< const DynamicFinfo* >( f );
				assert( df != 0 );
				T1 ( *g )( Eref, const T2& ) =
					reinterpret_cast< T1 (*)( Eref, const T2& ) >(
							df->innerGetFunc()
					);
				const T2* index = static_cast< const T2* >(
								df->generalIndex() );
				v = g( e, *index );
				return 1;
			}

			/**
			 * 'set' assigns a value. It requires that the indexing
			 * info be available through the Finfo, which must
			 * be a DynamicFinfo. Returns true on success.
			 * It specializes the generic version in the parent Ftype1
			 */
			bool set( Eref e, const Finfo* f, T1 v ) const {
				const DynamicFinfo* df =
						dynamic_cast< const DynamicFinfo* >( f );
				assert( df != 0 );

				void (*set)( const Conn*, T1 v, const T2& ) =
					reinterpret_cast<
					void (*)( const Conn*, T1, const T2& ) >(
						df->recvFunc()
					);
				SetConn c( e );
				const T2* index = static_cast< const T2* >(
								df->generalIndex() );
				set( &c, v, *index );
				return 1;
			}

			/**
			 * This gets the value and converts it to a string,
			 * returning true if everything worked.
			 * This may only be called from a DynamicFinfo
			 */
			bool strGet( Eref e, const Finfo* f, string& s ) const
			{
				T1 val;
				if ( get( e, f, val ) )
						return val2str( val, s );
				return 0;
			}

			/**
			 * This sets the value from a string,
			 * returning true if everything worked.
			 * This may only be called from a DynamicFinfo
			 */
			bool strSet( Eref e, const Finfo* f, const string& s )
					const
			{
				T1 val;
				if ( str2val( s, val ) )
						return set( e, f, val );
				return 0;
			}

			/**
			 * This virtual function returns a void* to an allocated
			 * T2 instance of the converted string. If the conversion
			 * fails it returns 0.
			 * Note that in LookupFtype it has a special meaning,
			 * because it converts to the T2 class which is used
			 * for indexing, rather than to the data class.
			 * I'm not sure if the function is even needed for
			 * any other ftype.
			 */
			void* strToIndexPtr( const string& s ) const {
				T2 ret;
				if ( str2val( s, ret ) ) {
					return new T2( ret );
				}
				return 0;
			}

			/**
			 * Here we can directly get the value without any
			 * intermediates like the DynamicFinfo.
			 * This may only be called from lookupGet< T1, T2 >
			 * It is happy with the Finfo either as a DynamicFinfo
			 * or as a LookupFinfo.
			 * \todo Unresolved issue: How to report bad index?
			 */
			bool lookupGet( Eref e, const Finfo* f,
							T1& v, const T2& index ) const {
				const LookupFinfo* lf = 
					dynamic_cast< const LookupFinfo* >( f );
				if ( lf ) {
					T1 ( *g )( Eref, const T2& ) =
						reinterpret_cast<
						T1 (*)( Eref, const T2& ) >( lf->innerGetFunc()
						);
					v = g( e, index );
					return 1;
				}
				const DynamicFinfo* df = 
					dynamic_cast< const DynamicFinfo* >( f );
				if ( df ) {
					T1 ( *g )( Eref, const T2& ) =
						reinterpret_cast<
						T1 (*)( Eref, const T2& ) >(
								df->innerGetFunc()
						);
					v = g( e, index );
					return 1;
				}
				return 0;
			}

			/**
			 * Here we can directly set the value without any
			 * intermediates like the DynamicFinfo.
			 * This may only be called from lookupSet< T1, T2 >
			 * It is happy with the Finfo either as a DynamicFinfo
			 * or as a LookupFinfo
			 * \todo Unresolved issue: How to report bad index?
			 */
			bool lookupSet( Eref e, const Finfo* f,
							T1 v, const T2& index ) const {
				const LookupFinfo* lf = 
					dynamic_cast< const LookupFinfo* >( f );
				void (*set)( const Conn*, T1 v, const T2& ) = 0;
				if ( lf ) {
					set = reinterpret_cast<
						void (*)( const Conn*, T1 v, const T2& ) >(
								lf->recvFunc()
						);
				} else {
					const DynamicFinfo* df = 
						dynamic_cast< const DynamicFinfo* >( f );
					if ( df ) {
						set = reinterpret_cast<
							void (*)( const Conn*, T1 v, const T2& ) >(
									df->recvFunc()
							);
					}
				}
				if ( set != 0 ) {
					SetConn c( e );
					set( &c , v, index );
					return 1;
				}
				return 0;
			}

			/**
			 * Free index data. Used only for LookupFinfo
			 */
			void destroyIndex( void* index ) const
			{
				delete static_cast< T2* >( index );
			}

			/**
			 * Copy index data. Used only for LookupFinfo
			 */
			void* copyIndex( void* index ) const
			{
				assert( index != 0 );
				return new T2( *static_cast< T2* >( index ) );
			}

			const Ftype* baseFtype() const {
				return Ftype1< T1 >::global();
			}
};

#endif // _LOOKUP_FTYPE_H
