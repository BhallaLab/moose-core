/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _LOOKUP_FINFO_H
#define _LOOKUP_FINFO_H
#include "RecvFunc.h"
#include "header.h"
/**
 * Finfo for handling data fields that are accessed through get/set
 * functions plus an arbitrary type index lookup.
 * The LookupFinfo works closely with its counterpart, the 
 * LookupFtype< T1, T2 >. This must be passed in as the second arg.
 *
 * The get_ and set_ funcs are similar to those for the ValueFinfo,
 * but have an extra argument for the index, which can be any type.
 * So, we have: T1 ( *get )( const Element*, const T2& index )
 * and void( *set )( const Conn*, T1 v, const T2& index )
 *
 * The ArrayFinfo is a special case of LookupFinfo, with an
 * unsigned int index.
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
 *
 */

class LookupFinfo: public Finfo
{
		public:
			LookupFinfo( const string& name,
						const Ftype* f,
						GetFunc get,
						RecvFunc set, 
						const string& doc="" )
					: Finfo( name, f, doc ), get_( get ), set_( set )
			{;}

			~LookupFinfo()
			{;}

			/**
			 * This operation requires the formation of a dynamic
			 * Finfo to handle the messaging, as Lookup fields are
			 * not assigned a message src or dest.
			 */
			bool add( 
					Eref e, Eref destElm, const Finfo* destFinfo,
					unsigned int connTainerOption
			) const ;
			
			bool respondToAdd(
					Eref e, Eref src, const Ftype *srcType,
					unsigned int& srcFuncId, unsigned int& returnFuncId,
					int& destMsgId, unsigned int& destIndex
			) const;

			/**
			 * Returns a flag for a bad msg.
			 */
			int msg() const {
				return INT_MAX;
			}

			/**
			 * The Ftype knows how to do this conversion.
			 */
			bool strSet( Eref e, const std::string &s ) const
			{ 
					return 0;
			}
			
			// The Ftype handles this conversion.
			bool strGet( Eref e, std::string &s ) const
			{
					return 0;
			}

			/** The LookupFinfo must handle indexing at this
			 * stage. It returns a DynamicFinfo to deal with it.
			 * It is up to the calling function to decide if the
			 * Dynamic Finfo should stay or be cleared out.
			 */
			const Finfo* match( Element* e, const string& name ) const;

			RecvFunc recvFunc() const {
					return set_;
			}

			GetFunc innerGetFunc() const {
					return get_;
			}

			bool isTransient() const {
					return 0;
			}

			/**
			 * Returns true only if the other object is also a 
			 * LookupFinfo. This will be true if LookupFtype matches.
			 */
			bool inherit( const Finfo* baseFinfo ) {
				return ( ftype()->isSameType( baseFinfo->ftype() ) );
			}

			Finfo* copy() const {
				return new LookupFinfo( *this );
			}

			void addFuncVec( const string& cname );

			/**
			 * Returns the identifier for its FuncVec, which handles
			 * its set_ RecvFunc
			 */
			unsigned int funcId() const {
				return fv_->id();
			}

			/// Looks at the ftype.
			unsigned int syncFuncId() const {
				return ftype()->syncFuncId();
			}

			/// Looks at the ftype.
			unsigned int asyncFuncId() const {
				return ftype()->asyncFuncId();
			}

			/// Looks at the ftype.
			unsigned int proxyFuncId() const {
				return ftype()->proxyFuncId();
			}

			/**
			 * The LookupFinfo does not handle any messages itself, so
			 * does not need to allocate any on the parent object.
			 */
			void countMessages( unsigned int& num ) {
				;
			}

		private:
			GetFunc get_;
			RecvFunc set_;
			FuncVec* fv_;
};

#endif // _LOOKUP_FINFO_H
