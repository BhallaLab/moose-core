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

/**
 * Finfo for handling data fields that are accessed through get/set
 * functions plus an arbitrary type index lookup.
 * The LookupFinfo works closely with its counterpart, the 
 * LookupFtype< T1, T2 >. This must be passed in as the second arg.
 *
 * The get_ and set_ funcs are similar to those for the ValueFinfo,
 * but have an extra argument for the index, which can be any type.
 * So, we have: T1 ( *get )( const Element*, const T2& index )
 * and void( *set )( const Conn&, T1 v, const T2& index )
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
						RecvFunc set )
					: Finfo( name, f ), get_( get ), set_( set )
			{;}

			~LookupFinfo()
			{;}

			/**
			 * This operation requires the formation of a dynamic
			 * Finfo to handle the messaging, as Lookup fields are
			 * not assigned a message src or dest.
			 */
			bool add( 
					Element* e, Element* destElm, const Finfo* destFinfo
			) const ;
			
			bool respondToAdd(
					Element* e, Element* src, const Ftype *srcType,
					FuncList& srcFl, FuncList& returnFl,
					unsigned int& destIndex, unsigned int& numDest
			) const;

			/// This is a dummy function. The Dynamic Finfo handles it.
			void dropAll( Element* e ) const;

			/// This is a dummy function. The Dynamic Finfo handles it.
			bool drop( Element* e, unsigned int i ) const;

			/**
			 * The Ftype knows how to do this conversion.
			 */
			bool strSet( Element* e, const std::string &s ) const
			{ 
					return 0;
			}
			
			// The Ftype handles this conversion.
			bool strGet( const Element* e, std::string &s ) const
			{
					return 0;
			}

			// The following 4 functions are dummies because the
			// DynamicFinfo deals with them.
			unsigned int numIncoming( const Element* e ) const {
					return 0;
			}

			unsigned int numOutgoing( const Element* e ) const {
					return 0;
			}

			unsigned int incomingConns(
					const Element* e, vector< Conn >& list ) const {
					return 0;
			}
			unsigned int outgoingConns(
					const Element* e, vector< Conn >& list ) const {
					return 0;
			}

			/**
			 * We don't need to do anything here because LookupFinfo
			 * does not deal with messages directly. If we need to
			 * send messages to a LookupFinfo, then a DynamicFinfo
			 * must be created
			 */
			void countMessages( 
					unsigned int& srcNum, unsigned int& destNum )
			{ ; }

			/** The LookupFinfo must handle indexing at this
			 * stage. It returns a DynamicFinfo to deal with it.
			 * It is up to the calling function to decide if the
			 * Dynamic Finfo should stay or be cleared out.
			 */
			const Finfo* match( Element* e, const string& name ) const;
			
			/**
			 * The LookupFinfo never has messages going to or from it:
			 * they all go via DynamicFinfo if needed. So it cannot
			 * match any connIndex.
			 */
			const Finfo* match( 
				const Element* e, unsigned int connIndex ) const {
				return 0;
			}

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

		private:
			GetFunc get_;
			RecvFunc set_;
};

#endif // _LOOKUP_FINFO_H
