/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _DYNAMIC_FINFO_H
#define _DYNAMIC_FINFO_H

/**
 * IndirectType is used to manage nested indirection and array lookups.
 */
typedef pair< void* (*)( void*, unsigned int ), unsigned int > IndirectType;

/**
 * Finfo that is built up dynamically on the Element, used for
 * situations where compiled-in slots for the messages have not
 * been assigned:
 * - Messages to and from ValueFinfos
 * - Messages involving indirection.
 * This is created in the respondToAdd of various other Finfos. In
 * those funcs the new src and dest slots are assigned and passed
 * in at creation time.
 * The trigFunc argument is a specialized func used in ValueFinfos,
 * the trigFunc. It is used when an incoming message of Ftype0 is
 * received. Normally the DynamicFinfo refers to the origFinfo
 * for its set and get funcs, provided the Ftype matches.
 * To avoid confusion, non ValueFinfo objects should pass in a
 * dummyFunc.
 */
class DynamicFinfo: public Finfo
{
		public:
			DynamicFinfo( const string& name, const Finfo* origFinfo,
							RecvFunc setFunc, GetFunc getFunc,
							RecvFunc recvFunc, RecvFunc trigFunc,
							unsigned int index = 0 )
					: Finfo( name, origFinfo->ftype() ),
					origFinfo_( origFinfo ), 
					setFunc_( setFunc ),
					getFunc_( getFunc ),
					recvFunc_( recvFunc ) ,
					trigFunc_( trigFunc ) ,
					arrayIndex_( index ),
					srcIndex_( 0 ),
					destIndex_( 0 )
			{;}

			DynamicFinfo( const string& name, const Finfo* origFinfo,
							vector< IndirectType >& indirection )
					: Finfo( name, origFinfo->ftype() ),
					origFinfo_( origFinfo ), 
					setFunc_( &dummyFunc ),
					getFunc_( 0 ),
					recvFunc_( &dummyFunc ) ,
					trigFunc_( &dummyFunc ) ,
					arrayIndex_( 0 ),
					srcIndex_( 0 ),
					destIndex_( 0 ),
					indirect_( indirection )
			{;}

			// Assert that the affected conns have been cleaned up
			// before deleteing this.
			~DynamicFinfo()
			{;}

			/**
			 * This should be almost the regular SrcFinfo::add
			 * operation, since at this point we have acquired
			 * a MsgSrc slot.
			 */
			bool add( 
					Element* e, Element* destElm, const Finfo* destFinfo
			) const;
			
			/**
			 * Again, this should be similar to the regular
			 * DestFinfo::respondToAdd operation, using the
			 * MsgDest slot.
			 */
			bool respondToAdd(
					Element* e, Element* src, const Ftype *srcType,
					FuncList& srcfl, FuncList& returnFl,
					unsigned int& destIndex, unsigned int& numDest
			) const;

			void dropAll( Element* e ) const;
			bool drop( Element* e, unsigned int i ) const;
			
			unsigned int numIncoming( const Element* e ) const;
			unsigned int numOutgoing( const Element* e ) const;
			unsigned int incomingConns(
					const Element* e, vector< Conn >& list ) const;
			unsigned int outgoingConns(
					const Element* e, vector< Conn >& list ) const;


			/**
			 * The Ftype knows how to do this conversion.
			 */
			bool strSet( Element* e, const std::string &s ) const;
			
			// The Ftype handles this conversion.
			bool strGet( const Element* e, std::string &s ) const;
			
			/// Public RecvFunc for receiving function args.
			RecvFunc recvFunc() const {
				return recvFunc_;
			}
			
			/*
			/// Public RecvFunc for receiving requests to send value.
			RecvFunc trigFunc() const {
				return trigFunc_;
			}
			*/
			
			/**
			 * Internal RecvFunc for passing to Ftype to construct
			 * recvFunc. May be the same as recvFunc for some classes,
			 * such as ValueFinfo or DestFinfo.
			 */
			RecvFunc innerSetFunc() const {
				return setFunc_;
			}
			
			/**
			 * Internal GetFunc for looking up values on object.
			 * Used to get the value used for trigFunc. Passed into
			 * Ftype to build the trigFunc. Often zero
			 * for objects that do not support gets.
			 */
			GetFunc innerGetFunc() const {
				return getFunc_;
			}

			/**
			 * Returns the matching finfo if it is interested.
			 * Reports if there is a match between the field name and
			 * the finfo. This is just like the base Finfo class
			 * operation, but we reimplement the function here
			 * for clarity.
			 * If this is handling a ValueFinfo, it just checks for the
			 * name.
			 * If an ArrayFinfo, it looks up an index as well
			 * If a NestFinfo, it checks for the entire traversal.
			 * All these are done trivially because at creation time
			 * the name_ field is set up so that it has the entire
			 * matching string.
			 * The only issue with this is that it does not permit
			 * variations in the string, such as spaces in index
			 * braces.
			 */
			const Finfo* match( Element* e, const string& n )
			const {
					if ( n == name() )
							return this;
					return 0;
			}

			const Finfo* match( 
				const Element* e, unsigned int connIndex ) const;

			void countMessages( 
				unsigned int& srcIndex, unsigned int& destIndex );

			/**
			 * The dynamicFinfo is one of the few Finfos that has
			 * a true response to this function.
			 */
			bool isTransient() const {
					return 1;
			}

			unsigned int getSlotIndex() const;

			/**
			 * This operation makes no sense for the DynamicFinfo
			 */
			bool inherit( const Finfo* baseFinfo ) {
				return 0;
			}

			/////////////////////////////////////////////////////////
			// Here we define the functions that are unique to this
			// class.
			/////////////////////////////////////////////////////////
			
			const Finfo* origFinfo() const {
					return origFinfo_;
			}
			
			unsigned int destIndex() const {
				return destIndex_;
			}
			
			unsigned int srcIndex() const {
				return srcIndex_;
			}

			unsigned int arrayIndex() const {
				return arrayIndex_;
			}

			/**
			 * This returns a pointer to a field specified by the
			 * indirect_ array, as looked up from the data pointer.
			 * Because we deal with void* pointers, this function
			 * must only be called within properly type-protected
			 * functions.
			 */
			void* traverseIndirection( void* data ) const;

			void setGeneralIndex( const void* index ) {
					generalIndex_ = index;
			}

			const void* generalIndex( ) const {
					return generalIndex_;
			}


		private:
			const Finfo* origFinfo_;

			/// Inner RecvFunc from Finfo for handling assignment
			RecvFunc setFunc_;

			/// Inner GetFunc from Finfo, gets value if applicable.
			GetFunc getFunc_;

			/// Public RecvFunc for receiving function args.
			RecvFunc recvFunc_;

			/// Public RecvFunc for receiving requests to send value.
			RecvFunc trigFunc_;

			/// This is used by ArrayFinfo. 
			unsigned int arrayIndex_;

			/// This is used by LookupFinfo.
			const void* generalIndex_;

			unsigned int srcIndex_;
			unsigned int destIndex_;
			vector< IndirectType > indirect_;
};

/**
 * This function looks up the DynamicFinfo matching the incoming Conn
 */
extern const DynamicFinfo* getDF( const Conn& );

#endif // _DYNAMIC_FINFO_H
