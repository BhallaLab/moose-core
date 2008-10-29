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
// typedef pair< void* (*)( void*, unsigned int ), unsigned int > IndirectType;

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
							GetFunc getFunc, void* index = 0, const string& doc="" )
					: Finfo( name, origFinfo->ftype(), doc ),
					origFinfo_( origFinfo ), 
					getFunc_( getFunc ),
					generalIndex_( index ),
					msg_( 0 )
			{;}

			// Assert that the affected conns have been cleaned up
			// before deleteing this.
			~DynamicFinfo();

			/**
			 * This sets up a DynamicFinfo on a given Element. If
			 * there is one sitting around unused it uses that, 
			 * otherwise it makes a new one.
			 */
			static DynamicFinfo* setupDynamicFinfo(
				Eref e, const string& name, const Finfo* origFinfo,
				GetFunc getFunc,
				void* index = 0 );

			/**
			 * This should be almost the regular SrcFinfo::add
			 * operation, since at this point we have acquired
			 * a MsgSrc slot.
			 */
			bool add( 
					Eref e, Eref destElm, const Finfo* destFinfo,
					unsigned int connTainerOption
			) const;
			
			/**
			 * Again, this should be similar to the regular
			 * DestFinfo::respondToAdd operation, using the
			 * MsgDest slot.
			 */
			bool respondToAdd(
					Eref e, Eref src, const Ftype *srcType,
					unsigned int& srcFuncId, unsigned int& returnFuncId,
					int& destMsgId, unsigned int& destIndex
			) const;


			/**
			 * This msg is always placed on the msg_ src vector, amid
			 * the nexts.
			 */
			int msg() const {
				return msg_;
			}

			/**
			 * This function erases contents of a DynamicFinfo, including
			 * getting rid of messages, its name, and functions. This
			 * is needed because we may otherwise get spurious matches to
			 * an old DynamicFinfo, in the Element::findFinfo( name )
			 */
			void clear( Element* e );

			/**
			 * The Ftype knows how to do this conversion.
			 */
			bool strSet( Eref e, const std::string &s ) const;
			
			// The Ftype handles this conversion.
			bool strGet( Eref e, std::string &s ) const;
			
			/// Public RecvFunc for receiving function args.
			RecvFunc recvFunc() const {
				return origFinfo_->recvFunc();
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
			const Finfo* match( Element* e, const string& n ) const;

			const Finfo* match( 
				const Element* e, const ConnTainer* c ) const;

			void countMessages( unsigned int& num );

			/**
			 * The dynamicFinfo is one of the few Finfos that has
			 * a true response to this function.
			 */
			bool isTransient() const {
					return 1;
			}

			bool getSlot( const string& name, Slot& ret ) const;

			/**
			 * This operation makes no sense for the DynamicFinfo
			 */
			bool inherit( const Finfo* baseFinfo ) {
				return 0;
			}

			/**
			 * The copy operation for DynamicFinfos is the most
			 * impartant among Finfos.
			 * Most of the fields are OK, but the generalIndex
			 * is not and needs a private copy, because it gets
			 * deleted when the DynamicFinfo is deleted.
			 */
			Finfo* copy() const;

			/**
			 * For the DynamicFinfo, we should pass in an existing
			 * FuncVec for set, get ,and recv. This will need changes
			 * in constructor and associated code. We do not expect to
			 * create a new FuncVec here.
			 */
			void addFuncVec( const string& cname )
			{;}

			/////////////////////////////////////////////////////////
			// Here we define the functions that are unique to this
			// class.
			/////////////////////////////////////////////////////////
			
			const Finfo* origFinfo() const {
					return origFinfo_;
			}

			const void* generalIndex( ) const {
					return generalIndex_;
			}

			/**
			 * Returns the func id of this DynamicFinfo by looking up
			 * the original.
			 */
			unsigned int funcId() const {
				return origFinfo_->funcId();
			}

			/// Looks at the original.
			unsigned int syncFuncId() const {
				return origFinfo_->syncFuncId();
			}

			/// Looks at the original.
			unsigned int asyncFuncId() const {
				return origFinfo_->asyncFuncId();
			}

			/// Looks at the original.
			unsigned int proxyFuncId() const {
				return origFinfo_->proxyFuncId();
			}

			/**
			 * This should be true if the Finfo never acts as a 
			 * message Source, but it is tricky here in the case of
			 * the DynamicFinfo. This is because the role of the 
			 * DynamicFinfo may be a pure dest (when the field is
			 * assigned by a message) or as a mixed one (when the
			 * field is requested to send its value back).
			 * Probably doesn't matter, so I'll choose the conservative
			 * option.
			 */
			bool isDestOnly() const {
				return 0;
			}


		private:
			const Finfo* origFinfo_;

			/// Inner GetFunc from Finfo, gets value if applicable.
			GetFunc getFunc_;

			/// This is used by LookupFinfo.
			void* generalIndex_;

			unsigned int msg_;
};

/**
 * This function looks up the DynamicFinfo matching the incoming Conn
 */
extern const DynamicFinfo* getDF( const Conn* );

#endif // _DYNAMIC_FINFO_H
