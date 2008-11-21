/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _FINFO_H
#define _FINFO_H

/**
 * Virtual base class for field info.
 */
class Finfo
{
		public:
			Finfo( const string& name, const Ftype *f, const string& doc = "" )
					: name_( name ), ftype_( f ), doc_( doc )
			{;}

			virtual ~Finfo()
			{;}

			/**
			 * This function creates a connection between two Finfos.
			 */
			virtual bool add( 
					Eref e, Eref destElm, const Finfo* destFinfo,
					unsigned int connTainerOption
			) const = 0;

			/**
			 * This function is executed at the destination Finfo and
			 * does all the critical type-checking to decide if the
			 * message creation is legal. It also handles passing the
			 * message function pointers back and forth between source
			 * and destination objects.
			 * Arguments:
			 * 	e: target Element ref
			 * 	src: src Element ref
			 * 	srcType: src type
			 *	srcFuncId: src func
			 *	destFuncId: dest func id, filled here and passed back.
			 *	destMsgId: Msg id of dest, filled here and passed back.
			 *	destIndex: Index of dest connection. Used in special cases
			 *		like synapses where the destination Conn needs to be
			 *		identified, but passed back every time. Typically is
			 *		the count of the # of incoming messages on this dest.
			 *		Note that if the connection is a vector we
			 *		increment this by the size of the vector.
			 */
			virtual bool respondToAdd(
					Eref e, Eref src, const Ftype *srcType,
					unsigned int& srcFuncId, unsigned int& destFuncId,
					int& destMsgId, unsigned int& destIndex
			) const = 0;
			
			/**
			 * This function utilizes the hard-coded string conversions
			 * for the type of this Finfo, if present, for doing Set.
			 * Multiple argument fields also can be handled this way.
			 * Returns true if the
			 * conversion and assignment succeeded.
			 */
			virtual bool strSet( Eref e, const std::string &s) const = 0;

			/**
			 * This function utilizes the hard-coded string conversions
			 * if present, for getting the field value. Returns true
			 * if the conversion and assignment succeeded.
			 */
			virtual bool strGet( Eref e, std::string &s) const = 0;

			/**
			 * This returns the message identifier for this Finfo.
			 * Zero and positive values are srcs, and are used by Slots.
			 * Negative values are dest only.
			 */
			virtual int msg() const = 0;

			/**
			 * In any Finfo that has a RecvFunc, this returns it.
			 * In ValueFinfo classes, this is used for assigning
			 * data values. But it could also be used for calling
			 * operations in any function. It is not limited to
			 * single-argument functions either. If the Finfo
			 * does not support RecvFuncs (or if it is a readonly
			 * ValueFinfo) it returns 0. Needs to be typecast into
			 * the correct function type under type protection.
			 */
			virtual RecvFunc recvFunc() const = 0;

			/**
			 * Returns the matching finfo if it is interested.
			 * Reports if there is a match between the field name and
			 * the finfo. If there is a nested finfo, this is where it
			 * would intercept indirection in the name and forward it
			 * on to subfields.
			 * We need the Element argument to build a forwarding
			 * Finfo in case indirection happens.
			 */
			virtual const Finfo* match( Element* e, const string& name )
			const {
					if ( name == name_ )
							return this;
					return 0;
			}

			/** 
			 * Returns the matching finfo if the specified ConnTainer
			 * is managed by the messages handled by this finfo.
			 * The primary use for this is to track down DynamicFinfos.
			 * The secondary
			 * use is to locate statically defined Finfos that
			 * manage the specified conn.
			 *
			 * Many Finfos don't handle messages, so the default is to
			 * return 0 here.
			 */
			virtual const Finfo* match( 
				const Element* e, const ConnTainer* c) const {
				return 0;
			}

			/**
			 * Helps to build up a list of Finfos. Most Finfos will
			 * just push themselves onto the flist, but some 
			 * (like the ThisFinfo and PtrFinfos) are wrappers for
			 * a whole lot more, and will want to push several
			 * Finfos on.
			 */
			virtual void listFinfos( vector< const Finfo* >& flist )
					const {
					flist.push_back( this );
			}

			/**
			 * returns the type of the Finfo
			 */
			const Ftype* ftype() const {
					return ftype_;
			}

			/**
			 * Returns the name of the Finfo
			 */
			const std::string& name() const {
					return name_;
			}

                        /**
                         *  return the documentation string attached to this finfo
                         */
                         const std::string& doc() const
                         {
                             return doc_;
                         }
    
			/**
			 * Returns true if the name matches and if the
			 * Finfo has a suitable srcIndex or destIndex
			 * to pass back in the 'ret' argument.
			 * Many Finfos do not have such a number, 
			 * so it returns 0 by default to indicate failure.
			 */
			virtual bool getSlot( const string& name, Slot& ret ) const {
				return 0;
			}

			/**
			 * This returns true if the Finfo is meant to be 
			 * created and destroyed during the lifetime of an
			 * Element. Otherwise all Finfos are permanent, 
			 * created at static initialization and never
			 * destroyed.
			 * Used by the SimpleElement destructor function.
			 */
			virtual bool isTransient() const = 0;

			/** This returns true if the Finfo is correctly derived
			 * from the base. It primarily checks that the Ftype 
			 * matches, but also does other inheritance operations
			 * such as transferring over message slot indices.
			 */
			virtual bool inherit( const Finfo* baseFinfo ) = 0;

			/**
			 * Makes a duplicate of the current Finfo. 
			 * Needed whenever a Finfo is inherited, as its index may
			 * differ in the duplicate.
			 * Is useful
			 * also for the DynamicFinfos, the ExtFieldFinfos and
			 * other ones that have an element-specific existence
			 * in the nether regions of the finfo_ array.
			 */
			virtual Finfo* copy() const = 0;

			/**
			 * This function sets up a FuncVec for the Finfo
			 * classes that might be targets for messages. The
			 * FuncVec organizes all the RecvFuncs that handle incoming
			 * messages. The argument is the name of the parent Cinfo,
			 * essential to avoid naming conflicts among the FuncVecs.
			 * This function alters the internal state of the Finfo, so
			 * it should only be called at set up time.
			 */
			virtual void addFuncVec( const string& cname ) = 0;

			/**
			 * This returns the id of the FuncVec belonging to this
			 * Finfo. Most of them don't handle FuncVecs, so return
			 * the empty id which is zero.
			 */
			virtual unsigned int funcId() const {
				return 0;
			}

			/**
			 * Returns true if the Finfo never acts as a Source. This
			 * is more precise than the isDest() from the FuncVec, because
			 * that does not know about what may happen on the 
			 * SharedMessages which have both Src and Dest.
			 * Usually true.
			 */
			virtual bool isDestOnly() const {
				return 1;
			}

			/**
			 * This function assignes the msg index to use for this
			 * Finfo, if needed.
			 */
			virtual void countMessages( unsigned int& num ) = 0;

			/**
			 * Returns the funcId for sending synchronous messages from 
			 * this Finfo via the postmaster. If this Finfo does not
			 * send out any messages (e.g., it is a DestFinfo or a 
			 * SharedFinfo with only DestFinfos in it) 
			 * this should return 0.
			 * We cannot do this on the Ftype because a Dest and a Src
			 * Finfo might have identical Ftypes.
			 */
			virtual unsigned int syncFuncId() const = 0;

			/**
			 * Returns the funcId for sending asynchronous messages from 
			 * this Finfo via the postmaster. Again, if this Finfo does not
			 * send out any messages (e.g., it is a DestFinfo or a 
			 * SharedFinfo with only DestFinfos in it) 
			 * this should return 0.
			 */
			virtual unsigned int asyncFuncId() const = 0;

			/**
			 * Returns the funcId for ProxyFunc.
			 * If this Finfo has one or more Dests, then this FuncId
			 * is non-zero.
			 * It takes an incoming data stream from the PostMaster,
			 * parses it into typed arguments, and calls the correct 
			 * Send function to pass data into the RecvFunc of this
			 * Finfo.
			 */
			virtual unsigned int proxyFuncId() const = 0;

		protected:
			/**
			 * setName is used by the DynamicFinfo when it renames
			 * and reuses an existing one.
			 */
			void setName( const string& name ) {
				name_ = name;
			}
		private:
			string name_;
			const Ftype* ftype_;
			const string doc_;
};

#endif // _FINFO_H
