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
			Finfo( const string& name, const Ftype *f )
					: name_( name ), ftype_( f )
			{;}

			virtual ~Finfo()
			{;}

			virtual bool add( 
					Element* e, Element* destElm, const Finfo* destFinfo
			) const = 0;
			virtual bool respondToAdd(
					Element* e, Element* src, const Ftype *srcType,
					FuncList& srcfl, FuncList& returnFl,
					unsigned int& destIndex, unsigned int& numDest
			) const = 0;
			
			/*
			virtual bool drop( Element* e, unsigned int i ) const = 0;

			virtual bool respondToDrop( Element* e, unsigned int i )
					const = 0;
					*/

			virtual unsigned int srcList( 
					const Element* e, vector< Conn >& list ) const = 0;
			virtual unsigned int destList( 
					const Element* e, vector< Conn >& list ) const = 0;

			virtual bool strSet(
							Element* e, const std::string &s
				) const = 0;
			virtual bool strGet( 
							const Element* e, std::string &s
				) const = 0;

			/**
			 * In Finfos that handle data values, this returns the
			 * GetFunc defined in RecvFunc.h:
			 * typedef double( *GetFunc )( const Element* )
			 * It has to be typecast into the correct type, of course,
			 * and that * needs to be done under the protection of a
			 * tyepcheck.
			 * If the Finfo does not support data values this
			 * returns 0. This is a common case so I set it up
			 * as a default.
			 * \todo
			 * I have eliminated this function. I don't  think it
			 * should be exposed here.
			virtual GetFunc getFunc() const {
					return 0;
			}
			 */

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
			 * Returns the matching finfo if the specified connIndex
			 * is managed by the messages handled by this finfo.
			 * The primary use for this is to track down DynamicFinfos.
			 * The secondary
			 * use is to locate statically defined Finfos that
			 * manage the specified conn.
			 */
			virtual const Finfo* match( 
				const Element* e, unsigned int connIndex ) const = 0;

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
			 * This function is used to build up both the internal
			 * Finfo numbering of current srcIndex and destIndex, and
			 * also for the calling function to keep track of the
			 * tally. The Finfo has to increment the indices depending
			 * on how many src and dest slots it uses.
			 * The ConnIndex is used in a few special cases like
			 * DynamicFinfos, but it is not returned.
			 * Note this is NOT a const
			 * function, as it may alter the Finfo.
			 * Normally called during Cinfo initialization, and
			 * also whenever a new Finfo is created dynamically.
			 */
			virtual void countMessages(
				unsigned int& srcIndex, unsigned int& destIndex ) = 0;

			/**
			 * This returns true if the Finfo is meant to be 
			 * created and destroyed during the lifetime of an
			 * Element. Otherwise all Finfos are permanent, 
			 * created at static initialization and never
			 * destroyed.
			 * Used by the SimpleElement destructor function.
			 */
			virtual bool isTransient() const = 0;
		private:
			string name_;
			const Ftype* ftype_;
};

#endif // _FINFO_H
