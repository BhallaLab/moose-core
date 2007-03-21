/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _ARRAY_FINFO_H
#define _ARRAY_FINFO_H

/**
 * Finfo for handling data fields that are accessed through get/set
 * functions plus an index lookup. Such fields are atomic, that is,
 * they cannot be the
 * base for indirection. This is because the set operation works on
 * the object as a whole.
 * The get_ and set_ funcs are similar to those for the ValueFinfo,
 * but have an extra argument for the index.
 *
 */
class ArrayFinfo: public Finfo
{
		public:
			ArrayFinfo( const string& name,
						const Ftype* f,
						GetFunc get,
						RecvFunc set )
					: Finfo( name, f ), get_( get ), set_( set )
			{;}

			~ArrayFinfo()
			{;}

			/**
			 * This operation requires the formation of a dynamic
			 * Finfo to handle the messaging, as Array fields are
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

			/**
			 * Both of these Drop functions are dummy functions.
			 * The Dynamic Finfo has to handle these operations.
			 */
			void dropAll( Element* e ) const;
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

			unsigned int srcList(
					const Element* e, vector< Conn >& list ) const {
					return 0;
			}
			unsigned int destList(
					const Element* e, vector< Conn >& list ) const {
					return 0;
			}

			/**
			 * We don't need to do anything here because ArrayFinfo
			 * does not deal with messages directly. If we need to
			 * send messages to a ArrayFinfo, then a DynamicFinfo
			 * must be created
			 */
			void countMessages( 
					unsigned int& srcNum, unsigned int& destNum )
			{ ; }

			/** The ArrayFinfo must handle indexing at this
			 * stage. It returns a DynamicFinfo to deal with it.
			 * It is up to the calling function to decide if the
			 * Dynamic Finfo should stay or be cleared out.
			 */
			const Finfo* match( Element* e, const string& name ) const;
			
			/**
			 * The ArrayFinfo never has messages going to or from it:
			 * they all go via DynamicFinfo if needed. So it cannot
			 * match any connIndex.
			 */
			const Finfo* match( 
				const Element* e, unsigned int connIndex ) const {
				return 0;
			}

			/*
			RecvFunc getFunc() const {
					return get_;
			}

			RecvFunc setFunc() const {
					return ftype()->recvFunc();
			}

			RecvFunc trigFunc() const {
					return ftype()->trigFunc();
			}
			*/

			RecvFunc recvFunc() const {
					return set_;
			}

			bool isTransient() const {
					return 0;
			}

			bool inherit( const Finfo* baseFinfo ) {
				return ftype()->isSameType( baseFinfo->ftype() );
			}

		private:
			GetFunc get_;
			RecvFunc set_;
};

#endif // _ARRAY_FINFO_H
