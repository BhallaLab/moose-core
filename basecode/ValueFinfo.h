/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _VALUE_FINFO_H
#define _VALUE_FINFO_H

/**
 * Finfo for handling data fields that are accessed through get/set
 * functions. Such fields are atomic, that is, they cannot be the
 * base for indirection. This is because the set operation works on
 * the object as a whole. The special use of such fields is if we 
 * want to present an evaluated field: a field whose value is not
 * stored explicitly, but computed (or assigned) on the fly. An example
 * would be the length of a dendritic compartment (a function of its
 * coordinates and those of its parent) or the concentration of a
 * molecule (from the # of molecules and its volume).
 */
class ValueFinfo: public Finfo
{
		public:
			ValueFinfo(
				const string& name,
				const Ftype* f,
				GetFunc get, RecvFunc set
			)
				: Finfo( name, f ), get_( get ), set_( set )
			{;}

			~ValueFinfo()
			{;}

			/**
			 * This operation requires the formation of a dynamic
			 * Finfo to handle the messaging, as Value fields are
			 * not assigned a message src or dest.
			 * \todo: Still to implement.
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
			 * The Ftype knows how to do this conversion.
			 */
			bool strSet( Element* e, const std::string &s ) const
			{ 
					return 0;
			}
			
			// The Ftype handles this conversion.
			bool strGet( const Element* e, std::string &s ) const
			{
					return ftype()->strGet( e, this, s );
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
			 * We don't need to do anything here because ValueFinfo
			 * does not deal with messages directly. If we need to
			 * send messages to a ValueFinfo, then a DynamicFinfo
			 * must be created
			 */
			void countMessages( 
					unsigned int& srcNum, unsigned int& destNum )
			{ ; }

			/**
			 * The ValueFinfo never has messages going to or from it:
			 * they all go via DynamicFinfo if needed. So it cannot
			 * match any connIndex.
			 */
			const Finfo* match( 
				const Element* e, unsigned int connIndex ) const {
				return 0;
			}

			bool isTransient() const {
					return 0;
			}

			RecvFunc recvFunc() const {
					return set_;
			}

			GetFunc innerGetFunc() const {
					return get_;
			}

		private:
			GetFunc get_;
			RecvFunc set_;
};

#endif // _VALUE_FINFO_H
