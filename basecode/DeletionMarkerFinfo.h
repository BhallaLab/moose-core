/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _DELETION_MARKER_FINFO_H
#define _DELETION_MARKER_FINFO_H

/**
 * Finfo for reporting that host element is about to be deleted
 * Does not participate in anything else.
 */
class DeletionMarkerFinfo: public Finfo
{
		public:
			DeletionMarkerFinfo();

			~DeletionMarkerFinfo()
			{;}

			bool add( 
					Element* e, Element* destElm, const Finfo* destFinfo
			) const 
			{
					return 0;
			}
			
			bool respondToAdd(
					Element* e, Element* dest, const Ftype *destType,
					FuncList& destfl, FuncList& returnFl,
					unsigned int& destIndex, unsigned int& numDeletionMarker
			) const 
			{
					return 0;
			}

			void dropAll( Element* e ) const { ; }
			bool drop( Element* e, unsigned int i ) const { return 0; }

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
			 * Call the RecvFunc with the arguments in the string.
			 */
			bool strSet( Element* e, const std::string &s )
					const
					{
							return 0;
					}
			
			/// strGet doesn't work for DeletionMarkerFinfo
			bool strGet( const Element* e, std::string &s ) const {
				return 0;
			}

			RecvFunc recvFunc() const {
					return 0;
			}

			void countMessages( 
					unsigned int& srcNum, unsigned int& destNum )
			{
				;
			}

			const Finfo* match( 
				const Element* e, unsigned int connIndex ) const
			{
					return 0;
			}

			bool isTransient() const {
					return 0;
			}

			bool inherit( const Finfo* baseFinfo ) {
				return ftype()->isSameType( baseFinfo->ftype() );
			}

			Finfo* copy() const {
				return new DeletionMarkerFinfo( *this );
			}

			///////////////////////////////////////////////////////
			// Class-specific functions below
			///////////////////////////////////////////////////////

			/**
			 * This function allows us to place a single statically
			 * created DeletionMarkerFinfo wherever it is needed.
			 */
			static DeletionMarkerFinfo* global();

		private:
};

#endif // _DELETION_MARKER_FINFO_H
