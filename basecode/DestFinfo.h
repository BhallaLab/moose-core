/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _DEST_FINFO_H
#define _DEST_FINFO_H
#include <string>
#include "Ftype.h"
#include "Finfo.h"
#include "Element.h"

using namespace std;

/**
 * Finfo for handling message destinations with a single message.
 */
class DestFinfo: public Finfo
{
		public:
#ifdef DO_UNIT_TESTS
			friend void cinfoTest(); // wants to look at destIndex_
#endif
			DestFinfo( const string& name, const Ftype *f, 
							RecvFunc rfunc, unsigned int destIndex = 0 );

			~DestFinfo()
			{;}

			/**
			 * This slightly odd looking operation is meant to 
			 * connect any calls made to this dest onward to other
			 * dests. 
			 * For now we ignore it in the DestFinfo.
			 */
			bool add( 
					Element* e, Element* destElm, const Finfo* destFinfo
			) const 
			{
					return 0;
			}
			
			bool respondToAdd(
					Element* e, Element* dest, const Ftype *destType,
					FuncList& destfl, FuncList& returnFl,
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
			 * Call the RecvFunc with the arguments in the string.
			 */
			bool strSet( Element* e, const std::string &s )
					const;
			
			/// strGet doesn't work for DestFinfo
			bool strGet( const Element* e, std::string &s ) const {
				return 0;
			}

			RecvFunc recvFunc() const {
					return rfunc_;
			}

			void countMessages( 
					unsigned int& srcNum, unsigned int& destNum )
			{
				destIndex_ = destNum++;
			}

			bool getSlotIndex( const string& name, 
					unsigned int& ret ) const;
			/*
			unsigned int getSlotIndex() const {
				return destIndex_;
			}
			*/

			const Finfo* match( 
				const Element* e, unsigned int connIndex ) const;

			bool isTransient() const {
					return 0;
			}

			bool inherit( const Finfo* baseFinfo );

			Finfo* copy() const {
				return new DestFinfo( *this );
			}

		private:
			RecvFunc rfunc_;
			unsigned int destIndex_;
};

#endif // _DEST_FINFO_H
