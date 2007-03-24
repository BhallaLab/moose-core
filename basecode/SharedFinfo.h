/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _SHARED_FINFO_H
#define _SHARED_FINFO_H

/**
 * Finfo for handling shared messages with arbitrary numbers of
 * srcs and dests.
 * The array of pair< Ftype*, RecvFunc > can indicate MsgSrc or MsgDest
 * in any order, and that order will be preserved in the created
 * Finfo. In this array, MsgSrcs are indicated by the presence of
 * a dummyFunc, and Dests by other funcs.
 * This order is enforced when messages are set up. This means that
 * you cannot send a shared message between the same Finfo on
 * identical objects. Other than that case, the convention is to
 * use the same name for the source and destination Finfo of a shared
 * message, because they are usually a bit of both. Furthermore,
 * it is fine to send the message in either direction.
 */
class SharedFinfo: public Finfo
{
		public:

			/**
			 * In the constructor, we need to build up a composite
			 * Ftype that manages the local vector of Ftypes. We
			 * also make a vector of RecvFuncs that will be used
			 * when adding messages.
			 */
			SharedFinfo( const string& name,
				pair< const Ftype*, RecvFunc >* types, 
				unsigned int nTypes );

			~SharedFinfo()
			{;}

			bool add(
					Element* e, Element* destElm, const Finfo* destFinfo
			) const;

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
			 * Send a message with the arguments in the string.
			 */
			bool strSet( Element* e, const std::string &s )
					const;
			
			/// strGet doesn't work for SharedFinfo
			bool strGet( const Element* e, std::string &s ) const {
				return 0;
			}

			/// This Finfo does not support recvFuncs.
			RecvFunc recvFunc() const {
					return 0;
			}
			
			/**
			 * In the case of the SharedFinfo, we need to assign
			 * MsgSrcs for every src entry in the types list. 
			 * These are identified by the entries where the RecvFunc
			 * is zero. These src entries are mostly redundant,
			 * because they all specify the same conn range. The
			 * only distinguishing feature is that each manages a
			 * different RecvFunc from the target.
			 * When the SharedFinfo has no MsgSrcs, then we set up
			 * a single MsgDest to deal with it all.
			 */
			void countMessages( 
					unsigned int& srcNum, unsigned int& destNum );

			const Finfo* match( 
				const Element* e, unsigned int connIndex ) const;

			bool isTransient() const {
					return 0;
			}

			/**
			 * Returns true only if the other finfo is the same type
			 * For the SharedFinfo this means that every function
			 * argument matches. Then on top of this, the function
			 * copies over the slot indices.
			 */
			bool inherit( const Finfo* baseFinfo );

			unsigned int getSlotIndex() const {
					return msgIndex_;
			}

		private:
			unsigned int numSrc_;
			FuncList rfuncs_;
			unsigned int msgIndex_;
};

#endif // _SHARED_FINFO_H
