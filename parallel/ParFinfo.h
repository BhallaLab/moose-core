/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _PAR_FINFO_H
#define _PAR_FINFO_H

/**
 * Finfo for handling data to be serialized, usually into an MPI
 * message.
 */
class ParFinfo: public Finfo
{
		public:

			/**
			 * In the constructor, we need to build up a composite
			 * Ftype that manages the local vector of Ftypes. We
			 * also make a vector of RecvFuncs that will be used
			 * when adding messages.
			 */
			ParFinfo( const string& name, const string& doc="" );

			~ParFinfo()
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
			 * Doesn't work for this Finfo.
			 */
			bool strSet( Element* e, const std::string &s ) const {
				return 0;
			}
			
			/// strGet doesn't work for ParFinfo
			bool strGet( const Element* e, std::string &s ) const {
				return 0;
			}

			/// This Finfo does not support recvFuncs.
			RecvFunc recvFunc() const;
			
			/**
			 * In the case of the ParFinfo, we need to assign
			 * MsgSrcs for every src entry in the types list. 
			 * These are identified by the entries where the RecvFunc
			 * is zero. These src entries are mostly redundant,
			 * because they all specify the same conn range. The
			 * only distinguishing feature is that each manages a
			 * different RecvFunc from the target.
			 * When the ParFinfo has no MsgSrcs, then we set up
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
			 * For the ParFinfo this means that every function
			 * argument matches. Then on top of this, the function
			 * copies over the slot indices.
			 */
			bool inherit( const Finfo* baseFinfo ) {
					return 0;
			}

			/*
			unsigned int getSlotIndex() const {
					return msgIndex_;
			}
			*/
			bool getSlotIndex( const string& name, unsigned int& ret ) 
				const;

			Finfo* copy() const {
				return new ParFinfo( *this );
			}

		private:
			unsigned int msgIndex_;
};

#endif // _PAR_FINFO_H
