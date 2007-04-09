/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _SRC_FINFO_H
#define _SRC_FINFO_H

/**
 * Finfo for handling message sources
 */
class SrcFinfo: public Finfo
{
		public:
#ifdef DO_UNIT_TESTS
			friend void cinfoTest(); // wants to look at srcIndex_
#endif
			SrcFinfo( const string& name, const Ftype *f, 
							unsigned int srcIndex = 0 )
					: Finfo( name, f ), srcIndex_( srcIndex )
			{;}

			~SrcFinfo()
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
			
			/// strGet doesn't work for SrcFinfo
			bool strGet( const Element* e, std::string &s ) const {
				return 0;
			}

			/// This Finfo does not support recvFuncs.
			RecvFunc recvFunc() const {
					return 0;
			}
			
			void countMessages( 
					unsigned int& srcNum, unsigned int& destNum ) {
				srcIndex_ = srcNum++;
			}

			const Finfo* match( 
				const Element* e, unsigned int connIndex ) const;

			bool isTransient() const {
					return 0;
			}

			/**
			 * Returns true only if the other finfo is the same type
			 * In addition it copies over the slot indexing.
			 */
			bool inherit( const Finfo* baseFinfo );

			unsigned int getSlotIndex() const {
				return srcIndex_;
			}

			Finfo* copy() const {
				return new SrcFinfo( *this );
			}

		private:
			unsigned int srcIndex_;
};

#endif // _SRC_FINFO_H
