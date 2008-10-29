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
			friend void cinfoTest(); // wants to look at msg_
#endif
			SrcFinfo( const string& name, const Ftype *f, const string& doc="" )
					: Finfo( name, f, doc ), msg_( 0 )
			{;}

			~SrcFinfo()
			{;}

			bool add(
					Eref e, Eref destElm, const Finfo* destFinfo,
					unsigned int connTainerOption
			) const;

			bool respondToAdd(
					Eref e, Eref src, const Ftype *srcType,
					unsigned int& srcFuncId, unsigned int& returnFuncId,
					int& destMsgId, unsigned int& destIndex
			) const;
			
			/**
			 * Returns index of Msg array. Always positive, that is always
			 * a src.
			 */
			int msg() const {
				return msg_;
			}

			/**
			 * Send a message with the arguments in the string.
			 */
			bool strSet( Eref e, const std::string &s )
					const;
			
			/// strGet doesn't work for SrcFinfo
			bool strGet( Eref e, std::string &s ) const {
				return 0;
			}

			/// This Finfo does not support recvFuncs.
			RecvFunc recvFunc() const {
					return 0;
			}
			
			void countMessages( unsigned int& num ) {
				msg_ = num++;
			}

			const Finfo* match( 
				const Element* e, const ConnTainer* c ) const;

			bool isTransient() const {
					return 0;
			}

			/**
			 * Returns true only if the other finfo is the same type
			 * In addition it copies over the slot indexing.
			 */
			bool inherit( const Finfo* baseFinfo );

			bool getSlot( const string& name, Slot& ret ) const;

			Finfo* copy() const {
				return new SrcFinfo( *this );
			}

			void addFuncVec( const string& cname )
			{;}

			bool isDestOnly() const {
				return 0;
			}

			/// Looks at the ftype.
			unsigned int syncFuncId() const {
				return ftype()->syncFuncId();
			}

			/// Looks at the ftype.
			unsigned int asyncFuncId() const {
				return ftype()->asyncFuncId();
			}

			/// Always zero. This cannot be a dest.
			unsigned int proxyFuncId() const {
				return 0;
			}

		private:
			int msg_;
};

#endif // _SRC_FINFO_H
