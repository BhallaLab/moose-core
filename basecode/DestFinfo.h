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
							RecvFunc rfunc,
					 		const string& doc="",
							unsigned int destIndex = 0 );

			~DestFinfo()
			{;}

			/**
			 * This slightly odd looking operation is meant to 
			 * connect any calls made to this dest onward to other
			 * dests. 
			 * For now we ignore it in the DestFinfo.
			 */
			bool add( 
					Eref e, Eref destElm, const Finfo* destFinfo,
					unsigned int connTainerOption
			) const 
			{
					return 0;
			}
			
			bool respondToAdd(
					Eref e, Eref dest, const Ftype *destType,
					unsigned int& myFuncId, unsigned int& returnFuncId,
					int& destMsgId, unsigned int& destIndex
			) const;
			
			int msg() const;

			/**
			 * Call the RecvFunc with the arguments in the string.
			 */
			bool strSet( Eref e, const std::string &s )
					const;
			
			/// strGet doesn't work for DestFinfo
			bool strGet( Eref e, std::string &s ) const {
				return 0;
			}

			RecvFunc recvFunc() const {
					return rfunc_;
			}

			void countMessages( unsigned int& num );

			bool getSlot( const string& name, Slot& ret ) const;

			const Finfo* match( 
				const Element* e, const ConnTainer* c ) const;

			bool isTransient() const {
					return 0;
			}

			bool inherit( const Finfo* baseFinfo );

			Finfo* copy() const {
				return new DestFinfo( *this );
			}

			/**
			 * Sets up the FuncVec on the DestFinfo, by filling in the
			 * name and RecvFunc.
			 */
			void addFuncVec( const string& cname );

			/**
			 * Returns the func id of this DestFinfo.
			 */
			unsigned int funcId() const {
				return fv_->id();
			}

			/// This is a dest, so it does not send out any info.
			unsigned int syncFuncId() const {
				return 0;
			}

			/// This is a dest, so it does not send out any info.
			unsigned int asyncFuncId() const {
				return 0;
			}

			/// This is a dest, so we do have to define proxyFuncs.
			unsigned int proxyFuncId() const {
				return ftype()->proxyFuncId();
			}

		private:
			/**
			 * This is the function executed when a message arrives at this
			 * Finfo.
			 */
			RecvFunc rfunc_;

			/**
			 * This identifies the msg associated with this DestFinfo.
			 */
			int msg_;

			/**
			 * The FuncVec data structure manages RecvFuncs
			 */
			FuncVec* fv_;
};

#endif // _DEST_FINFO_H
