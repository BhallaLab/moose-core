/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _MSGSRC_H
#define _MSGSRC_H

/**
 * The MsgSrc class specifies the ranges of conns to be used for
 * messaging, and specifies which function to be used within the
 * range. In case there are additional ranges with different funcs
 * but the same src, we use the next_ index to point to a location
 * further up in the array, past the predefined src range.
 */
class MsgSrc
{
		public:
			MsgSrc( unsigned int begin,
					unsigned int end,
					RecvFunc rf )
					: begin_( begin ), end_( end ), next_( 0 ),
					recvFunc_( rf )
			{;}
			
			MsgSrc( )
					: begin_( 0 ), end_( 0 ), next_( 0 ),
					recvFunc_( dummyFunc )
			{;}

			unsigned int begin() const {
					return begin_;
			}

			unsigned int end() const {
					return end_;
			}

			unsigned int next() const {
					return next_;
			}

			RecvFunc recvFunc() const {
					return recvFunc_;
			}

			void insertWithin() {
				++end_;
			}

			void insertBefore() {
				++begin_;
				++end_;
			}

			void dropConn( unsigned int i ) {
				if ( i < end_ ) {
					--end_;
					if ( i < begin_ )
						--begin_;
				}
			}

			void setFunc( RecvFunc rf ) {
					recvFunc_ = rf;
			}

			void setNext( unsigned int i ) {
					next_ = i;
			}

		private:
			unsigned int begin_;
			unsigned int end_;
			unsigned int next_; /// Offset of next range, or 0 if none.
			RecvFunc recvFunc_;
};

#endif // _MSGSRC_H
