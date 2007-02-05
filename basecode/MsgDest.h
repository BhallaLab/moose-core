/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _MSGDEST_H
#define _MSGDEST_H

/**
 * The MsgDest class specifies the set of Conns to be used for a
 * message. It just needs the start and end.
 */
class MsgDest
{
		public:
			MsgDest( unsigned int begin, unsigned int end )
					: begin_( begin ), end_( end )
			{;}
			MsgDest( )
					: begin_( 0 ), end_( 0 )
			{;}

			unsigned int begin() const {
					return begin_;
			}

			unsigned int end() const {
					return end_;
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

		private:
			unsigned int begin_;
			unsigned int end_;
};

#endif // _MSGDEST_H
