/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This class manages information going into and out of the async queue.
 */
class Qinfo
{
	public:
		Qinfo( FuncId f, unsigned int s, MsgId m )
			: f_( f ), s_( s ), m_( m )
		{;}

		void addToQ( vector< char >& q_, const char* arg ) const
		{
			q_.resize( q_.size() + sizeof( Qinfo ) + s_ );
			char* pos = &( q_.back() );
			memcpy( pos, this, sizeof( Qinfo ) );
			memcpy( pos + sizeof( Qinfo ), arg, s_ );
		}
		
	private:
		FuncId f_;
		unsigned int s_;
		MsgId m_;
};
