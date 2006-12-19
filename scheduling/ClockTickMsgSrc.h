/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ClockTickMsgSrc_h
#define _ClockTickMsgSrc_h

// We maintain a list of these. Each goes out to a number of ClockTicks.
class ClockTickMsgSrc
{
	public:
		ClockTickMsgSrc( Element* e )
			: dt_( 1.0 ), stage_( 0 ), nextTime_( 0.0 ),
				nextClockTime_( 0.0 ), next_( 0 ), index_( 0 )
		{
			;
		}
		~ClockTickMsgSrc( );

		ClockTickMsgSrc( Element* e, Element* target,
			unsigned long index );
		void start( ProcInfo info, double maxTime, 
			NMsgSrc1< ProcInfo >& processSrc );
		double incrementClock( ProcInfo info, double prevClockTime,
			NMsgSrc1< ProcInfo >& processSrc );
		ClockTickMsgSrc** next() {
			return &next_;
		}

		bool operator<( const ClockTickMsgSrc& other ) const {
			if ( dt_ < other.dt_ ) return 1;
			if ( dt_ == other.dt_ && stage_ < other.stage_ ) return 1;
			return 0;
		}

		bool operator==( const ClockTickMsgSrc& other ) const {
			return ( dt_ == other.dt_ && 
				stage_ == other.stage_ &&
				procFunc_ == other.procFunc_ && 
				reinitFunc_ == other.reinitFunc_ &&
				reschedFunc_ == other.reschedFunc_ );
		}
		double dt() {
			return dt_;
		}

		void updateDt( double newdt, unsigned long index );

		// Swaps the order of the ClockTickMsgSrc in their linked list
		void swap( ClockTickMsgSrc** other );

		void updateNextClockTime( );

		void schedNewObject( Element* object );

	private:
		double dt_;
		int stage_;
		double nextTime_;
		double nextClockTime_;
		RecvFunc procFunc_;
		RecvFunc reinitFunc_;
		RecvFunc reschedFunc_;
		// PlainMultiConn conn_;
		ClockTickMsgSrc* next_;
		Op1< ProcInfo > op_;
		Element* target_;
		unsigned long index_;
};

#endif // _ClockTickMsgSrc_h
