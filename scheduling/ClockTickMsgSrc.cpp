/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ClockTickMsgSrc.h"
// #include "Job.h"
// #include "JobWrapper.h"
// #include "ClockJob.h"
// #include "ClockJobWrapper.h"


////////////////////////////g///////////////////////////////////
// Member functions for ClockTickMsgSrc
////////////////////////////////////////////////////////////////

ClockTickMsgSrc::ClockTickMsgSrc( Element* e, Element* target, unsigned long index )
	: dt_( 1.0 ), nextTime_( 0.0 ), nextClockTime_( 0.0 ),
		next_( 0 ), target_( target ), index_( index )
{
 	if ( Ftype1< double >::get( target, "dt", dt_ ) ) {
 		if ( Ftype1< int >::get( target, "stage", stage_ ) ) {
			// Hacks all, to set up the recvFuncs in either direction.
			procFunc_ = target->field( "processIn" )->recvFunc();
			reinitFunc_ = target->field( "reinitIn" )->recvFunc();
			reschedFunc_ = target->field( "reschedIn" )->recvFunc();
			return;
		}
	}
	cerr << "Error: Failed to initialize ClockTickMsgSrc\n";
}

ClockTickMsgSrc::~ClockTickMsgSrc( )
{
	// Disconnect!! Actually the destruction of the Conn should do it.
	if ( next_ )
		delete next_;
}

//    Here is the func that scans through child clockticks
// It can be stopped if info.halt == 1. It can resume without
// a hiccup at any point.
// It stops as soon as any clock exceeds maxTime.
void ClockTickMsgSrc::start( ProcInfo info, double maxTime, 
		NMsgSrc1< ProcInfo >& processSrc )
{
	static double NEARLY_ONE = 0.999999999999;
	static double JUST_OVER_ONE = 1.000000000001;
	double endTime;
	// setProcInfo( info );
	maxTime = maxTime * NEARLY_ONE;

	while ( info->currTime_ < maxTime ) {
		endTime = maxTime + dt_;
		if ( next_ && endTime > nextClockTime_ )
			endTime = nextClockTime_ * JUST_OVER_ONE;

		info->dt_ = dt_;
		while ( nextTime_ <= endTime ) {
			info->currTime_ = nextTime_;
			processSrc.sendTo( index_, info );
			// for_each( conn_.begin(), conn_.end(), op_ );
			nextTime_ += dt_;
		}

		if ( next_ ) {
			while ( nextClockTime_ < nextTime_ )
				nextClockTime_ = 
					next_->incrementClock( info, nextTime_, processSrc);
		}
	}
}

// Increment whichever clock is due for updating. Cascades to multiple
// clocks if possible.
// Return the smallest nextt of this and further clocks
// Assume dt of successive clocks is in increasing order.
// Assume that prevClockTime exceeds nextTime_: this func wouldn't be
// called otherwise.
double ClockTickMsgSrc::incrementClock( 
	ProcInfo info, double prevClockTime,
		NMsgSrc1< ProcInfo >& processSrc )
{
	if ( next_ ) {
		if ( nextTime_ <= nextClockTime_ ) {
			info->currTime_ = nextTime_;
			info->dt_ = dt_;
			processSrc.sendTo( index_, info );
			nextTime_ += dt_;
		}
		if ( nextTime_ > nextClockTime_ && 
			prevClockTime > nextClockTime_ ) {
			nextClockTime_ = 
				next_->incrementClock( info, prevClockTime, processSrc);
		}
		if ( nextClockTime_ < nextTime_ )
			return nextClockTime_;
	} else {
		info->currTime_ = nextTime_;
		info->dt_ = dt_;
		processSrc.sendTo( index_, info );
		nextTime_ += dt_;
	}
	return nextTime_;
}

void ClockTickMsgSrc::updateDt( double newdt, unsigned long index )
{
	if ( index == index_ ) {
		nextTime_ += newdt - dt_;
		dt_ = newdt;
	} else if (next_) {
		next_->updateDt( newdt, index );
	}
}

// Swaps the order of the ClockTickMsgSrc in their linked list
// this is the second
// other is the first, so (*other)->next_ == this
void ClockTickMsgSrc::swap( ClockTickMsgSrc** other )
{
	ClockTickMsgSrc* temp = next_;
	next_ = *other;
	( *other )->next_ = temp;
	*other = this;
}

void ClockTickMsgSrc::updateNextClockTime( )
{
	if ( next_ ) {
		nextClockTime_ = next_->nextTime_;
		next_->updateNextClockTime();
	} else {
		nextClockTime_ = 0;
	}
}

void ClockTickMsgSrc::schedNewObject( Element* object )
{
	cout << "ClockTickMsgSrc::schedNewObject: placeholder\n";
}
