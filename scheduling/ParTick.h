/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// This variant of ClockTick supports 5 stages involved in 
// managing parallel messaging.
//
// Stage 0: post irecv for this tick.
// Stage 1: Call all processes that have outgoing data on this tick.
// Stage 2: Post send
// Stage 3: Call all processes that only have local data.
// Stage 4: Poll for posted irecvs, as they arrive, send their contents.
//          The poll process relies on return info from each postmaster
//
// Stage 0, 2, 4 pass only tick stage info.
// Stage 1 and 3 pass regular ProcInfo

// Should really happen automatically when mpp sees it is derived.

#ifndef _ParTick_h
#define _ParTick_h
class ParTick
{
	friend class ParTickWrapper;
	public:
		ParTick()
		{
			handleAsync_ = 0;
		}

	private:
		int handleAsync_;
};
#endif // _ParTick_h
