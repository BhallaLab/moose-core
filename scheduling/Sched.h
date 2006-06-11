/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Sched_h
#define _Sched_h

class Sched
{
	friend class SchedWrapper;
	public:
		Sched()
		{
		}

	private:
		bool terminate_;
		bool running_;
};
#endif // _Sched_h
