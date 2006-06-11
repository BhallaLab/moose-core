/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Job_h
#define _Job_h
class Job
{
	friend class JobWrapper;
	public:
		Job()
			: terminate_( 0 ), wakeUpTime_( 0.0 ) 
		{
			;
		}

	private:
		int running_;
		int doLoop_;
		int doTiming_;
		double realTimeInterval_;
		int priority_;
		bool terminate_;
		double wakeUpTime_;
};
#endif // _Job_h
