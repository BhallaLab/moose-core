/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _ClockJob_h
#define _ClockJob_h
class ClockJob
{
	friend class ClockJobWrapper;
	public:
		ClockJob()
			: runTime_( 0.0 ), currentTime_( 0.0 ),
			nSteps_( 0 ), currentStep_( 0 )
		{
		}

	private:
		double runTime_;
		double currentTime_;
		int nSteps_;
		int currentStep_;
};
#endif // _ClockJob_h
