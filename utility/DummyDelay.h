/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _DummyDelay_h
#define _DummyDelay_h
class DummyDelay
{
	friend class DummyDelayWrapper;
	public:
		DummyDelay()
		{
			threshold_ = 0.0;
			delay_ = 0;
		}

	private:
		double threshold_;
		int delay_;
		double amplitude_;
		int stepsRemaining_;	
};
#endif // _DummyDelay_h
