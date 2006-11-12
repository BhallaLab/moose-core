/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _AutoCorr_h
#define _AutoCorr_h
class AutoCorr
{
	friend class AutoCorrWrapper;
	public:
		AutoCorr()
		{
			threshold_ = 0.0;
			binCount_ = 0;
			spikeCount_ = 0;
		}

	private:
		double threshold_;
		int binCount_;
		double binWidth_;
		int spikeCount_;
		deque<double>  spikeTime_;
		vector<int>    correlogram_;
		double         ccWidth_;
};
#endif // _AutoCorr_h
