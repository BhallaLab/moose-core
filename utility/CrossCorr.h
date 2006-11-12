/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _CrossCorr_h
#define _CrossCorr_h
class CrossCorr
{
	friend class CrossCorrWrapper;
	public:
		CrossCorr()
		{
			threshold_ = 0.0;
			binCount_ = 0;
			aSpikeCount_ = 0;
			bSpikeCount_ = 0;
		}

	private:
		double threshold_;
		int binCount_;
		double binWidth_;
		int aSpikeCount_;
		int bSpikeCount_;
		deque<double>  aSpikeTime_;
		deque<double>  bSpikeTime_;
		vector<int>    correlogram_;
		double         ccWidth_;
};
#endif // _CrossCorr_h
