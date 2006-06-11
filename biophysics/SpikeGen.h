#ifndef _SpikeGen_h
#define _SpikeGen_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,** also known as GENESIS 3 base code.**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS** It is made available under the terms of the** GNU Lesser General Public License version 2.1** See the file COPYING.LIB for the full notice.
**********************************************************************/
class SpikeGen
{
	friend class SpikeGenWrapper;
	public:
		SpikeGen()
		{
			threshold_ = 0.0;
			absoluteRefractoryPeriod_ = 0.0;
			amplitude_ = 1.0;
			state_ = 0.0;
			lastEvent_ = 0.0;
		}

	private:
		double threshold_;
		double absoluteRefractoryPeriod_;
		double amplitude_;
		double state_;
		double lastEvent_;
};
#endif // _SpikeGen_h
