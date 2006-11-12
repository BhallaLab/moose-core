/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#define mtrandf(l,h)     (l)==(h) ? (l) : mtrand() * ((h)-(l)) + (l)

#ifndef _RandomSpike_h
#define _RandomSpike_h
class RandomSpike
{
	friend class RandomSpikeWrapper;
	public:
		RandomSpike()
		{
			absoluteRefractoryPeriod_ = 0.0;
			state_ = 0.0;
			reset_ = 1;
			resetValue_ = 0.0;
			minAmp_ = 1.0;
			maxAmp_ = 1.0;
		}

	private:
		double rate_;
		double absoluteRefractoryPeriod_;
		double state_;
		int reset_;
		double resetValue_;
		double minAmp_;
		double maxAmp_;
		double lastEvent_;
		double realRate_;
};
#endif // _RandomSpike_h
