#ifndef _ConcChan_h
#define _ConcChan_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
class ConcChan
{
	friend class ConcChanWrapper;
	public:
		ConcChan()
		{
			permeability_ = 0.0;
			ENernst_ = 0.0;
			valence_ = 0;
			temperature_ = 300.0;
		}

	private:
		double permeability_;
		double n_;
		double Vm_;
		double ENernst_;
		int valence_;
		double temperature_;
		double inVol_;
		double outVol_;
		double A_;
		double B_;
		double inVolumeScale_;
		double outVolumeScale_;
		double nernstScale_;
		static const double R;
		static const double F;
};
#endif // _ConcChan_h
