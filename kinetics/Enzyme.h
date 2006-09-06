/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// We need to manually initialize sk1 to 1.0, till mpp is fixed.
#ifndef _Enzyme_h
#define _Enzyme_h
class Enzyme
{
	friend class EnzymeWrapper;
	public:
		Enzyme()
		{
			k1_ = 0.1;
			k2_ = 0.4;
			k3_ = 0.1;
			sk1_ = 1.0;
		}

	private:
		double k1_;
		double k2_;
		double k3_;
		double sA_;
		double pA_;
		double eA_;
		double B_;
		double e_;
		double s_;
		double sk1_;	
		double Km_;
};
#endif // _Enzyme_h
