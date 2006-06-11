#ifndef _CaConc_h
#define _CaConc_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
class CaConc
{
	friend class CaConcWrapper;
	public:
		CaConc()
		{
			Ca_ = 0.0;
			CaBasal_ = 0.0;
			tau_ = 1.0;
			B_ = 1.0;
		}

	private:
		double Ca_;
		double CaBasal_;
		double tau_;
		double B_;
		double c_;
		double activation_;
};
#endif // _CaConc_h
