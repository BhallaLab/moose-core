/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Molecule_h
#define _Molecule_h
class Molecule
{
	public:
		Molecule()
		{
			nInit_ = 0.0;
			volumeScale_ = 1.0;
			n_ = 0.0;
			mode_ = 0;
		}

	private:
		double nInit_;
		double volumeScale_;
		double n_;
		int mode_;
		double total_;
		double A_;
		double B_;
		static const double EPSILON;
};
#endif // _Molecule_h
