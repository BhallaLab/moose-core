#ifndef _Compartment_h
#define _Compartment_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
class Compartment
{
	friend class CompartmentWrapper;
	public:
		Compartment()
		{
			Vm_ = -0.06;
			Em_ = -0.06;
			Cm_ = 1.0;
			Rm_ = 1.0;
			Ra_ = 1.0;
			Im_ = 0.0;
			Inject_ = 0.0;
			initVm_ = -0.06;
		}

	private:
		double Vm_;
		double Em_;
		double Cm_;
		double Rm_;
		double Ra_;
		double Im_;
		double Inject_;
		double initVm_;
		double diameter_;
		double length_;
		double A_;
		double B_;
		double dt_;
		double invRm_;
		double sumInject_;
		static const double EPSILON;
};
#endif // _Compartment_h
