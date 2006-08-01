/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// The header stuff is now copied over verbatim, except that the
// includes go selectively into the Wrapper.cpp


#ifndef _MyClass_h
#define _MyClass_h
class MyClass
{
	friend class MyClassWrapper;
	public:
		MyClass()
		{
			Cm_ = 1e-6;
			values_.reserve( 10 ) ;
		}

	private:
		double Vm_;
		double Cm_;
		double Rm_;
		const double pi_;
		double Ra_;
		double inject_;
		vector < double > coords_;
		vector < double > values_;
		double I_;
		double Ca_;
		double volscale_;
		double Erest_;
		static const double specificMembraneCapacitance_ = 0.01;
};
#endif // _MyClass_h
