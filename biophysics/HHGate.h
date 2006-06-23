#ifndef _HHGate_h
#define _HHGate_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class HHGate
{
	friend class HHGateWrapper;
	public:
		HHGate()
		{
			takePower_ = power0;
			instant_ = 0;
		}

	private:
		double power_;
		double state_;
		int instant_;
		Interpol A_;
		Interpol B_;
		double g_;	
		bool useConcentration_;	
		static double power0( double x ) {
			return 1.0;
		}
		static double power1( double x ) {
			return x;
		}
		static double power2( double x ) {
			return x * x;
		}
		static double power3( double x ) {
			return x * x * x;
		}
		static double power4( double x ) {
			return power2( x * x );
		}
		double ( *takePower_ )( double );
};
#endif // _HHGate_h
