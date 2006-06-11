#ifndef _Nernst_h
#define _Nernst_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
class Nernst
{
	friend class NernstWrapper;
	public:
		Nernst()
		{
			E_ = 0.0;
			Temperature_ = 295;
			valence_ = 1;
			Cin_ = 1.0;
			Cout_ = 1.0;
			scale_ = 1.0;
			factor_ = scale_ * R_OVER_F * Temperature_ / valence_;
		}

		void localSetTemperature( double value ) {
			if ( value > 0.0 ) {
				Temperature_ = value;
				factor_ = scale_ * R_OVER_F * Temperature_ / valence_;
			}
		}

		void localSetValence( int value ) {
			if ( value != 0 ) {
				valence_ = value;
				factor_ = scale_ * R_OVER_F * Temperature_ / valence_;
			}
		}

	private:
		double E_;
		double Temperature_;
		int valence_;
		double Cin_;
		double Cout_;
		double scale_;
		double factor_; 
		static const double R_OVER_F;
		static const double ZERO_CELSIUS;
};
#endif // _Nernst_h
