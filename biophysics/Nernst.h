/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Nernst_h
#define _Nernst_h
class Nernst
{
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

	///////////////////////////////////////////////////
	// Field function definitions
	///////////////////////////////////////////////////
	static double getE( Eref e );

	void localSetTemperature( double value );
	static void setTemperature( const Conn* c, double value );
	static double getTemperature( Eref e );

	void localSetValence( int value );
	static void setValence( const Conn* c, int value );
	static int getValence( Eref e );

	static void setCin( const Conn* c, double value );
	static double getCin( Eref e );

	static void setCout( const Conn* c, double value );
	static double getCout( Eref e );

	static void setScale( const Conn* c, double value );
	static double getScale( Eref e );


	///////////////////////////////////////////////////
	// Dest function definitions
	///////////////////////////////////////////////////

	void cinFuncLocal( const Conn* c, double conc );
	static void cinFunc( const Conn* c, double value );

	void coutFuncLocal( const Conn* c, double conc );
	static void coutFunc( const Conn* c, double value );

	private:
		void updateE( );
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

// Used by solver, readcell, etc.
extern const Cinfo* initNernstCinfo();

#endif // _Nernst_h
