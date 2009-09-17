/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _CACONC_H
#define _CACONC_H

/**
 * The CaConc object manages calcium dynamics in a single compartment
 * without diffusion. It uses a simple exponential return of Ca
 * to baseline, with influxes from ion channels. It solves the
 * equation:
 * dC/dt = B*Ik - C/tau
 * where Ca = Ca_base + C.
 *
 * From the GENESIS notes:
 * In SI units, where concentration is moles/m^3
 * (milli-moles/liter) and current is in amperes, theory gives B
 * = 5.2e-6/(shell volume).  In practice, B is a parameter to be
 * fitted or estimated from experiment, as buffering, non-uniform
 * distribution of Ca, etc., will modify this value.  If thick =
 * 0, the readcell routine calculates B by dividing the "density"
 * parameter in the cell parameter file by the volume of the
 * compartment.  Otherwise, it scales as a true shell, with the
 * volume of a shell having thickness thick.  A negative value of
 * the "density" parameter may be used to indicate that it should
 * be taken as an absolute value of B, without scaling. 
 */

class CaConc
{
	public:
		CaConc()
		{
			Ca_ = 0.0;
			CaBasal_ = 0.0;
			tau_ = 1.0;
			B_ = 1.0;
                        thickness_ = 0.0;
                        ceiling_ = DBL_MAX;
                        floor_ = -DBL_MAX;
		}

		///////////////////////////////////////////////////////////////
		// Message handling functions
		///////////////////////////////////////////////////////////////
		static void reinitFunc( const Conn* c, ProcInfo info );
		void innerReinitFunc( const Conn* c );
		static void processFunc( const Conn* c, ProcInfo info );
		void innerProcessFunc( const Conn* conn, ProcInfo info );

		static void currentFunc( const Conn* c, double I );
		static void currentFractionFunc(
				const Conn* c, double I, double fraction );
		static void increaseFunc( const Conn* c, double I );
		static void decreaseFunc( const Conn* c, double I );
		static void basalMsgFunc( const Conn* c, double value );
		///////////////////////////////////////////////////////////////
		// Field handling functions
		///////////////////////////////////////////////////////////////
		static void setCa( const Conn* c, double val );
		static double getCa( Eref e );
		static void setCaBasal( const Conn* c, double val );
		static double getCaBasal( Eref e );
		static void setTau( const Conn* c, double val );
		static double getTau( Eref e );
		static void setB( const Conn* c, double val );
		static double getB( Eref e );
                static void setThickness( const Conn* c, double val );
                static double getThickness( Eref e);
                static void setCeiling( const Conn* c, double val );
                static double getCeiling( Eref e);
                static void setFloor( const Conn* c, double val );
                static double getFloor( Eref e);
	private:
		double Ca_;
		double CaBasal_;
		double tau_;
		double B_;
		double c_;
		double activation_;
                double thickness_;
                double ceiling_;
                double floor_;
};

extern const Cinfo* initCaConcCinfo();

#endif // _CACONC_H
