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
		CaConc();
		///////////////////////////////////////////////////////////////
		// Message handling functions
		///////////////////////////////////////////////////////////////
		void reinit( const Eref&, ProcPtr info );
		void process( const Eref&, ProcPtr info );

		void current( double I );
		void currentFraction( double I, double fraction );
		void increase( double I );
		void decrease( double I );
		///////////////////////////////////////////////////////////////
		// Field handling functions
		///////////////////////////////////////////////////////////////
		void setCa( double val );
		double getCa() const;
		void setCaBasal( double val );
		double getCaBasal() const;
		void setTau( double val );
		double getTau() const;
		void setB( double val );
		double getB() const;
        void setThickness( double val );
        double getThickness() const;
        void setCeiling( double val );
        double getCeiling() const;
        void setFloor( double val );
        double getFloor() const;

		static const Cinfo* initCinfo();
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


#endif // _CACONC_H
