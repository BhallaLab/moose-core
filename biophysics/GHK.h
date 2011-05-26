/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _GHK_H
#define _GHK_H

#define GAS_CONSTANT	8.314			/* (V * C)/(deg K * mol) */
#define FARADAY		9.6487e4			/* C / mol */
#define ZERO_CELSIUS	273.15			/* deg */
#define R_OVER_F        8.6171458e-5		/* volt/deg */
#define F_OVER_R        1.1605364e4		/* deg/volt */


/**
 * The GHK class handles the Goldman-Hodgkin-Katz equations for
 * current for a single ion species.
 */
class GHK
{

	public:
		GHK();

		double getIk() const;
		double getGk() const;
		double getEk() const;

		void setTemperature( double T );
		double getTemperature() const;

		void setPermeability( double p );
		double getPermeability() const;
		void addPermeability( double p );


		void setVm( double Vm );
		double getVm() const;

		void setCin( double Cin );
		double getCin() const;

		void setCout( double Cout );
		double getCout() const;

		void setValency( double valecny );
		double getValency() const;

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );

		void handleVm( const Eref& e, const Qinfo* q, double Vm );

		static const Cinfo* initCinfo();

	private:
		double Ik_;
		double Gk_;
		double Ek_;
		double p_;
		double T_;
		double Vm_;
		double Cin_;
		double Cout_;
		double valency_;
		double GHKconst_;
};



#endif // _GHK_H
