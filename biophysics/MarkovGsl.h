/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MARKOVGSL_H
#define _MARKOVGSL_H

//Author : Vishaka Datta S, 2011, NCBS

//This is a slightly modified version of the GSL integrator from the ksolve
//library. Minor changes have been made to adapt it to the Markov channel
//system. Importantly, it is not a MOOSE class. This has been done because the
//use of GSL for the Markov Channel is only a temporary affair. 

class MarkovGsl
{
	public:
		MarkovGsl( );
		~MarkovGsl();

		bool getIsInitialized() const;
		string getMethod() const;
		void setMethod( string method );
		double getRelativeAccuracy() const;
		void setRelativeAccuracy( double value );
		double getAbsoluteAccuracy() const;
		void setAbsoluteAccuracy( double value );
		double getInternalDt() const;
		void setInternalDt( double value );
		const gsl_odeiv_step_type* getGslStepType( ) const;
		void init( gsl_odeiv_system, unsigned int );

		void solve( double, double, double* );

	private:
		bool isInitialized_;
		string method_;
		double absAccuracy_;
		double relAccuracy_;
		double internalStepSize_;
		unsigned int nVars_;

		const gsl_odeiv_step_type* gslStepType_;
		gsl_odeiv_step* gslStep_;
		gsl_odeiv_control* gslControl_;
		gsl_odeiv_evolve* gslEvolve_;
		gsl_odeiv_system gslSys_;
};
#endif // _MARKOVGSL_H
