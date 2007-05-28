/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _GSL_INTEGRATOR_H
#define _GSL_INTEGRATOR_H
class GslIntegrator
{
	public:
		GslIntegrator();

		static bool getIsInitialized( const Element* e );
		static string getMethod( const Element* e );
		static void setMethod( const Conn& c, string method );
		void innerSetMethod( const string& method );

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

		static void allocateFunc( const Conn& c, vector< double >* y );
		void allocateFuncLocal( vector< double >*  y );
		static void processFunc( const Conn& c, ProcInfo info );
		void innerProcessFunc( Element* e, ProcInfo info );
		static void reinitFunc( const Conn& c, ProcInfo info  );

		static void assignStoichFunc( const Conn& c, void* stoich );
		void assignStoichFuncLocal( void* stoich );

	private:
		bool isInitialized_;
		string method_;
		double absAccuracy_;
		double relAccuracy_;
		double internalStepSize_;
		double* y_;
		unsigned int nVarMols_;
		void* stoich_;
		const gsl_odeiv_step_type* gslStepType_;
		gsl_odeiv_step* gslStep_;
		gsl_odeiv_control* gslControl_;
		gsl_odeiv_evolve* gslEvolve_;
		gsl_odeiv_system gslSys_;
};
#endif // _GSL_INTEGRATOR_H
