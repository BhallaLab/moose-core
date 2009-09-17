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


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
		static bool getIsInitialized( Eref e );
		static string getMethod( Eref e );
		static void setMethod( const Conn* c, string method );
		void innerSetMethod( const string& method );
		static double getRelativeAccuracy( Eref e );
		static void setRelativeAccuracy( const Conn* c, double value );
		static double getAbsoluteAccuracy( Eref e );
		static void setAbsoluteAccuracy( const Conn* c, double value );
		static double getInternalDt( Eref e );
		static void setInternalDt( const Conn* c, double value );

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

		static void allocateFunc( const Conn* c, vector< double >* y );
		void allocateFuncLocal( vector< double >*  y );
		static void processFunc( const Conn* c, ProcInfo info );
		void innerProcessFunc( Eref e, ProcInfo info );
		static void reinitFunc( const Conn* c, ProcInfo info  );

		static void assignStoichFunc( const Conn* c, void* stoich );
		void assignStoichFuncLocal( void* stoich );
		static void assignY( const Conn* c, double* S );
		void innerAssignY( double* S );

		static void setMolN( const Conn* c, double y, unsigned int i );
		void innerSetMolN( double y, unsigned int i ) ;

	private:
		bool isInitialized_;
		string method_;
		double absAccuracy_;
		double relAccuracy_;
		double internalStepSize_;
		double* y_;
		unsigned int nVarMols_;
		void* stoich_;
		const vector< unsigned int >* dynamicBuffers_;

		const gsl_odeiv_step_type* gslStepType_;
		gsl_odeiv_step* gslStep_;
		gsl_odeiv_control* gslControl_;
		gsl_odeiv_evolve* gslEvolve_;
		gsl_odeiv_system gslSys_;
};
#endif // _GSL_INTEGRATOR_H
