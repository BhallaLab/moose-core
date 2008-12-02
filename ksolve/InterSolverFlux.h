/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _INTER_SOLVER_FLUX_H
#define _INTER_SOLVER_FLUX_H
class InterSolverFlux
{
	public:
		InterSolverFlux();
		InterSolverFlux( vector< double* > localPtrs,
			vector< double > fluxRate );

		static string getMethod( Eref e );
		static void setMethod( const Conn* c, string method );
		void innerSetMethod( const string& method );

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

		static void sumFlux( const Conn* c, vector< double > n );
		void innerSumFlux( vector< double >& n );
		static void processFunc( const Conn* c, ProcInfo info );
		void innerProcessFunc( Eref e, ProcInfo info );
		static void transferFunc( const Conn* c, ProcInfo info );
		void innerTransferFunc( Eref e, ProcInfo info );
		static void reinitFunc( const Conn* c, ProcInfo info  );

	private:
		string method_;
		vector< double* > localPtrs_; // Pointers to diffusing S_ entries 
//		vector< double > inRate_;		// Influx rates
		vector< double > fluxRate_;		// Efflux rates.
		// Should I pre-multiply?
		// vector< double > lastLocal_; // Used for trapezoidal integ.
		// vector< double > lastRemote_; // Used for trapezoidal integ.
		// vector< double > currLocal_; // Used for trapezoidal integ.
		vector< double > flux_; // The total flux through this port.
								// positive is outgoing.
};

extern const Cinfo* initInterSolverFluxCinfo();

#endif // _INTER_SOLVER_FLUX_H
