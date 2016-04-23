/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MARKOVGSLSOLVER_H
#define _MARKOVGSLSOLVER_H

////////////////////////////////////////////////////
// Class : MarkovBoostSolver
// Author : Vishaka Datta S, 2011, NCBS.
//
// The GslIntegrator class in ksolve deals with a system whose coefficients stay
// constant. In the case of a Markov channel, the coefficients of the system
// vary with time. 
//
// This makes it necessary for the system to keep track of changes in the system
// matrix, which is implemented by the message handler. 
///////////////////////////////////////////////////

class MarkovBoostSolver
{
	public:
		MarkovBoostSolver();
		~MarkovBoostSolver();

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
		bool getIsInitialized() const;
		string getMethod() const;
		void setMethod( string method );
		double getRelativeAccuracy() const;
		void setRelativeAccuracy( double value );
		double getAbsoluteAccuracy() const;
		void setAbsoluteAccuracy( double value );
		double getInternalDt() const;
		void setInternalDt( double value );

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr info );
		void reinit( const Eref& e, ProcPtr info );

		void init( vector< double > );
		void handleQ( vector< vector< double > > );

		static const Cinfo* initCinfo();
	private:
		bool isInitialized_;
		string method_;
		double absAccuracy_;
		double relAccuracy_;
		double internalStepSize_;
		double* stateGsl_;

		//The following four variables should be members of any solver class that
		//will be implmented.
		unsigned int nVars_;
		vector< double > state_;
		vector< double > initialState_;
		vector< vector< double > > Q_;

                // Gsl replacement
		unsigned int boostStep_;
                unsigned int boostStepType_;
                double* stateBoost_;

//		static int evalSystem( double, const double*, double*, void* );
                static int evalSystem( 
                        const std::vector<double> &
                        , std::vector<double> &
                        , double 
                        , vector< vector< double > > &
                        );
};

#endif 
