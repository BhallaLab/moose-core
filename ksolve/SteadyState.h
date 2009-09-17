/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SteadyState_h
#define _SteadyState_h
class SteadyState
{
#ifdef DO_UNIT_TESTS
	friend void testSteadyState();
#endif
	public:
		SteadyState();
		~SteadyState();
		
		///////////////////////////////////////////////////
		// Field function definitions
		///////////////////////////////////////////////////
		static bool badStoichiometry( Eref e );
		static bool isInitialized( Eref e );
		static unsigned int getRank( Eref e );
		static unsigned int getNvarMols( Eref e );
		static unsigned int getNiter( Eref e );
		static unsigned int getMaxIter( Eref e );
		static void setMaxIter( const Conn* c, unsigned int value );
		static string getStatus( Eref e );
		static double getConvergenceCriterion( Eref e );
		static void setConvergenceCriterion( const Conn* c, double value );
		static double getTotal( Eref e, const unsigned int& i );
		static void setTotal( 
			const Conn* c, double val, const unsigned int& i );
		double localGetTotal( const unsigned int& i ) const;
		void localSetTotal( double val, const unsigned int& i );
		static double getEigenvalue( Eref e, const unsigned int& i );
		static void setEigenvalue( 
			const Conn* c, double val, const unsigned int& i );
		double localGetEigenvalue( const unsigned int& i ) const;
		static unsigned int getStateType( Eref e );
		static unsigned int getNnegEigenvalues( Eref e );
		static unsigned int getNposEigenvalues( Eref e );
		static unsigned int getSolutionStatus( Eref e );

		///////////////////////////////////////////////////
		// Msg Dest function definitions
		///////////////////////////////////////////////////
		static void setupMatrix( const Conn* c );
		static void settleFunc( const Conn* c );
		static void resettleFunc( const Conn* c );
		void settle( bool forceSetup );
		static void showMatricesFunc( const Conn* c );
		void showMatrices();
		static void randomizeInitialConditionFunc( const Conn* c );
		void randomizeInitialCondition(Eref e);
		static void assignY( const Conn* c, double* S );
		// static void randomInitFunc( const Conn* c );
		// void randomInit();
		////////////////////////////////////////////////////
		// Utility functions for randomInit
		////////////////////////////////////////////////////
		/*
		int isLastConsvMol( int i );
		void recalcRemainingTotal(
			vector< double >& y, vector< double >& tot );
		*/
		void fitConservationRules( 
			gsl_matrix* U, 
			const vector< double >& eliminatedTotal,
			vector< double >&yi
		);
		
		////////////////////////////////////////////////////
		// funcs to handle externally imposed changes in mol N
		////////////////////////////////////////////////////
		static void setMolN( const Conn* c, double y, unsigned int i );
		static void assignStoichFunc( const Conn* c, void* stoich );
		void assignStoichFuncLocal( void* stoich );
		void classifyState( const double* T );
		static const double EPSILON;
		static const double DELTA;

	private:
		void setupSSmatrix();
		
		///////////////////////////////////////////////////
		// Internal fields.
		///////////////////////////////////////////////////
		unsigned int nIter_;
		unsigned int maxIter_;
		bool badStoichiometry_;
		string status_;
		bool isInitialized_;
		bool isSetup_;
		double convergenceCriterion_;

		gsl_matrix* LU_;
		gsl_matrix* Nr_;
		gsl_matrix* gamma_;
		Stoich* s_;
		unsigned int nVarMols_;
		unsigned int nReacs_;
		unsigned int rank_;

		vector< double > total_;
		bool reassignTotal_;
		unsigned int nNegEigenvalues_;
		unsigned int nPosEigenvalues_;
		vector< double > eigenvalues_;
		unsigned int stateType_;
		unsigned int solutionStatus_;
		unsigned int numFailed_;
};

extern const Cinfo* initSteadyStateCinfo();
#endif // _SteadyState_h
