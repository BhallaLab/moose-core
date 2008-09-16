/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _Stoich_h
#define _Stoich_h
class Stoich
{
#ifdef DO_UNIT_TESTS
	friend void testStoich();
#endif
	public:
		Stoich();
		
		///////////////////////////////////////////////////
		// Field function definitions
		///////////////////////////////////////////////////
		static unsigned int getNmols( Eref e );
		static unsigned int getNvarMols( Eref e );
		static unsigned int getNsumTot( Eref e );
		static unsigned int getNbuffered( Eref e );
		static unsigned int getNreacs( Eref e );
		static unsigned int getNenz( Eref e );
		static unsigned int getNmmEnz( Eref e );
		static unsigned int getNexternalRates( Eref e );
		static void setUseOneWayReacs( const Conn* c, int value );
		static bool getUseOneWayReacs( Eref e );
		static string getPath( Eref e );
		static void setPath( const Conn* c, string value );
		static unsigned int getRateVectorSize( Eref e );

		///////////////////////////////////////////////////
		// Msg Dest function definitions
		///////////////////////////////////////////////////
		static void scanTicks( const Conn* c );
		static void reinitFunc( const Conn* c );
		static void integrateFunc( 
			const Conn* c, vector< double >* v, double dt );

		unsigned int nVarMols() const {
			return nVarMols_;
		}
		void clear( Eref stoich );

		///////////////////////////////////////////////////
		// Functions used by the GslIntegrator
		///////////////////////////////////////////////////
#ifdef USE_GSL
		static int gslFunc( double t, const double* y, 
			double* yprime, void* params);

		int innerGslFunc( double t, const double* y, double* yprime );

		// Dangerous func, meant only for the GslIntegrator which is
		// permitted to look at the insides of the Stoich class.
		double* S() {
			return &S_[0];
		}
		double* Sinit() {
			return &Sinit_[0];
		}
		void runStats();
#endif // USE_GSL
		void rebuildMatrix( Eref stoich, vector< Id >& ret );
		void localScanTicks( Eref stoich );
	private:
		///////////////////////////////////////////////////
		// Setup function definitions
		///////////////////////////////////////////////////
		void localSetPath( Eref e, const string& value );

		void setupMols(
			Eref e,
			vector< Eref >& varMolVec,
			vector< Eref >& bufVec,
			vector< Eref >& sumTotVec
			);

		void addSumTot( Eref e );

		/**
		 * Finds all target molecules of the specified msgField on 
		 * Eref e. Puts the points into the vector ret, which is 
		 * cleaned out first.
		 * This function replaces findIncoming and findReactants.
		 */
		bool findTargets(
			Eref e, const string& msgFieldName, 
			vector< const double* >& ret );

		void fillHalfStoich( const double* baseptr, 
			vector< const double* >& reactant,
		       	int sign, int reacNum );

		void fillStoich( 
			const double* baseptr, 
			vector< const double* >& sub,
			vector< const double* >& prd, 
			int reacNum );

		void addReac( Eref stoich, Eref e );
		bool checkEnz( Eref e,
				vector< const double* >& sub,
				vector< const double* >& prd,
				vector< const double* >& enz,
				vector< const double* >& cplx,
				double& k1, double& k2, double& k3,
				bool isMM
		);
		void addEnz( Eref stoich, Eref e );
		void addMmEnz( Eref stoich, Eref e );
		void addTab( Eref stoich, Eref e );
		void addRate( Eref stoich, Eref e );
		void setupReacSystem( Eref stoich );

		///////////////////////////////////////////////////
		// These functions control the updates of state
		// variables by calling the functions in the StoichMatrix.
		///////////////////////////////////////////////////
		void updateV( );
		void updateRates( vector< double>* yprime, double dt  );
		
		///////////////////////////////////////////////////
		// Internal fields.
		///////////////////////////////////////////////////
		unsigned int nMols_;
		unsigned int nVarMols_;
		unsigned int nSumTot_;
		unsigned int nBuffered_;
		unsigned int nReacs_;
		unsigned int nEnz_;
		unsigned int nMmEnz_;
		unsigned int nExternalRates_;
		bool useOneWayReacs_;
		string path_;
		vector< double > S_; 	
		vector< double > Sinit_; 	
		vector< double > v_;	
		vector< RateTerm* > rates_; 
		vector< SumTotal > sumTotals_;
		KinSparseMatrix N_; 
		vector< int > path2mol_;
		vector< int > mol2path_;
		map< Eref, unsigned int > molMap_;
#ifdef DO_UNIT_TESTS
		map< Eref, unsigned int > reacMap_;
#endif
		static const double EPSILON;
		///////////////////////////////////////////////////
		// Fields used by the GslIntegrator
		///////////////////////////////////////////////////
		const double* lasty_;
		unsigned int nVarMolsBytes_;
		unsigned int nCopy_;
		unsigned int nCall_;
};
#endif // _Stoich_h
