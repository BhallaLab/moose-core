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
		virtual ~Stoich();
		
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
		static vector< Id > getPathVec( Eref e );
		static void setPathVec( const Conn* c, vector< Id > value );
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
		unsigned int nReacs() const {
			return nReacs_;
		}
		void clear( Eref stoich );

		// funcs to handle externally imposed changes in mol N
		static void setMolN( const Conn* c, double y, unsigned int i );
		// Virtual so that derived classes handle things like
		// molecule dependencies.
		virtual void innerSetMolN( 
			const Conn* c, double y, unsigned int i );

		static void rescaleVolume( const Conn* c, double ratio );
		void innerRescaleVolume( double ratio );

		static void setBuffer( const Conn* c, 
			int mode, unsigned int mol );
		void innerSetBuffer( int mode, unsigned int mol );


		/**
 		* Puts the data into a new entry in the flux vector, and creates
 		* a stub child for handling the messages to and from the entry.
 		*/
		static void makeFlux( const Conn* c, 
			string stubName, vector< unsigned int >molIndices, 
			vector< double > fluxRates );
		void innerMakeFlux( Eref e,
			string stubName, vector< unsigned int >molIndices, 
			vector< double > fluxRates );

		static void startFromCurrentConcs( const Conn* c ); 
		void innerStartFromCurrentConcs();
		static void requestY( const Conn* c );

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
		vector< double >& velocity() {
			return v_;
		}
		int getStoichEntry( unsigned int i, unsigned int j ) {
			return N_.get( i, j );
		}
		const KinSparseMatrix& N() const {
			return N_;
		} 
#endif // USE_GSL
		void updateV( );
		/**
 		 * Virtual function to make the data structures from the 
 		 * object oriented specification of the signaling network.
 		 */
		virtual void rebuildMatrix( Eref stoich, vector< Id >& ret );

		void localScanTicks( Eref stoich );

		/**
		 * Nasty function to return ptr to dynamicBuffers, used in 
		 * GslIntegrator.
		 */
		const vector< unsigned int >* dynamicBuffers() const {
			return &dynamicBuffers_;
		}

	protected:
		///////////////////////////////////////////////////
		// Setup function definitions
		///////////////////////////////////////////////////
		virtual void localSetPath( Eref e, const string& value );
		void localSetPathVec( Eref e, vector< Id >& value );

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
		       	int sign, unsigned int reacNum );

		void fillStoich( 
			const double* baseptr, 
			vector< const double* >& sub,
			vector< const double* >& prd, 
			unsigned int reacNum );

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
		void updateRates( vector< double>* yprime, double dt  );
		void updateDynamicBuffers();
		
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
		vector< Id > pathVec_;

		/**
		 * S_ holds the state variables: n for all the molecules. This
		 * includes the variable as well as the buffered and sumtotal
		 * molecules.
		 */
		vector< double > S_; 	

		/**
		 * Sinit_ holds the initial values for all the molecules.
		 */
		vector< double > Sinit_; 	

		/**
		 * v is the velocity of each reaction. Its size is numRates. 
		 */
		vector< double > v_;	

		/**
		 * This is the vector of the actual rate calculations
		 */
		vector< RateTerm* > rates_; 

		vector< SumTotal > sumTotals_;

		/**
		 * Indices of molecules whose buffer state changes during
		 * the simulation. The system goes through this list and
		 * assigns any entries to their Sinit_. Only works if the
		 * molecule starts out as non-buffered.
		 */
		vector< unsigned int > dynamicBuffers_;

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

		///////////////////////////////////////////////////
		// Fields used for coupling between solvers: developmental
		///////////////////////////////////////////////////
		vector< InterSolverFlux* > flux_;
		/*
		vector< double* > fluxMol_;	// Pointers to diffusing S_ entries 
		vector< double > fluxRates_;		// Flux scale factors
		vector< double > prevFluxMol_; // Used for trapezoidal integ.
		*/
		// vector< unsigned int > fluxMap_; // Redundant 
};

extern const Cinfo* initStoichCinfo();
#endif // _Stoich_h
