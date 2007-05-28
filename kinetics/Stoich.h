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
		static unsigned int getNmols( const Element* e );
		static unsigned int getNvarMols( const Element* e );
		static unsigned int getNsumTot( const Element* e );
		static unsigned int getNbuffered( const Element* e );
		static unsigned int getNreacs( const Element* e );
		static unsigned int getNenz( const Element* e );
		static unsigned int getNmmEnz( const Element* e );
		static unsigned int getNexternalRates( const Element* e );
		static void setUseOneWayReacs( const Conn& c, int value );
		static bool getUseOneWayReacs( const Element* e );
		static string getPath( const Element* e );
		static void setPath( const Conn& c, string value );
		static unsigned int getRateVectorSize( const Element* e );

		///////////////////////////////////////////////////
		// Msg Dest function definitions
		///////////////////////////////////////////////////
		static void reinitFunc( const Conn& c );
		static void integrateFunc( 
			const Conn& c, vector< double >* v, double dt );

		unsigned int nVarMols() const {
			return nVarMols_;
		}

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
#endif // USE_GSL
	private:
		///////////////////////////////////////////////////
		// Setup function definitions
		///////////////////////////////////////////////////
		void localSetPath( Element* e, const string& value );

		void setupMols(
			Element* e,
			vector< Element* >& varMolVec,
			vector< Element* >& bufVec,
			vector< Element* >& sumTotVec
			);

		void addSumTot( Element* e );

		unsigned int findReactants( 
			Element* e, const string& msgFieldName, 
			vector< const double* >& ret );

		unsigned int findProducts( 
			Element* e, const string& msgFieldName, 
			vector< const double* >& ret );

		void fillHalfStoich( const double* baseptr, 
			vector< const double* >& reactant,
		       	int sign, int reacNum );
		void fillStoich( 
			const double* baseptr, 
			vector< const double* >& sub,
			vector< const double* >& prd, 
			int reacNum );

		void addReac( Element* stoich, Element* e );
		bool checkEnz( Element* e,
				vector< const double* >& sub,
				vector< const double* >& prd,
				vector< const double* >& enz,
				vector< const double* >& cplx,
				double& k1, double& k2, double& k3,
				bool isMM
		);
		void addEnz( Element* stoich, Element* e );
		void addMmEnz( Element* stoich, Element* e );
		void addTab( Element* stoich, Element* e );
		void addRate( Element* stoich, Element* e );
		void setupReacSystem( );

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
		vector< int > sumTotals_;
		SparseMatrix N_; 
		vector< int > path2mol_;
		vector< int > mol2path_;
		map< const Element*, unsigned int > molMap_;
#ifdef DO_UNIT_TESTS
		map< const Element*, unsigned int > reacMap_;
#endif
		static const double EPSILON;
		///////////////////////////////////////////////////
		// Fields used by the GslIntegrator
		///////////////////////////////////////////////////
		const double* lasty_;
		unsigned int nVarMolsBytes_;
		unsigned int nCopy_;
};
#endif // _Stoich_h
