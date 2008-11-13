/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _KineticHub_h
#define _KineticHub_h

class KineticHub
{
	public:
		KineticHub();

		///////////////////////////////////////////////////
		// Field function definitions
		///////////////////////////////////////////////////
		static unsigned int getNmol( Eref e );
		static unsigned int getNreac( Eref e );
		static unsigned int getNenz( Eref e );


		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////
		static void processFunc( const Conn* c, ProcInfo info );
		void processFuncLocal( Eref er, ProcInfo info );
		static void reinitFunc( const Conn* c, ProcInfo info );
		static void rateTermFunc( const Conn* c,
			vector< RateTerm* >* rates, 
			KinSparseMatrix* N,
			bool useHalfReacs );
		static void rateSizeFunc( const Conn* c,
			unsigned int nReac, unsigned int nEnz, 
			unsigned int nMmEnz);
		void rateSizeFuncLocal( Eref hub,
			unsigned int nReac, unsigned int nEnz, 
			unsigned int nMmEnz );
		static void molSizeFunc( const Conn* c,
			unsigned int nVarMol, unsigned int nBuf,
			unsigned int nSumTot );
		void molSizeFuncLocal(
			unsigned int nVarMol, unsigned int nBuf,
			unsigned int nSumTot );
		static void molConnectionFunc( const Conn* c,
				vector< double >* S,
				vector< double >* Sinit,
				vector< Eref >* elist
		);
		void molConnectionFuncLocal( Eref e,
				vector< double >* S,
				vector< double >* Sinit,
				vector< Eref >* elist
		);
		static void reacConnectionFunc( const Conn* c,
				unsigned int index, Eref reac );
		void reacConnectionFuncLocal( 
				Eref e,
				int rateTermIndex, 
				Eref reac );
		static void enzConnectionFunc( const Conn* c,
				unsigned int index, Eref enz );
		void enzConnectionFuncLocal(
				Eref e,
				int enzTermIndex, 
				Eref enz );
		static void mmEnzConnectionFunc( const Conn* c,
				unsigned int index, Eref mmEnz );
		void mmEnzConnectionFuncLocal(
				Eref e,
				int enzTermIndex, 
				Eref enz );
		static void childFunc( const Conn* c, int stage );

		
		/// Clears out all messages to solved objects.
		static void clearFunc( const Conn* c );

		///////////////////////////////////////////////////
		// Dest functions for handlng inter-hub flux 
		///////////////////////////////////////////////////
		static void flux( const Conn* c, vector< double > influx );
		
		///////////////////////////////////////////////////
		// Overrides Neutral::destroy to clean up zombies.
		///////////////////////////////////////////////////
		static void destroy( const Conn* c);

		///////////////////////////////////////////////////
		// Functions to override zombie messages
		///////////////////////////////////////////////////
		static void molSum( const Conn* c, double val );

		///////////////////////////////////////////////////
		// Functions to override zombie field access funcs.
		///////////////////////////////////////////////////
		static void setMolN( const Conn* c, double value );
		static double getMolN( Eref e );
		static void setMolNinit( const Conn* c, double value );
		static double getMolNinit( Eref e );
		static void setMolConc( const Conn* c, double value );
		static double getMolConc( Eref e );
		static void setMolConcInit( const Conn* c, double value );
		static double getMolConcInit( Eref e );
		static void setMolMode( const Conn* c, int value );
		static int getMolMode( Eref e );

		static void setReacKf( const Conn* c, double value );
		static double getReacKf( Eref e );
		static void setReacKb( const Conn* c, double value );
		static double getReacKb( Eref e );

		static void setEnzK1( const Conn* c, double value );
		static double getEnzK1( Eref e );
		static void setEnzK2( const Conn* c, double value );
		static double getEnzK2( Eref e );
		static void setEnzK3( const Conn* c, double value );
		static double getEnzK3( Eref e );
		static void setEnzKm( const Conn* c, double value );
		static double getEnzKm( Eref e );
		static void setEnzKcat( const Conn* c, double value );
		static double getEnzKcat( Eref e );

		static void setMmEnzK1( const Conn* c, double value );
		static double getMmEnzK1( Eref e );
		static void setMmEnzK2( const Conn* c, double value );
		static double getMmEnzK2( Eref e );
		static void setMmEnzK3( const Conn* c, double value );
		static void setMmEnzKm( const Conn* c, double value );
		static double getMmEnzKm( Eref e );
		static void setMmEnzKcat( const Conn* c, double value );
		static double getMmEnzKcat( Eref e );


		static void zombify( 
			Eref hub, Eref e, const Finfo* hubFinfo, Finfo* solveFinfo );
	private:
		vector< double >* S_;
		vector< double >* Sinit_;
		vector< RateTerm* >* rates_;
		bool useHalfReacs_;
		bool rebuildFlag_;
		unsigned long nVarMol_;
		unsigned long nBuf_;
		unsigned long nSumTot_;
		vector< unsigned int > reacMap_;
		vector< unsigned int > enzMap_;
		vector< unsigned int > mmEnzMap_;
		vector< unsigned int > molSumMap_;
		vector< unsigned int > nSrcMap_;

		vector< unsigned int > dynamicBuffers_; // vec of buffered mols
		/**
		 * The next field manages exchange of molecules with other
		 * solvers, typically diffusive exchange at specified junctions
		 * between the spatial domains of the respective solvers.
		 * Indexed by # of target hub
		 */
		vector< InterHubFlux > flux_;
};

#endif // _KineticHub_h
