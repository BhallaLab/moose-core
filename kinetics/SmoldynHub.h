/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
** 
** This part of the program uses Smoldyn which is developed by 
** Steven Andrews. The Smoldyn code is in a separate subdirectory.
**********************************************************************/

#ifndef _SmoldynHub_h
#define _SmoldynHub_h

// Forward declaration
struct simstruct;
class KinSparseMatrix;
/**
 * SmoldynHub provides an interface between MOOSE and the internal
 * Smoldyn operations and data structures.
 * This class is a wrapper for the Smoldyn simptr class, which
 * is a complete definition of the Smoldyn simulation.
 */
class SmoldynHub
{
	public:
		SmoldynHub();
		///////////////////////////////////////////////////
		// Zombie utility functions
		///////////////////////////////////////////////////
		static SmoldynHub* getHubFromZombie( const Element* e, 
			const Finfo *f, unsigned int& index );
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////

		void setPos( unsigned int molIndex, double value, 
			unsigned int i, unsigned int dim );
		double getPos( unsigned int molIndex, unsigned int i, 
			unsigned int dim );

		void setPosVector( unsigned int molIndex, 
			const vector< double >& value, unsigned int dim );
		void getPosVector( unsigned int molIndex,
			vector< double >& value, unsigned int dim );

		void setNinit( unsigned int molIndex, unsigned int value );
		unsigned int getNinit( unsigned int molIndex );

// Sets the number of particles of the mol species specified by molIndex
		void setNparticles( unsigned int molIndex, unsigned int value );
// Returns the number of particles of the mol species specified by molIndex
		unsigned int getNparticles( unsigned int molIndex );

		void setD( unsigned int molIndex, double value );
		double getD( unsigned int molIndex );

		unsigned int numSpecies() const;
		static unsigned int getNspecies( const Element* e );
		
		static string getPath( const Element* e );
		static void setPath( const Conn* c, string value );
		void localSetPath( Element* stoich, const string& value );

		static unsigned int getNreac( const Element* e );
		unsigned int numReac() const;

		static unsigned int getNenz( const Element* e );
		unsigned int numEnz() const;

		static double getDt( const Element* e );
		static void setDt( const Conn* c, double value );

		static unsigned int getSeed( const Element* e );
		static void setSeed( const Conn* c, unsigned int value );

		///////////////////////////////////////////////////
		// Functions to override zombie field access funcs.
		///////////////////////////////////////////////////
		static void setMolN( const Conn* c, double value );
		static double getMolN( const Element* e );
		static void setMolNinit( const Conn* c, double value );
		static double getMolNinit( const Element* e );
		static void setReacKf( const Conn* c, double value );
		static double getReacKf( const Element* e );
		static void setReacKb( const Conn* c, double value );
		static double getReacKb( const Element* e );

		static void setEnzK1( const Conn* c, double value );
		static double getEnzK1( const Element* e );
		static void setEnzK2( const Conn* c, double value );
		static double getEnzK2( const Element* e );
		static void setEnzK3( const Conn* c, double value );
		static double getEnzK3( const Element* e );
		static void setEnzKm( const Conn* c, double value );
		static double getEnzKm( const Element* e );
		static void setEnzKcat( const Conn* c, double value );
		static double getEnzKcat( const Element* e );

		static void setMmEnzK1( const Conn* c, double value );
		static double getMmEnzK1( const Element* e );
		static void setMmEnzK2( const Conn* c, double value );
		static double getMmEnzK2( const Element* e );
		static void setMmEnzK3( const Conn* c, double value );
		static void setMmEnzKm( const Conn* c, double value );
		static double getMmEnzKm( const Element* e );
		static void setMmEnzKcat( const Conn* c, double value );
		static double getMmEnzKcat( const Element* e );
		
		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////
		
		static void flux( const Conn* c, vector< double > influx );
		void handleEfflux( Element* hub, ProcInfo info ); // called from processFuncLocal

		
		static void reinitFunc( const Conn* c, ProcInfo info );
		void reinitFuncLocal( Element* e, ProcInfo info );
		static void processFunc( const Conn* c, ProcInfo info );
		void processFuncLocal( Element* e, ProcInfo info );

		static const Finfo* particleFinfo;

		static void molSum( const Conn* c, double val );

		///////////////////////////////////////////////////
		// Zombie setup functions
		///////////////////////////////////////////////////

		static void rateTermFunc( const Conn* c,
			vector< RateTerm* >* rates, 
			KinSparseMatrix* N,
			bool useHalfReacs );
		void localRateTermFunc( vector< RateTerm* >* rates,
			KinSparseMatrix* N );
		static void rateSizeFunc( const Conn* c,
			unsigned int nReac, unsigned int nEnz, 
			unsigned int nMmEnz);
		void rateSizeFuncLocal( Element* hub,
			unsigned int nReac, unsigned int nEnz, 
			unsigned int nMmEnz );
		static void molSizeFunc( const Conn* c,
			unsigned int nMol, unsigned int nBuf,
			unsigned int nSumTot );
		void molSizeFuncLocal(
			unsigned int nMol, unsigned int nBuf,
			unsigned int nSumTot );
		static void molConnectionFunc( const Conn* c,
				vector< double >* S,
				vector< double >* Sinit,
				vector< Element* >* elist
		);
		void molConnectionFuncLocal( Element* e,
				vector< double >* S,
				vector< double >* Sinit,
				vector< Element* >* elist
		);
		static void reacConnectionFunc( const Conn* c,
				unsigned int index, Element* reac );
		void reacConnectionFuncLocal( 
				Element* hub, 
				int rateTermIndex, 
				Element* reac );
		static void enzConnectionFunc( const Conn* c,
				unsigned int index, Element* enz );
		void enzConnectionFuncLocal(
				Element* hub, 
				int enzTermIndex, 
				Element* enz );
		static void mmEnzConnectionFunc( const Conn* c,
				unsigned int index, Element* mmEnz );
		void mmEnzConnectionFuncLocal(
				Element* hub, 
				int enzTermIndex, 
				Element* enz );

		static void clearFunc( const Conn* c );

		static void completeReacSetupFunc( const Conn* c, string s );
		void completeReacSetupLocal( const string& s );

		static void childFunc( const Conn* c, int stage );
		static void destroy( const Conn* c );
		static void zombify( Element* hub, Element* e, 
			const Finfo* hubFinfo, Finfo* solveFinfo );
		void findProducts( vector< unsigned int >& molIndex, 
				size_t reacIndex );

		static const Finfo* molSolveFinfo;

		// Inner function for parsing the path and assigning surfaces.
		void setSurfaces( const string& path );


	private:
		unsigned int nMol_; /// Number of molecules in model.
		unsigned int nBuf_; /// Number of buffered molecules in model.
		unsigned int nSumTot_; /// Number of sumtotalled molecules in model.

		/// A pointer to the entire Smoldyn data structure
		struct simstruct* simptr_;	

		/// Path of objects managed by Smoldyn
		string path_;
		map< Element*, unsigned int > molMap_;
		vector< unsigned int > molSumMap_;
		vector< unsigned int > reacMap_;
		vector< unsigned int > enzMap_;
		vector< unsigned int > mmEnzMap_;
		vector< RateTerm* >* rates_;
		vector< double >* S_;
		vector< double >* Sinit_;
		KinSparseMatrix* N_;
		double dt_;	// Timestep used by Smoldyn

		unsigned int seed_; // Random number seed used by Smoldyn
		static const double MINIMUM_DT;
};

// Used by the solver
extern const Cinfo* initSmoldynHubCinfo();

#endif // _SmoldynHub_h
