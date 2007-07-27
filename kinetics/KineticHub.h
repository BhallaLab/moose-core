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
		static unsigned int getNmol( const Element* e );
		static unsigned int getNreac( const Element* e );
		static unsigned int getNenz( const Element* e );

		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////
		static void processFunc( const Conn& c, ProcInfo info );
		static void reinitFunc( const Conn& c, ProcInfo info );
		static void rateTermFunc( const Conn& c,
			vector< RateTerm* >* rates, bool useHalfReacs );
		static void rateSizeFunc( const Conn& c,
			unsigned int nReac, unsigned int nEnz, 
			unsigned int nMmEnz);
		void rateSizeFuncLocal( Element* hub,
			unsigned int nReac, unsigned int nEnz, 
			unsigned int nMmEnz );
		static void molSizeFunc( const Conn& c,
			unsigned int nMol, unsigned int nBuf,
			unsigned int nSumTot );
		void molSizeFuncLocal(
			unsigned int nMol, unsigned int nBuf,
			unsigned int nSumTot );
		static void molConnectionFunc( const Conn& c,
				vector< double >* S,
				vector< double >* Sinit,
				vector< Element* >* elist
		);
		void molConnectionFuncLocal( Element* e,
				vector< double >* S,
				vector< double >* Sinit,
				vector< Element* >* elist
		);
		static void reacConnectionFunc( const Conn& c,
				unsigned int index, Element* reac );
		void reacConnectionFuncLocal( 
				Element* hub, 
				int rateTermIndex, 
				Element* reac );
		static void enzConnectionFunc( const Conn& c,
				unsigned int index, Element* enz );
		void enzConnectionFuncLocal(
				Element* hub, 
				int enzTermIndex, 
				Element* enz );
		static void mmEnzConnectionFunc( const Conn& c,
				unsigned int index, Element* mmEnz );
		void mmEnzConnectionFuncLocal(
				Element* hub, 
				int enzTermIndex, 
				Element* enz );
		
		///////////////////////////////////////////////////
		// Overrides Neutral::destroy to clean up zombies.
		///////////////////////////////////////////////////
		static void destroy( const Conn& c);

		///////////////////////////////////////////////////
		// Functions to override zombie messages
		///////////////////////////////////////////////////
		static void molSum( const Conn& c, double val );

		///////////////////////////////////////////////////
		// Functions to override zombie field access funcs.
		///////////////////////////////////////////////////
		static void setMolN( const Conn& c, double value );
		static double getMolN( const Element* e );
		static void setMolNinit( const Conn& c, double value );
		static double getMolNinit( const Element* e );
		static void setReacKf( const Conn& c, double value );
		static double getReacKf( const Element* e );
		static void setReacKb( const Conn& c, double value );
		static double getReacKb( const Element* e );

		static void setEnzK1( const Conn& c, double value );
		static double getEnzK1( const Element* e );
		static void setEnzK2( const Conn& c, double value );
		static double getEnzK2( const Element* e );
		static void setEnzK3( const Conn& c, double value );
		static double getEnzK3( const Element* e );
		static void setEnzKm( const Conn& c, double value );
		static double getEnzKm( const Element* e );
		static void setEnzKcat( const Conn& c, double value );
		static double getEnzKcat( const Element* e );

		static void setMmEnzK1( const Conn& c, double value );
		static double getMmEnzK1( const Element* e );
		static void setMmEnzK2( const Conn& c, double value );
		static double getMmEnzK2( const Element* e );
		static void setMmEnzK3( const Conn& c, double value );
		static void setMmEnzKm( const Conn& c, double value );
		static double getMmEnzKm( const Element* e );
		static void setMmEnzKcat( const Conn& c, double value );
		static double getMmEnzKcat( const Element* e );


		static void zombify( 
			Element* hub, Element* e, const Finfo* hubFinfo,
	       		Finfo* solveFinfo );
	private:
		vector< double >* S_;
		vector< double >* Sinit_;
		vector< RateTerm* >* rates_;
		bool useHalfReacs_;
		bool rebuildFlag_;
		unsigned long nMol_;
		unsigned long nBuf_;
		unsigned long nSumTot_;
		vector< unsigned int > reacMap_;
		vector< unsigned int > enzMap_;
		vector< unsigned int > mmEnzMap_;
		vector< unsigned int > molSumMap_;
};
#endif // _KineticHub_h
