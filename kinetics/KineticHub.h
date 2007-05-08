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
	friend class KineticHubWrapper;
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
		void rateSizeFuncLocal( 
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
		static void enzConnectionFunc( const Conn& c,
				unsigned int index, Element* enz );
		static void mmEnzConnectionFunc( const Conn& c,
				unsigned int index, Element* mmEnz );
		
		///////////////////////////////////////////////////
		// Functions to override zombie field access funcs.
		///////////////////////////////////////////////////
		static void setMolN( const Conn& c, double value );
		static double getMolN( const Element* e );
		static void setMolNinit( const Conn& c, double value );
		static double getMolNinit( const Element* e );

		static void zombify( 
			Element* hub, Element* e, 
			const Finfo* hubFinfo, Finfo* solveFinfo
		);
	private:
		vector< double >* S_;
		vector< double >* Sinit_;
		vector< RateTerm* >* rates_;
		bool useHalfReacs_;
		bool rebuildFlag_;
		unsigned long nMol_;
		unsigned long nBuf_;
		unsigned long nSumTot_;
};
#endif // _KineticHub_h
