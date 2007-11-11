/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Molecule_h
#define _Molecule_h
class Molecule
{
	public:
		Molecule();
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////
		
		static void setNinit( const Conn& c, double value );
		static double getNinit( const Element* e );
		static void setVolumeScale( const Conn& c, double value );
		static double getVolumeScale( const Element* e );
		static void setN( const Conn& c, double value );
		static double getN( const Element* e );
		static void setMode( const Conn& c, int value );
		static int getMode( const Element* e );
		int localGetMode( const Element* e );
		double localGetConc() const;
		static double getConc( const Element* e );
		void localSetConc( double value );
		static void setConc( const Conn& c, double value );
		double localGetConcInit() const;
		static double getConcInit( const Element* e );
		void localSetConcInit( double value );
		static void setConcInit( const Conn& c, double value );
		static double getD( const Element* e );
		static void setD( const Conn& c, double value );
		
		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////
		
		static void reacFunc( const Conn& c, double A, double B );
		static void sumTotalFunc( const Conn& c, double n );
		void sumProcessFuncLocal( );
		static void sumProcessFunc( const Conn& c, ProcInfo info );
		static void reinitFunc( const Conn& c, ProcInfo info );
		void reinitFuncLocal( Element* e );
		static void processFunc( const Conn& c, ProcInfo info );
		void processFuncLocal( Element* e, ProcInfo info );

	private:
		/// Initial number of molecules: t=0 boundary condition.
		double nInit_;	
		/// Scale factor to go from n to conc. Deprecated. Should refer to geometry
		double volumeScale_; 
		double n_; /// Current number of molecules.
		int mode_; /// 0: Normal. 1: nSumTot. 2: concSumTot. 4: Buffered.
		double total_; /// State variable used for sumtotal calculations
		double A_;	/// Internal state variable
		double B_;	/// Internal state variable
		static const double EPSILON; /// Used for Exp Euler calculations
		double D_;	/// Diffusion constant
};

// Used by the solver
extern const Cinfo* initMoleculeCinfo();

#endif // _Molecule_h
