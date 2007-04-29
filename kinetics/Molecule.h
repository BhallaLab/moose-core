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
		static double getMode( const Element* e );
		double localGetConc() const;
		static double getConc( const Element* e );
		void localSetConc( double value );
		static void setConc( const Conn& c, double value );
		double localGetConcInit() const;
		static double getConcInit( const Element* e );
		void localSetConcInit( double value );
		static void setConcInit( const Conn& c, double value );
		
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
		double nInit_;
		double volumeScale_;
		double n_;
		int mode_;
		double total_;
		double A_;
		double B_;
		static const double EPSILON;
};
#endif // _Molecule_h
