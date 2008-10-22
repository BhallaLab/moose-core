/**********************************************************************
** This program is part of 'MOOSE', the
** Multiscale Object Oriented Simulation Environment.
**           copyright (C) 2003-2008
**           Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _DifShell_h
#define _DifShell_h

class DifShell
{
	public:
		DifShell();
		
		/////////////////////////////////////////////////////////////
		// Field access functions
		/////////////////////////////////////////////////////////////
		static double getC( Eref e );

		static void setCeq( const Conn* c, double Ceq );
		static double getCeq( Eref e );

		static void setD( const Conn* c, double D );
		static double getD( Eref e );

		static void setValence( const Conn* c, double valence );
		static double getValence( Eref e );

		static void setLeak( const Conn* c, double leak );
		static double getLeak( Eref e );

		static void setShapeMode( const Conn* c, unsigned int shapeMode );
		static unsigned int getShapeMode( Eref e );

		static void setLength( const Conn* c, double length );
		static double getLength( Eref e );

		static void setDiameter( const Conn* c, double diameter );
		static double getDiameter( Eref e );

		static void setThickness( const Conn* c, double thickness );
		static double getThickness( Eref e );

		static void setVolume( const Conn* c, double volume );
		static double getVolume( Eref e );

		static void setOuterArea( const Conn* c, double outerArea );
		static double getOuterArea( Eref e );

		static void setInnerArea( const Conn* c, double innerArea );
		static double getInnerArea( Eref e );

		/////////////////////////////////////////////////////////////
		// Dest functions
		/////////////////////////////////////////////////////////////
		static void reinit_0( const Conn* c, ProcInfo p );

		static void process_0( const Conn* c, ProcInfo p );

		static void process_1( const Conn* c, ProcInfo p );

		static void buffer(
			const Conn* c,
			double kf,
			double kb,
			double bFree,
			double bBound );

		static void fluxFromOut(
			const Conn* c,
			double outerC,
			double outerThickness );

		static void fluxFromIn(
			const Conn* c,
			double innerC,
			double innerThickness );

		static void influx(
			const Conn* c,
			double I );

		static void outflux(
			const Conn* c,
			double I );

		static void fInflux(
			const Conn* c,
			double I,
			double fraction );

		static void fOutflux(
			const Conn* c,
			double I,
			double fraction );

		static void storeInflux(
			const Conn* c,
			double flux );

		static void storeOutflux(
			const Conn* c,
			double flux );

		static void tauPump(
			const Conn* c,
			double kP,
			double Ceq );

		static void eqTauPump(
			const Conn* c,
			double kP );

		static void mmPump(
			const Conn* c,
			double vMax,
			double Kd );

		static void hillPump(
			const Conn* c,
			double vMax,
			double Kd,
			unsigned int hill );

	private:
		void localReinit_0( ProcInfo p );
		void localProcess_0( Eref difshell, ProcInfo p );
		void localProcess_1( ProcInfo p );
		void localBuffer( double kf, double kb, double bFree, double bBound );
		void localFluxFromOut( double outerC, double outerThickness );
		void localFluxFromIn( double innerC, double innerThickness );
		void localInflux(	double I );
		void localOutflux( double I );
		void localFInflux( double I, double fraction );
		void localFOutflux( double I, double fraction );
		void localStoreInflux( double flux );
		void localStoreOutflux( double flux );
		void localTauPump( double kP, double Ceq );
		void localEqTauPump( double kP );
		void localMMPump( double vMax, double Kd );
		void localHillPump( double vMax, double Kd, unsigned int hill );

		double dCbyDt_;
		double C_;
		double Ceq_;
		double D_;
		double valence_;
		double leak_;
		unsigned int shapeMode_;
		double length_;
		double diameter_;
		double thickness_;
		double volume_;
		double outerArea_;
		double innerArea_;
		
		/// Faraday's constant (Coulomb / Mole)
		static const double F_;
};

#endif // _DifShell_h
