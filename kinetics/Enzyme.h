/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Enzyme_h
#define _Enzyme_h
class Enzyme
{
	friend class EnzymeWrapper;
	public:
		Enzyme();

		///////////////////////////////////////////////////
		// Field function definitions
		///////////////////////////////////////////////////
		static void setK1( const Conn* c, double value );
		static double getK1( Eref e );
		static void setK2( const Conn* c, double value );
		static double getK2( Eref e );
		static void setK3( const Conn* c, double value );
		static double getK3( Eref e );
		static double getKm( Eref e );
		double innerGetKm( Eref e );
		static void setKm( const Conn* c, double value );
		void innerSetKm( Eref e, double value );
		static double getKcat( Eref e );
		static void setKcat( const Conn* c, double value );
		void innerSetKcat( double value );
		static bool getMode( Eref e );
		bool innerGetMode() const;
		static void setMode( const Conn* c, bool value );
		void innerSetMode( Eref e, bool mode );
		static double getNinitComplex( Eref e );
		static void setNinitComplex( const Conn* c, double value );
		static double getConcInitComplex( Eref e );
		static void setConcInitComplex( const Conn* c, double value );
		static double getX( Eref e );
		static void setX( const Conn* c, double value );
		static double getY( Eref e );
		static void setY( const Conn* c, double value );
		static string getColor( Eref e );
		static void setColor( const Conn* c, string value );
		static string getBgColor( Eref e );
		static void setBgColor( const Conn* c, string value );
		
		///////////////////////////////////////////////////
		// Shared message function definitions
		///////////////////////////////////////////////////
		static void processFunc( const Conn* c, ProcInfo p );
		void innerProcessFunc( Eref e );
		void implicitProcFunc( Eref e );
		void explicitProcFunc( Eref e );
		void innerReinitFunc(  );
		static void reinitFunc( const Conn* c, ProcInfo p );
		static void substrateFunc( const Conn* c, double n );
		static void enzymeFunc( const Conn* c, double n );
		static void complexFunc( const Conn* c, double n );
		
		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////
		static void intramolFunc( const Conn* c, double n );
		void innerIntramolFunc( double n );
		static void scaleKmFunc( const Conn* c, double k );
		void innerScaleKmFunc( double k );
		static void scaleKcatFunc( const Conn* c, double k );
		static void rescaleRates( const Conn* c, double ratio );
		void innerRescaleRates( Eref e, double ratio );

		///////////////////////////////////////////////////////
		// Other func definitions
		///////////////////////////////////////////////////////
		void makeComplex( Eref e );
		
	private:
		double k1_;
		double k2_;
		double k3_;
		double sA_;
		double pA_;
		double eA_;
		double B_;
		double e_;
		double s_;
		double sk1_;	
		double Km_;
		void (Enzyme::*procFunc_ )( Eref e );
		double x_;		/// x coordinate for display
		double y_;		/// y coordinate for display
		string xtree_textfg_req_; //text color
		string xtree_fg_req_; //background color
};

// Used by the solver
extern const Cinfo* initEnzymeCinfo();

#endif // _Enzyme_h
