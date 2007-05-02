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
		static void setK1( const Conn& c, double value );
		static double getK1( const Element* e );
		static void setK2( const Conn& c, double value );
		static double getK2( const Element* e );
		static void setK3( const Conn& c, double value );
		static double getK3( const Element* e );
		static double getKm( const Element* e );
		static void setKm( const Conn& c, double value );
		void innerSetKm( double value );
		static double getKcat( const Element* e );
		static void setKcat( const Conn& c, double value );
		void innerSetKcat( double value );
		static bool getMode( const Element* e );
		bool innerGetMode() const;
		static void setMode( const Conn& c, bool value );
		void innerSetMode( Element* e, bool mode );

		///////////////////////////////////////////////////
		// Shared message function definitions
		///////////////////////////////////////////////////
		static void processFunc( const Conn& c, ProcInfo p );
		void innerProcessFunc( Element* e );
		void implicitProcFunc( Element* e );
		void explicitProcFunc( Element* e );
		void innerReinitFunc(  );
		static void reinitFunc( const Conn& c, ProcInfo p );
		static void substrateFunc( const Conn& c, double n );
		static void enzymeFunc( const Conn& c, double n );
		static void complexFunc( const Conn& c, double n );
		
		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////
		static void intramolFunc( const Conn& c, double n );
		void innerIntramolFunc( double n );
		static void scaleKmFunc( const Conn& c, double k );
		void innerScaleKmFunc( double k );
		static void scaleKcatFunc( const Conn& c, double k );

		///////////////////////////////////////////////////////
		// Other func definitions
		///////////////////////////////////////////////////////
		void makeComplex( Element* e );
		
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
		void (Enzyme::*procFunc_ )( Element* e );
};
#endif // _Enzyme_h
