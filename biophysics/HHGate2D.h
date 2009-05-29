#ifndef _HHGate2D_h
#define _HHGate2D_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
class HHGate2D: public HHGate
{
	public:
		HHGate2D()
		{ ; }
		
		static double getAValue( Eref e, const vector< double >& v );
		static double getBValue( Eref e, const vector< double >& v );
		
		static void gateFunc( const Conn* c, double v1, double v2 );
		
		static void createInterpols( const Conn* c, IdGenerator idGen );
		
		static void setupAlpha( const Conn* c, vector< double > parms )
		{ cerr << "Error: HHGate2D: setupAlpha not implemented.\n"; }
		
		static void setupTau( const Conn* c, vector< double > parms )
		{ cerr << "Error: HHGate2D: setupTau not implemented.\n"; }
		
		static void tweakAlpha( const Conn* c )
		{ cerr << "Error: HHGate2D: tweakAlpha not implemented.\n"; }
		
		static void tweakTau( const Conn* c )
		{ cerr << "Error: HHGate2D: tweakTau not implemented.\n"; }
		
		static void setupGate( const Conn* c, vector< double > parms )
		{ cerr << "Error: HHGate2D: setupGate not implemented.\n"; }
		
	private:
		Interpol2D A_;
		Interpol2D B_;
};

// Used by solver, readcell, etc.
extern const Cinfo* initHHGate2DCinfo();

#endif // _HHGate2D_h
