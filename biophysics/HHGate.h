#ifndef _HHGate_h
#define _HHGate_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
class HHGate
{
	friend class HHGateWrapper;
	public:
		HHGate()
		{ ; }

		static void gateFunc(
				const Conn* c, double v );
		static void postCreate( const Conn* c );
		static void setupAlpha( const Conn* c, vector< double > parms );
		static void setupTau( const Conn* c, vector< double > parms );
		static void tweakAlpha( const Conn* c );
		static void tweakTau( const Conn* c );
		static void setupGate( const Conn* c, vector< double > parms );
		void setupTables( const vector< double >& parms, bool doTau );
		void tweakTables( bool doTau );
		void innerSetupGate( const vector< double >& parms );

	private:
		Interpol A_;
		Interpol B_;
};

// Used by solver, readcell, etc.
extern const Cinfo* initHHGateCinfo();

#endif // _HHGate_h
