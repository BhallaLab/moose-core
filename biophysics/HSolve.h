/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _HSOLVE_H
#define _HSOLVE_H

/**
 * HSolve adapts the integrator HSolveBase into a MOOSE class.
 */
class HSolve: public HSolveBase
{
public:
	HSolve()
	:
		scanData_( structure_ ),
		scanElm_( 0 )
	{ ; }
	
	static string getPath( const Element* e );
	static void setNDiv( const Conn& c, int NDiv );
	static int getNDiv( const Element* e );
	static void setVLo( const Conn& c, double VLo );
	static double getVLo( const Element* e );
	static void setVHi( const Conn& c, double VHi );
	static double getVHi( const Element* e );
	static void processFunc( const Conn& c, ProcInfo p );
	static void scanCreateFunc( const Conn& c );
	static void initFunc( const Conn& c, const Element* seed, double dt );
	
private:
	void innerProcessFunc( );
	void innerScanCreateFunc( Element* e );
	void innerInitFunc( Element* solve, const Element* seed, double dt );
	
	NeuroScan scanData_;
	Element* scanElm_;
	string path_;
};

#endif // _HSOLVE_H
