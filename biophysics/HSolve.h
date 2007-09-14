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

class HSolve: public HSolveBase
{
public:
	HSolve()
	:
		scanData_( structure_ ),
		scanElm_( 0 ),
		seed_( 0 )
	{ ; }
	
	static void setPath( const Conn& c, string path );
	static string getPath( const Element* e );
	static void setNDiv( const Conn& c, int NDiv );
	static int getNDiv( const Element* e );
	static void setVLo( const Conn& c, double VLo );
	static double getVLo( const Element* e );
	static void setVHi( const Conn& c, double VHi );
	static double getVHi( const Element* e );
	static void processFunc( const Conn& c, ProcInfo p );
	static void reinitFunc( const Conn& c, ProcInfo p );
	static void postCreateFunc( const Conn& c );
	static void scanTicksFunc( const Conn& c );
	
private:
	void innerSetPath( Element* e, const string& path );
	void innerReinitFunc( Element* e, const ProcInfo& p );
	void innerProcessFunc( );
	void innerPostCreateFunc( Element* e );
	void innerScanTicksFunc( );
	
	NeuroScan scanData_;
	Element* scanElm_;
	string path_;
	Element* seed_;
};

#endif // _HSOLVE_H
