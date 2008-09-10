/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _HSOLVE_H
#define _HSOLVE_H

/**
 * HSolve adapts the integrator HSolveActive into a MOOSE class.
 */
class HSolve: public HSolveActive
{
public:
	HSolve()
	{ ; }
	
	static void processFunc( const Conn* c, ProcInfo p );
	static void setupFunc( const Conn* c, Id seed, double dt );
	
	static string getPath( Eref e );
	static void setCaAdvance( const Conn* c, int value );
	static int getCaAdvance( Eref e );
	static void setVDiv( const Conn* c, int vDiv );
	static int getVDiv( Eref e );
	static void setVMin( const Conn* c, double vMin );
	static double getVMin( Eref e );
	static void setVMax( const Conn* c, double vMax );
	static double getVMax( Eref e );
	static void setCaDiv( const Conn* c, int caDiv );
	static int getCaDiv( Eref e );
	static void setCaMin( const Conn* c, double caMin );
	static double getCaMin( Eref e );
	static void setCaMax( const Conn* c, double caMax );
	static double getCaMax( Eref e );
	
private:
	void setup( Eref integ, Id seed, double dt );
	void setupHub( Eref integ );
	
	string path_;
};

#endif // _HSOLVE_H
