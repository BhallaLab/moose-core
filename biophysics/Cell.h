/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Messaging Object Oriented Simulation Environment.
 **   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser General Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

/*
 * 24 December 2007
 * Updating Subhasis' Cell class to manage automatic solver creation.
 * Niraj Dudani
 */

/*******************************************************************
 * File:            Cell.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-02 13:38:29
 ********************************************************************/

#ifndef _CELL_H
#define _CELL_H

struct MethodInfo
{
	string description;
	bool isVariableDt;
	bool isImplicit;
	// May need other info here as well
};

class Cell
{
public:
	Cell();
	
	static void setMethod( const Conn* c, string value );
	static string getMethod( Eref e );
	
	// Some readonly fields with more info about the methods.
	static bool getVariableDt( Eref e );
	static bool getImplicit( Eref e );
	static string getDescription( Eref e );
	
	static void reinitFunc( const Conn* c, ProcInfo p );
	static void comptListFunc( const Conn* c,
		const vector< Element* >* clist );
	
	static void addMethod( const string& name, 
		const string& description,
		bool isVariableDt, bool isImplicit );
	
private:
	void innerReinitFunc( Id e, ProcInfo p );
	void innerSetMethod( string value );
	
	static Id findCompt( Id cell );
	void setupSolver( Id cell, Id seed, double dt ) const;
	void checkTree( ) const;
	
	string method_;
	bool implicit_;
	bool variableDt_;
	string description_;
    static map< string, MethodInfo >& methodMap();
};

#endif // _CELL_H
