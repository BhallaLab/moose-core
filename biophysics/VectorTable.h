#ifndef _VectorTable_H
#define _VectorTable_H
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

//Class : VectorTable
//Author : Vishaka Datta S, 2011, NCBS
//Extreme barebones implementation of a vector lookup table.
//This is a minimal 1D equivalent of the Interpol2D class. Provides simple
//functions for getting and setting up the table, along with a lookup function.
//All the parameters of the lookup table are read-only to avoid the hassle of
//re-initializing and refilling the table. Admittedly too restrictive, should be
//rewritten later on.

using namespace std;

class VectorTable 
{
	public : 
	VectorTable();

	double innerLookupTable( double ) const;

	//All members except table_ are read-only. Avoids the hassle of recomputing the table when one of the terms are changed. 
	vector< double > getTable() const;

	//Setting up the lookup table. 
	void setTable( vector< double >, double, double );

	unsigned int getDiv() const;	
	double getMin() const;
	double getMax() const;
	double getInvDx() const;

	private : 
	unsigned int xDivs_;
	double xMin_;
	double xMax_;
	double invDx_;

	vector< double > table_;
};

#endif
