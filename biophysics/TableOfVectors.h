#ifndef _TableOfVectors_H
#define _TableOfVectors_H
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

//Class :TableOfVectors 
//Author : Vishaka Datta S, 2011, NCBS.
//
//This class implements a 2D table of 1D lookup tables.
//The (i,j)'th entry in this table points to the lookup table of the rate of
//transition between states i and j.
//An interface is provided set and get entire lookup tables at any location in
//the table. A lookup function for each table is also provided.
//If a rate of transition is either zero (i.e. not possible) or constant, the
//corresponding entry in this table is set to NULL.
//
//Each lookup table is of type vectorTable, which is a wrapper around the vector<
//double > class.
using namespace std;

class TableOfVectors
{
	public : 
	TableOfVectors();

	TableOfVectors(unsigned int);

	//Size of the parent table
	unsigned int getSize() const;
	void setSize( unsigned int );

	VectorTable* getChildTable( unsigned int, unsigned int ) const;	
	void setChildTable( vector<unsigned int> intParams, vector< double > doubleParams, vector< double > );
	double lookupChildTable( unsigned int, unsigned int, double ) const;

	//Returns true if there is a lookup table at location (i,j).
	bool doesChildExist( unsigned int, unsigned int );

	private :
	vector< vector< VectorTable* > > parentTable_;
	unsigned int parentTableSize_;
};

#endif
