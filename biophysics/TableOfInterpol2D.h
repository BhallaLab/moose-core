#ifndef _TableOfInterpol2D_H
#define _TableOfInterpol2D_H
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

//Class : TableOfInterpol2D 
//Author : Vishaka Datta S, 2011, NCBS
//
//Implements a 2D table of 2D lookup tables, where the lookup tables are of type
//Interpol2D.
//This class is used when a rate of transition between two given states is a
//function of two parameters, usually ligand concentration and membrane voltage. 
//Its implementation and interface is identical to that of the TableOfVectors
//class.
//Owing to the commonality between both classes, might make sense to implement a
//base class first. 

class TableOfInterpol2D 
{
	public : 
	TableOfInterpol2D();

	TableOfInterpol2D(unsigned int);

	void setSize( unsigned int );
	unsigned int getSize( );

	Interpol2D* getChildTable( unsigned int, unsigned int );
	
	void setChildTable( vector< unsigned int >, vector< double >, vector< vector< double > > );   	
	
	double lookupChildTable( unsigned int, unsigned int, double, double );

	bool doesChildExist( unsigned int, unsigned int );

	private :
	vector< vector< Interpol2D* > > parentTable_;
	unsigned int parentTableSize_;
};

#endif
