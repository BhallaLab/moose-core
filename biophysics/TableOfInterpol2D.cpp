/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "../builtins/Interpol2D.h"
#include "TableOfInterpol2D.h"

TableOfInterpol2D::TableOfInterpol2D() : parentTable_(0), parentTableSize_(0) 
{;}

TableOfInterpol2D::TableOfInterpol2D( unsigned int size ) : parentTableSize_(size)
{
	parentTable_.resize( size );

	for ( unsigned int i = 0; i < size; ++i )
		parentTable_[i].resize( size );
}

unsigned int TableOfInterpol2D::getSize( )
{
	return parentTableSize_;
}

void TableOfInterpol2D::setSize( unsigned int parentTableSize )
{
	if ( parentTableSize < parentTableSize_ )
	{
		cerr << "Error : Cannot shrink table!. Table was not resized\n";
		return;
	}

	parentTableSize_ = parentTableSize;
	parentTable_.resize( parentTableSize_ );

	for ( unsigned int i = 0; i < parentTableSize_; ++i )
		parentTable_[i].resize( parentTableSize_, 0 );

}

Interpol2D* TableOfInterpol2D::getChildTable( unsigned int i, unsigned int j )
{
	if ( i > parentTableSize_ || j > parentTableSize_ )
	{
		cerr << "Error : Table requested is out of bounds\n";
		return 0;
	}

	if ( parentTable_[i][j] == 0 )
	{
		cerr << "Error : This table has not been set!\n";
		return 0;
	}

	return parentTable_[i][j];
}

void TableOfInterpol2D::setChildTable( vector< unsigned int > intParams, vector< double > doubleParams, vector< vector< double > > table ) 
{
	if ( intParams.size() != 4 )
	{
		cerr << "Error : Four integer parameters must be present -> i, j, xDivs, yDivs where i,j -> Indices to lookup table, xDivs, yDivs -> Number of X and Y divisions of lookup table\n";
		return;
	}

	if ( doubleParams.size() != 4 )
	{
		cerr << "Error : Four double parameters must be present -> xMin, xMax, yMin, yMax where xMin, xMax -> End points along X-direction and yMin, yMax -> End points along Y-direction.\n";
		return;
	}

	unsigned int i = intParams[0];
	unsigned int j = intParams[1];
	unsigned int xDivs = intParams[2];
	unsigned int yDivs = intParams[3];

	double xMin = doubleParams[0];
	double xMax = doubleParams[1];
	double yMin = doubleParams[2];
	double yMax = doubleParams[3];

	if ( i > parentTableSize_ || j > parentTableSize_ )
	{
		cerr << "Error : Table requested is out of bounds\n";
		return;
	}
	
	//If table isn't already initialized, do so.
	if (parentTable_[i][j] == 0)
		parentTable_[i][j] = new Interpol2D( xDivs, xMin, xMax, yDivs, yMin, yMax );

	parentTable_[i][j]->setTableVector( table ); 
}

double TableOfInterpol2D::lookupChildTable( unsigned int i, unsigned int j, double x, double y )
{
	if ( i > parentTableSize_ || j > parentTableSize_ )
	{
		cerr << "Error : Table requested is out of bounds\n";
		return 0;
	}

	return parentTable_[i][j]->innerLookup( x, y );
}

bool TableOfInterpol2D::doesChildExist( unsigned int i, unsigned int j )
{
	if ( i < 0 || i > parentTableSize_ || 
			 j < 0 || j > parentTableSize_ )
	{
		cerr << "Error : Table requested is out of bounds!\n";
		return false;
	}

	return (parentTable_[i][j] != 0);
}
