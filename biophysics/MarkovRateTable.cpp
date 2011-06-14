/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "VectorTable.h"
#include "../builtins/Interpol2D.h"
#include "MarkovRateTable.h"

MarkovRateTable::MarkovRateTable() :
	size_(0)
{ ; }

MarkovRateTable::MarkovRateTable( unsigned int size ) : 
	size_(size)
{
	resize< VectorTable* >( vtTables_, size, 0 );
	resize< Interpol2D* >( int2dTables_, size, 0 );
}

vector< double > MarkovRateTable::getVtChildTable( unsigned int i, unsigned int j ) const
{
	vector< double > emptyVec;
	if ( isRateOneParam( i, j ) )
		return vtTables_[i][j]->getTable();
	else	
		cerr << "Error : No one parameter rate table set for this rate!\n";

	return emptyVec;
}

vector< vector< double > > MarkovRateTable::getInt2dChildTable( unsigned int i, unsigned int j ) const
{
	vector< vector< double > > emptyVec;

	if ( isRateTwoParam( i, j ) )
		return int2dTables_[i][j]->getTableVector();
	else	
		cerr << "Error : No two parameter rate table set for this rate!\n";

	return emptyVec;
}

void MarkovRateTable::setVtChildTable( vector< unsigned int > intParams, vector< double > doubleParams, vector< double > table, bool ligandFlag )
{
	if ( intParams.size() != 2 )  
	{
		cerr << "Error : Only two integer parameters must be present i.e. (i, j) where i,j are indices to the i,j'th lookup table.\n";
		return;
	}

	if ( doubleParams.size() != 2 ) 
	{
		cerr << "Error : Only two double parameters must be present i.e. (xmin, xmax).\n";
		return;
	}

	if ( table.size() == 0 )
	{
		cerr << "Error : Cannot set with empty table!.\n";
		return;
	}

	unsigned int i = intParams[0];
	unsigned int j = intParams[1];
	double xMin = doubleParams[0];
	double xMax = doubleParams[1];

	if ( areIndicesOutOfBounds( i, j ) )
	{
		cerr << "Error : Table requested is out of bounds!.\n";
		return; 
	}

	//If table isn't already initialized, do so.
	if ( vtTables_[i][j] == 0 )
		vtTables_[i][j] = new VectorTable();

	//Checking to see if this rate has already been set with a 2-parameter rate.
	if ( isRateTwoParam( i, j ) ) 
	{
		cerr << "This rate has already been set with a 2 parameter lookup table.\n";
		return;
	}

	vtTables_[i][j]->setTable( table, xMin, xMax );

	if ( ligandFlag )
		useLigandConc_[i][j] = true;

}

void MarkovRateTable::setInt2dChildTable( vector< unsigned int > intParams, vector< double > doubleParams, vector< vector< double > > table ) 
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

	if ( areIndicesOutOfBounds( i, j ) ) 
	{
		cerr << "Error : Table requested is out of bounds\n";
		return;
	}
	
	//If table isn't already initialized, do so.
	if ( int2dTables_[i][j] == 0 )
		int2dTables_[i][j] = new Interpol2D( xDivs, xMin, xMax, yDivs, yMin, yMax );

	//Checking to see if this rate has already been set with a one parameter rate
	//table.
	if ( vtTables_[i][j] != 0 ) 
	{
		cerr << "Error : This rate has already been set with a one parameter rate table!\n";
		return;
	}

	int2dTables_[i][j]->setTableVector( table ); 
}

double MarkovRateTable::lookup1D( unsigned int i, unsigned int j, double x )
{
	if ( areIndicesOutOfBounds( i, j ) ) 
		return 0;

	return vtTables_[i][j]->innerLookup( x );
}

double MarkovRateTable::lookup2D( unsigned int i, unsigned int j, double x, double y )
{
	if ( areIndicesOutOfBounds( i, j ) ) 
		return 0;

	return int2dTables_[i][j]->innerLookup( x, y );
}

bool MarkovRateTable::isRateZero( unsigned int i, unsigned int j ) const
{
	return ( vtTables_[i][j] == 0 && int2dTables_[i][j] == 0 );
}

bool MarkovRateTable::isRateConstant( unsigned int i, unsigned int j ) const
{
	if ( !isRateOneParam( i, j ) )
		return false;

	//No need to check if isRateTwoParam(i,j) is true as this condition is
	//impossible i.e. in such a case, the (i,j)'th rate contains pointers to both
	//a 1D and 2D lookup table. The setter functions for the child tables contain
	//guards to prevent this from occurring.
	return ( vtTables_[i][j]->getDiv() == 0 );
}

bool MarkovRateTable::isRateOneParam( unsigned int i, unsigned int j ) const
{
	return ( vtTables_[i][j] != 0 );
}

bool MarkovRateTable::isRateLigandDep( unsigned int i, unsigned int j ) const
{
	return ( isRateOneParam( i, j ) && useLigandConc_[i][j] ); 
}

bool MarkovRateTable::isRateTwoParam( unsigned int i, unsigned int j ) const
{
	return ( int2dTables_[i][j] != 0 );
}

bool MarkovRateTable::areIndicesOutOfBounds( unsigned int i, unsigned int j ) const
{
	return ( i > size_ || j > size_ );
}


