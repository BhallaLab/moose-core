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
#include "TableOfVectors.h"

using namespace std;

TableOfVectors::TableOfVectors() : parentTable_(0), parentTableSize_(0) 
{;}

TableOfVectors::TableOfVectors( unsigned int size ) : parentTableSize_(size)
{
	parentTable_.resize( size );

	for ( unsigned int i = 0; i < size; ++i )
		parentTable_[i].resize( size, 0 );
}

unsigned int TableOfVectors::getSize() const
{
	return parentTableSize_;
}

void TableOfVectors::setSize( unsigned int newTableSize )
{
	if (newTableSize < parentTableSize_)
	{
		cerr << "Cannot shrink this table!\n";
		return;
	}

	parentTableSize_ = newTableSize;
	parentTable_.resize( newTableSize );

	for ( unsigned int i = 0; i < newTableSize; ++i )
		parentTable_[i].resize( newTableSize, 0 );
}

VectorTable* TableOfVectors::getChildTable( unsigned int i, unsigned int j ) const
{
	if ( i < 0 || i > parentTableSize_ || 
			 j < 0 || j > parentTableSize_ )
	{
		cerr << "Error : Table requested is out of bounds!\n";
		return 0;
	}
	
	if (parentTable_[i][j] == 0)
	{
		cerr << "Error : This table has not been set!\n";
		return 0;
	}

	return parentTable_[i][j];
}

void TableOfVectors::setChildTable( vector< unsigned int > intParams, vector< double > doubleParams, vector< double > table )
{
	if ( intParams.size() != 2 ) 
	{
		cerr << "Error : Only two integer parameters must be present. Table not set.\n";
		return;
	}

	if ( doubleParams.size() != 2 ) 
	{
		cerr << "Error : Only two double parameters must be present. Table not set.\n";
		return;
	}

	unsigned int i = intParams[0];
	unsigned int j = intParams[1];
	double xMin = doubleParams[0];
	double xMax = doubleParams[1];

	if ( i > parentTableSize_ || j > parentTableSize_ )
	{
		cerr << "Error : Table requested is out of bounds!. Table not set.\n";
		return; 
	}

	if (table.size() == 0)
	{
		cerr << "Error : Cannot set with empty table!. Table not set.\n";
		return;
	}

	if (parentTable_[i][j] == 0)
		parentTable_[i][j] = new VectorTable();

	parentTable_[i][j]->setTable( table, xMin, xMax );
}

double TableOfVectors::lookupChildTable( unsigned int i , unsigned int j , double x ) const
{
	if ( i < 0 || i > parentTableSize_ || 
			 j < 0 || j > parentTableSize_ )
	{
		cerr << "Error : Table requested is out of bounds!\n";
		return 0;
	}

	if (parentTable_[i][j]->getDiv() == 0)
	{
		cerr << "Error : Table not set\n";
		return 0;			//Not quite sure what the appropriate value to return here would be.
	}

	return parentTable_[i][j]->innerLookupTable( x );
}

bool TableOfVectors::doesChildExist( unsigned int i, unsigned int j )
{
	if ( i < 0 || i > parentTableSize_ || 
			 j < 0 || j > parentTableSize_ )
	{
		cerr << "Error : Table requested is out of bounds!\n";
		return false;
	}

	return (parentTable_[i][j] != 0);
}

#ifdef DO_UNIT_TESTS
void testTableOfVectors()
{
	TableOfVectors motherTable(4);
	vector< double > childVec, doubleParams;			//Table that will be pointed to by motherTable[2][3].
	vector< unsigned int > intParams;
	VectorTable *child;
		
	double data[] = {0.31, 0.17, 0.05, 0.11, 0.23, 0.19, 0.30, 0.43, 0.67, 0.56, 0.51};
	
	childVec.reserve( 11 );

	for (int i = 0; i < 11; ++i)
		childVec.push_back( data[i] ); 	

	//REWRITE WITH INTPARAMS AND DOUBLEPARAMS
	intParams.push_back(2);
	intParams.push_back(3);
	doubleParams.push_back(-0.5);
	doubleParams.push_back(0.5);

	motherTable.setChildTable( intParams, doubleParams, childVec );
	
	child = motherTable.getChildTable( 2, 3 );
	assert( child->getTable().size() > 0 );	

	assert( doubleEq(motherTable.lookupChildTable( 2, 3, 0.34 ), 0.626 ) ) ; 
	assert( doubleEq(motherTable.lookupChildTable( 2, 3, 0.7 ), 0.51 ) ) ; 
	assert( doubleEq(motherTable.lookupChildTable( 2, 3, -1.5 ), 0.31 ) ) ; 

	cout << "." << flush;
}
#endif
