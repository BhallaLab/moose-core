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

using namespace std;

VectorTable::VectorTable() : xDivs_(0), xMin_(0), xMax_(0), invDx_(0),
														table_(0) 
{;}

//Implementation identical to that of HHGate::lookupTable.
double VectorTable::innerLookupTable( double x ) const
{
	if (x <= xMin_) 
		return table_[0];
	if (x >= xMax_)
		return table_.back();

	unsigned int index = ( x - xMin_ ) * invDx_;
	double frac = ( x - xMin_ - index / invDx_ ) * invDx_;
	return table_[ index ] * ( 1 - frac ) + table_[ index + 1 ] * frac;
}

vector< double > VectorTable::getTable() const
{
	if ( table_.size() == 0 )
	{
		cerr << "Warning : Table is empty\n";
	}

	return table_;
}

//Function to set up the lookup table. 
void VectorTable::setTable( vector< double > table, double xMin, double xMax )
{
	if (table.size() < 2)
	{
		cerr << "VectorTable : Error : Table must have at least two entries!\n"; 
		return;
	}

	if (table_.size() > 2)
		cout << "Warning : Current table will be erased\n";

	table_ = table;
	xMin_ = xMin;
	xMax_ = xMax; 
	xDivs_ = table.size() - 1;
	invDx_ = xDivs_ / (xMax - xMin);
}

unsigned int VectorTable::getDiv() const
{
	return xDivs_;
}

double VectorTable::getMin() const
{
	return xMin_;
}

double VectorTable::getMax() const
{
	return xMax_;
}

double VectorTable::getInvDx() const
{
	return invDx_;
}

#ifdef DO_UNIT_TESTS
void testVectorTable()
{
	VectorTable unInitedTable;	

	vector< double > data;
				
	double arr[11] = {0.0, 0.23, 0.41, 0.46, 0.42, 0.52, 0.49, 0.43, 0.38, 0.43, 0.44};

	data.reserve( 11 );
	//Filling up user table.
	for ( int i = 0; i < 11; i++ )	
		data.push_back( arr[i] );

	unInitedTable.setTable( data, -1.0, 1.0 ); 

	assert( unInitedTable.getDiv() == 10 );
	assert( doubleEq(unInitedTable.getMin(), -1.0) );
	assert( doubleEq(unInitedTable.getMax(), 1.0) );
	assert( doubleEq(unInitedTable.getInvDx(), 5) );

	assert( doubleEq(unInitedTable.innerLookupTable( 0.25 ), 0.475 ) );
	assert( doubleEq(unInitedTable.innerLookupTable( 0.47 ), 0.4125 ) );
	assert( doubleEq(unInitedTable.innerLookupTable( 1.5 ), 0.44 ) );
	assert( doubleEq(unInitedTable.innerLookupTable( -1.8 ), 0.0 ) );

	cout << "." << flush;
}
#endif
