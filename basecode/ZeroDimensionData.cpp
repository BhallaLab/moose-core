/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

ZeroDimensionData::~ZeroDimensionData()
{
	dinfo()->destroyData( data_ );
}

void ZeroDimensionData::getArraySizes( vector< unsigned int >& sizes ) const
{
	sizes.resize( 0 );
	sizes.push_back( 1 );
}

/**
 * Returns true if the node decomposition has the data on the
 * current node
 */
bool ZeroDimensionData::isDataHere( DataId index ) const {
	return ( Shell::myNode() == 0 );
}

bool ZeroDimensionData::isAllocated() const {
	return data_ != 0;
}

void ZeroDimensionData::allocate() {
	if ( data_ ) 
		dinfo()->destroyData( data_ );
	data_ = reinterpret_cast< char* >( dinfo()->allocData( 1 ) );
}
