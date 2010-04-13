/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

OneDimensionData::OneDimensionData( const DinfoBase* dinfo )
		: DataHandler( dinfo ), 
			data_( 0 ), size_( 0 ), 
			start_( 0 ), end_( 0 )
{;}

OneDimensionData::~OneDimensionData() {
	dinfo()->destroyData( data_ );
}

char* OneDimensionData::data( DataId index ) const {
	if ( isDataHere( index ) )
		return data_ + index.data() - start_;
	return 0;
}

char* OneDimensionData::data1( DataId index ) const {
	if ( isDataHere( index ) )
		return data_ + index.data() - start_;
	return 0;
}

/**
* Assigns the sizes of all array field entries at once.
* If data is allocated, resizes that.
* If data is not allocated, does not touch it.
*/
void OneDimensionData::setArraySizes( const vector< unsigned int >& sizes )
{
	assert( sizes.size() == 1 );
	size_ = sizes[0];
	unsigned int start =
		( size_ * Shell::myNode() ) / Shell::numNodes();
	unsigned int end = 
		( size_ * ( 1 + Shell::myNode() ) ) / Shell::numNodes();
	if ( data_ ) {
		if ( start == start_ && end == end_ ) // already done
			return;
		// Otherwise reallocate.
		dinfo()->destroyData( data_ );
		data_ = reinterpret_cast< char* >( 
			dinfo()->allocData( end - start ) );
	}
	start_ = start;
	end_ = end;
}

/**
 * Looks up the sizes of all array field entries at once. Returns
 * all ones for regular Elements. 
 * Note that a single Element may have more than one array field.
 * However, each FieldElement instance will refer to just one of
 * these array fields, so there is no ambiguity.
 */
void OneDimensionData::getArraySizes( vector< unsigned int >& sizes ) const {
	sizes.resize( 0 );
	sizes.push_back( size_ );
}

/**
 * Returns true if the node decomposition has the data on the
 * current node
 */
bool OneDimensionData::isDataHere( DataId index ) const {
	return ( index.data() >= start_ && index.data() < end_ );
}

bool OneDimensionData::isAllocated() const {
	return data_ != 0;
}

void OneDimensionData::allocate() {
	if ( data_ )
		dinfo()->destroyData( data_ );
	data_ = reinterpret_cast< char* >( dinfo()->allocData( end_ - start_ ));
}
