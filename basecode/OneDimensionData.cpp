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
		return data_ + ( index.data() - start_ ) * dinfo()->size();
	return 0;
}

char* OneDimensionData::data1( DataId index ) const {
	if ( isDataHere( index ) )
		return data_ + index.data() - start_;
	return 0;
}

/**
 * Assigns the size to use for the first (data) dimension
* If data is allocated, resizes that.
* If data is not allocated, does not touch it.
* For now: allocate it every time.
 */
void OneDimensionData::setNumData1( unsigned int size )
{
	size_ = size;
	unsigned int start =
		( size_ * Shell::myNode() ) / Shell::numNodes();
	unsigned int end = 
		( size_ * ( 1 + Shell::myNode() ) ) / Shell::numNodes();
	// if ( data_ ) {
		if ( start == start_ && end == end_ ) // already done
			return;
		// Otherwise reallocate.
		if ( data_ )
			dinfo()->destroyData( data_ );
		data_ = reinterpret_cast< char* >( 
			dinfo()->allocData( end - start ) );
	// }
	start_ = start;
	end_ = end;
}

/**
* Assigns the sizes of all array field entries at once.
* Ignore if 1 or 0 dimensions.
*/
void OneDimensionData::setNumData2( const vector< unsigned int >& sizes )
{
	;
}

/**
 * Looks up the sizes of all array field entries at once.
 * Ignore in this case
 */
void OneDimensionData::getNumData2( vector< unsigned int >& sizes ) const
{;}

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
