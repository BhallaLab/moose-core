/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

OneDimHandler::OneDimHandler( const DinfoBase* dinfo )
		: DataHandler( dinfo ), 
			data_( 0 ), size_( 0 ), 
			start_( 0 ), end_( 0 )
{;}

OneDimHandler::~OneDimHandler() {
	dinfo()->destroyData( data_ );
}


/**
 * Handles both the thread and node decomposition
 */
void OneDimHandler::process( const ProcInfo* p, Element* e ) const
{
	/**
	 * This is the variant with interleaved threads.
	char* temp = data_ + p->threadIndexInGroup * dinfo()->size();
	unsigned int stride = dinfo()->size() * p->numThreadsInGroup;
	for ( unsigned int i = start_ + p->threadIndexInGroup; i < end_;
		i += p->numThreadsInGroup ) {
		reinterpret_cast< Data* >( temp )->process( p, Eref( e, i ) );
		temp += stride;
	}
	 */

	/**
	 * This is the variant with threads in a block.
	 */
	unsigned int startIndex = start_ + 
		( ( end_ - start_ ) * p->threadIndexInGroup + 
		p->numThreadsInGroup - 1 ) /
			p->numThreadsInGroup;
	unsigned int endIndex = start_ + 
		( ( end_ - start_ ) * ( 1 + p->threadIndexInGroup ) +
		p->numThreadsInGroup - 1 ) /
			p->numThreadsInGroup;
	
	char* temp = data_ + ( startIndex - start_ ) * dinfo()->size();
	for ( unsigned int i = startIndex; i != endIndex; ++i ) {
		reinterpret_cast< Data* >( temp )->process( p, Eref( e, i ) );
		temp += dinfo()->size();
	}
}


char* OneDimHandler::data( DataId index ) const {
	if ( isDataHere( index ) )
		return data_ + ( index.data() - start_ ) * dinfo()->size();
	return 0;
}

char* OneDimHandler::data1( DataId index ) const {
	if ( isDataHere( index ) )
		return data_ + ( index.data() - start_ ) * dinfo()->size();
	return 0;
}

/**
 * Assigns the size to use for the first (data) dimension
* If data is allocated, resizes that.
* If data is not allocated, does not touch it.
* For now: allocate it every time.
 */
void OneDimHandler::setNumData1( unsigned int size )
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
void OneDimHandler::setNumData2( unsigned int start,
	const vector< unsigned int >& sizes )
{
	;
}

/**
 * Looks up the sizes of all array field entries at once.
 * Ignore in this case
 */
unsigned int OneDimHandler::getNumData2( vector< unsigned int >& sizes )
	const
{
	return 0;
}

/**
 * Returns true if the node decomposition has the data on the
 * current node
 */
bool OneDimHandler::isDataHere( DataId index ) const {
	return ( index.data() >= start_ && index.data() < end_ );
}

bool OneDimHandler::isAllocated() const {
	return data_ != 0;
}

void OneDimHandler::allocate() {
	if ( data_ )
		dinfo()->destroyData( data_ );
	data_ = reinterpret_cast< char* >( dinfo()->allocData( end_ - start_ ));
}
