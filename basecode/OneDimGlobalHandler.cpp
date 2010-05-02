/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

OneDimGlobalHandler::OneDimGlobalHandler( const DinfoBase* dinfo )
		: DataHandler( dinfo ), 
			data_( 0 ), size_( 0 )
{;}

OneDimGlobalHandler::~OneDimGlobalHandler() {
	dinfo()->destroyData( data_ );
}


/**
 * Handles both the thread and node decomposition
 * Here there is no node decomposition: all entries are present
 * on all nodes.
 */
void OneDimGlobalHandler::process( const ProcInfo* p, Element* e ) const
{
	char* temp = data_ + p->threadIndexInGroup * dinfo()->size();
	unsigned int stride = dinfo()->size() * p->numThreadsInGroup;

	for ( unsigned int i = p->threadIndexInGroup; i < size_; 
		i+= p->numThreadsInGroup )
	{
		reinterpret_cast< Data* >( temp )->process( p, Eref( e, i ) );
		temp += stride;
	}

/*


	unsigned int tbegin = p->threadIndexInGroup;

	unsigned int tbegin = numPerThread * p->threadIndexInGroup;
	unsigned int tend = tbegin + numPerThread;
	if ( tend > size_ ) 
		tend = size_;

	// for ( unsigned int i = 0; i != size_; ++i )
	for ( unsigned int i = tbegin; i != tend_; ++i )
	{
		reinterpret_cast< Data* >( temp )->process( p, Eref( e, i ) );
		temp += dinfo()->size();
	}
	*/
}


char* OneDimGlobalHandler::data( DataId index ) const {
	return data_ + index.data() * dinfo()->size();
}

char* OneDimGlobalHandler::data1( DataId index ) const {
	return data_ + index.data() * dinfo()->size();
}

/**
 * Assigns the size to use for the first (data) dimension
* If data is allocated, resizes that.
* If data is not allocated, does not touch it.
* For now: allocate it every time.
 */
void OneDimGlobalHandler::setNumData1( unsigned int size )
{
	size_ = size;
	if ( data_ )
		dinfo()->destroyData( data_ );
	data_ = reinterpret_cast< char* >( 
		dinfo()->allocData( size ) );
}

/**
* Assigns the sizes of all array field entries at once.
* Ignore if 1 or 0 dimensions.
*/
void OneDimGlobalHandler::setNumData2( const vector< unsigned int >& sizes )
{
	;
}

/**
 * Looks up the sizes of all array field entries at once.
 * Ignore in this case
 */
void OneDimGlobalHandler::getNumData2( vector< unsigned int >& sizes ) const
{;}

/**
 * Returns true if the node decomposition has the data on the
 * current node
 */
bool OneDimGlobalHandler::isDataHere( DataId index ) const {
	return 1;
}

bool OneDimGlobalHandler::isAllocated() const {
	return data_ != 0;
}

void OneDimGlobalHandler::allocate() {
	if ( data_ )
		dinfo()->destroyData( data_ );
	data_ = reinterpret_cast< char* >( dinfo()->allocData( size_ ));
}
