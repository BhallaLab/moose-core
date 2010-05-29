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


DataHandler* OneDimGlobalHandler::copy( unsigned int n, bool toGlobal )
	const
{
	if ( n > 1 ) {
		cout << Shell::myNode() << ": Error: OneDimGlobalHandler::copy: Cannot yet handle 2d arrays\n";
		exit( 0 );
	}

	if ( toGlobal ) {
		if ( n <= 1 ) { // Don't need to boost dimension.
			OneDimGlobalHandler* ret = new OneDimGlobalHandler( dinfo() );
			ret->size_ = size_;
			ret->reserve_ = reserve_;
			ret->data_ = dinfo()->copyData( data_, size_, 1 );
			return ret;
		} else {
			OneDimGlobalHandler* ret = new OneDimGlobalHandler( dinfo() );
			ret->setData( dinfo()->copyData( data_, size_, n ), size_ * n );
			return ret;
		}
	} else {
		if ( n <= 1 ) { // do copy only on node 0.
			OneDimHandler* ret = new OneDimHandler( dinfo() );
			ret->setNumData1( size_ );
			ret->setData( dinfo()->copyData( data_, size_, 1 ), size_ );
			return ret;
		} else {
			OneDimHandler* ret = new OneDimHandler( dinfo() );
			unsigned int size = ret->end() - ret->begin();
			if ( size > 0 ) {
				ret->setNumData1( size_ * size );
				ret->setData( dinfo()->copyData( data_, size_, n * size_ ), 
					size_ * size );
			}
			return ret;
		}
	}
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
	reserve_ = size_ = size;
	if ( data_ )
		dinfo()->destroyData( data_ );
	data_ = reinterpret_cast< char* >( 
		dinfo()->allocData( size ) );
}

/**
* Assigns the sizes of all array field entries at once.
* Ignore if 1 or 0 dimensions.
*/
void OneDimGlobalHandler::setNumData2( unsigned int start,
	const vector< unsigned int >& sizes )
{
	;
}

/**
 * Looks up the sizes of all array field entries at once.
 * Returns the start index.
 * Ignore in this case
 */
unsigned int OneDimGlobalHandler::getNumData2( 
	vector< unsigned int >& sizes ) const
{	
	return 0;
}

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

void OneDimGlobalHandler::allocate()
{
	if ( data_ )
		dinfo()->destroyData( data_ );
	data_ = reinterpret_cast< char* >( dinfo()->allocData( size_ ));
	reserve_ = size_;
}

unsigned int OneDimGlobalHandler::addOneEntry( const char* data )
{
	if ( size_ == reserve_ ) {
		reserve_ = size_ + 10;
		char* temp = dinfo()->allocData( reserve_ );
		if ( size_ > 0 ) {
			memcpy( temp, data_, size_ * dinfo()->size() );
			dinfo()->destroyData( data_ );
			data_ = temp;
		}
	}
	memcpy( data_ + size_ * dinfo()->size(), data, dinfo()->size() );
	++size_;
	return ( size_ - 1 );
}
