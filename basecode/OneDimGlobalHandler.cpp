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

OneDimGlobalHandler::OneDimGlobalHandler( const OneDimGlobalHandler* other )
		: DataHandler( other->dinfo() ), 
			size_( other->size_ )
{
	data_ = dinfo()->copyData( other->data_, size_, size_ );
}

OneDimGlobalHandler::~OneDimGlobalHandler() {
	dinfo()->destroyData( data_ );
}

DataHandler* OneDimGlobalHandler::copy() const
{
	return ( new OneDimGlobalHandler( this ) );
}

DataHandler* OneDimGlobalHandler::copyExpand( unsigned int copySize ) const
{
	OneDimGlobalHandler* ret = new OneDimGlobalHandler( dinfo() );
	vector< unsigned int > dims( 1, copySize );
	ret->resize( dims );
	unsigned int s = size_ * dinfo()->size();
	for ( unsigned int offset = 0; offset < copySize; offset += size_ ) {
		if ( s > ( copySize - offset ) )
			s = copySize - offset;
		memcpy( ret->data_ + offset, data_, s * dinfo()->size() );
	}
	return ret;
}

/**
 * Expand it into a 2-dimensional version of AnyDimGlobalHandler.
 */
DataHandler* OneDimGlobalHandler::copyToNewDim( unsigned int newDimSize ) 
	const
{
	AnyDimGlobalHandler* ret = new AnyDimGlobalHandler( dinfo() );
	vector< unsigned int > dims( 2 );
	dims[1] = newDimSize;
	dims[0] = size_
	ret->resize( dims );

	for ( unsigned int i = 0; i < wq



	for ( iterator i = ret->begin(); i != ret->end(); ++i ) {
		char* temp = *i;
		memcpy( temp, data_, dinfo()->size() );
	}
	return ret;
}

/**
 * Handles both the thread and node decomposition
 * Here there is no node decomposition: all entries are present
 * on all nodes.
 */
void OneDimGlobalHandler::process( const ProcInfo* p, Element* e, 
	FuncId fid ) const
{
	char* temp = data_ + p->threadIndexInGroup * dinfo()->size();
	unsigned int stride = dinfo()->size() * p->numThreadsInGroup;

	const OpFunc* f = e->cinfo()->getOpFunc( fid );
	const ProcOpFuncBase* pf = dynamic_cast< const ProcOpFuncBase* >( f );
	assert( pf );

	for ( unsigned int i = p->threadIndexInGroup; i < size_; 
		i+= p->numThreadsInGroup )
	{
		// reinterpret_cast< Data* >( temp )->process( p, Eref( e, i ) );
		pf->proc( temp, Eref( e, i ), p );
		temp += stride;
	}
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
