/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "DataDimensions.h"
#include "AnyDimGlobalHandler.h"
#include "AnyDimHandler.h"

AnyDimGlobalHandler::AnyDimGlobalHandler( const DinfoBase* dinfo )
	: DataHandler( dinfo ),
		data_( 0 ), size_( 0 )
{
	;
}

AnyDimGlobalHandler::AnyDimGlobalHandler( const AnyDimGlobalHandler* other )
	: DataHandler( other->dinfo() ),
		data_( other->dinfo()->copyData( other->data_, other->size_, other->size_ ) ),
		size_( other->size_ ),
		dims_( other->dims_ )
{
	;
}

AnyDimGlobalHandler::~AnyDimGlobalHandler()
{
	dinfo()->destroyData( data_ );
}

DataHandler* AnyDimGlobalHandler::globalize() const
{
	return copy();  // It is already global.
}

DataHandler* AnyDimGlobalHandler::unGlobalize() const
{
	AnyDimHandler* ret = new AnyDimHandler( dinfo() );
	// ret->nodeBalance( size_ );
	ret->resize( dims_ );
	ret->assimilateData( data_, 0, size_ );
	/*
	unsigned int numLocal = ret->end_ - ret->start_;
	char* newData = dinfo()->copyData( 
		data_ + ret->start_ * dinfo->size(), end_ - start_, end_ - start_ );
	dinfo()->destroyData( data_ );
	data_ = newData;
	isGlobal_ = 0;
	*/
	return ret;
}

/**
 * Determines how to decompose data among nodes for specified size
 * Returns true if there is a change from the current configuration
 */
bool AnyDimGlobalHandler::nodeBalance( unsigned int size )
{
	unsigned int oldsize = size_;
	size_ = size;
	return ( size != oldsize );
}

/**
 * For copy we won't worry about global status. 
 * Instead define function: globalize, which converts local data to global.
 * Version 1: Just copy as original
 */
DataHandler* AnyDimGlobalHandler::copy() const
{
	return new AnyDimGlobalHandler( this );
}


// Version 2: Copy same dimensions but different # of entries.
// The copySize is the total number of targets, here we need to figure out
// what belongs on the current node.
DataHandler* AnyDimGlobalHandler::copyExpand( unsigned int copySize ) const
{
	AnyDimGlobalHandler* ret = new AnyDimGlobalHandler( this );
	if ( ret->nodeBalance( copySize ) ) {
		dinfo()->destroyData( ret->data_ );
		ret->data_ = dinfo()->copyData( data_, copySize , copySize );
	}
	return ret;
}

DataHandler* AnyDimGlobalHandler::copyToNewDim( unsigned int newDimSize ) const
{
	AnyDimGlobalHandler* ret = new AnyDimGlobalHandler( this );

	ret->nodeBalance( size_ * newDimSize );
	ret->dims_.push_back( newDimSize );
	ret->data_ = dinfo()->copyData( data_, size_, 
		ret->size_ );
	return ret;
}

void AnyDimGlobalHandler::process( const ProcInfo* p, Element* e, FuncId fid ) 
	const
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
	unsigned int startIndex = 
		( ( size_ ) * p->threadIndexInGroup + 
		p->numThreadsInGroup - 1 ) /
			p->numThreadsInGroup;
	unsigned int endIndex = 
		( ( size_ ) * ( 1 + p->threadIndexInGroup ) +
		p->numThreadsInGroup - 1 ) /
			p->numThreadsInGroup;
	
	char* temp = data_ + startIndex * dinfo()->size();
	/*
	for ( unsigned int i = startIndex; i != endIndex; ++i ) {
		reinterpret_cast< Data* >( temp )->process( p, Eref( e, i ) );
		temp += dinfo()->size();
	}
	*/

	const OpFunc* f = e->cinfo()->getOpFunc( fid );
	const ProcOpFuncBase* pf = dynamic_cast< const ProcOpFuncBase* >( f );
	assert( pf );
	for ( unsigned int i = startIndex; i != endIndex; ++i ) {
		pf->proc( temp, Eref( e, i ), p );
		temp += dinfo()->size();
	}
}

char* AnyDimGlobalHandler::data( DataId index ) const
{
	if ( isDataHere( index ) )
		return data_ + index.data() * dinfo()->size();
	return 0;
}

unsigned int AnyDimGlobalHandler::totalEntries() const
{
	return size_;
}

unsigned int AnyDimGlobalHandler::numDimensions() const
{
	return dims_.size();
}

unsigned int AnyDimGlobalHandler::sizeOfDim( unsigned int dim ) const
{
	if ( dim < dims_.size() )
		return dims_[dim];
	return 0;
}

bool AnyDimGlobalHandler::resize( vector< unsigned int > dims )
{
	size_ = 1;
	for ( vector< unsigned int >::iterator i = dims.begin();
		i != dims.end(); ++i ) {
		size_ *= *i;
	}
	if ( nodeBalance( size_ ) ) { // It changed, reallocate
		if ( data_ )
			dinfo()->destroyData( data_ );
			data_ = reinterpret_cast< char* >( 
				dinfo()->allocData( size_ ) );
	}
	dims_ = dims;
	return ( data_ != 0 );
}

vector< unsigned int > AnyDimGlobalHandler::dims() const 
{
	return dims_;
}

bool AnyDimGlobalHandler::isDataHere( DataId index ) const {
	return ( index.data() >= 0 && index.data() < size_ );
}

bool AnyDimGlobalHandler::isAllocated() const
{
	return ( data_ != 0 );
}

bool AnyDimGlobalHandler::isGlobal() const
{
	// return ( Shell::numNodes() <= 1 || ( start_ == 0 && end_ == size_ ) );
	return 0;
}

AnyDimGlobalHandler::iterator AnyDimGlobalHandler::begin() const
{
	return iterator( this, 0 );
}

AnyDimGlobalHandler::iterator AnyDimGlobalHandler::end() const
{
	return iterator( this, size_ );
}

/**
 * Returns true if slice is legal. Passes back index of start of slice,
 * and size of slice.
bool AnyDimGlobalHandler::sliceInfo( 
	const vector< unsigned int >& slice,
	unsigned int& sliceStart, unsigned int& sliceSize )
{
	if ( slice.size() > dims_.size() )
		return 0;
	for ( unsigned int i = 1; i <= slice.size(); ++i ) {
		if ( slice[ slice.size() - i ] >= dims_[ dims.size() - i ] )
			return 0;
	}

	sliceSize = 1;
	vector< unsigned int > temp = slice;
	for ( unsigned int i = slice.size(); i < dims_.size(); ++i ) {
		slizeSize *= dims_[i];
		temp.push_back( 0 );
	}

	DataDimensions dd( dims_ );
	sliceStart = dd.linearIndex( temp );
}
 */

bool AnyDimGlobalHandler::setDataBlock( 
	const char* data, unsigned int numData,
	const vector< unsigned int >& startIndex )
{

	DataDimensions dd( dims_ );
	unsigned int start = dd.linearIndex( index );
	
	return setDataBlock( data, numData, start );
}

bool AnyDimGlobalHandler::setDataBlock( 
	const char* data, unsigned int numData,
	unsigned int startIndex )
{
	if ( size < size_ ) {
		memcpy( data_ + startIndex * dinfo()->size(), data, 
			numData * dinfo()->size() );
		return 1;
	}
	return 0;
}

/**
 * To set a single value in a 0-dim dataset: setDataBlock( data, 0, 1, 0, 0)
 * To set a single value at index 'i' in a 1-dim dataset: 
 *	setDataBlock( data, i, i+1, 0, 0) or
 *	setDataBlock( data, 0, 1, 0, i)
 *
 * To set a single value at index [i][j] in a 2-dim dataset:
 *	setDataBlock( data, j, j+1, 1, i) or
 *	setDataBlock( data, 0, 1, 0, i * numDim0 + j )
 * 
 * I don't like this.
 * 

bool AnyDimGlobalHandler::setDataBlock( const char* data,
	unsigned int begin, unsigned int end, 
	const vector< unsigned int >& slice )
{
	unsigned int numData = end - begin;
	unsigned int sliceStart;
	unsigned int sliceSize;
	if ( !sliceInfo( slice, sliceStart, sliceSize ) )
		return 0;
	assert( numData == sliceSize );

	if ( slice.size() == 0 && numData <= size_ ) {
		memcpy( data_ + begin * dinfo()->size(), data, 
			(end - begin ) * dinfo()->size() );
		return 1;
	}

	if ( dims_.size() == 0 )
		return 0;

	if ( dims_[ dims.size() -1 ] <= slice[0] ) // bad index on slice.
		return 0;

	unsigned int num = size_ / dims_[ dims.size() - 1 ];
	if ( slice.size() == 1 && numData <= num ) {
		memcpy( data_ + ( begin + num * slice_[0] ) * dinfo()->size(), 
			data, numData * dinfo()->size() );
		return 1;
	}

	// some checks here.

	num = size_ / dims_[ dims.size() - 1 ];
	if ( slice.size() == 2 && numData <= num ) {
		unsigned int num = size_ / dims_[ dims.size() - 1 ];
		memcpy( data_ + ( begin + num * slice_[0] ) * dinfo()->size(), 
			data, numData * dinfo()->size() );
		return 1;
	}


	

	assert( size_ != 0 );
	assert( isAllocated() );
	assert( end <= size_ );
	assert( dimNum == 0 || dimNum < dims_.size() );
	unsigned int dimSize = 1;
	for ( unsigned int i = 0; i < dims_.size() && i < dimNum; ++i ) {
		dimSize *= dims_[i];
	}
	assert( dimIndex < dims_[dimNum ] );
	assert( end <= dimSize );

	if ( begin < end ) {
		memcpy( data_ + begin * dinfo()->size(), data, 
			(end - begin ) * dinfo()->size() );
	}
}
 */


unsigned int AnyDimGlobalHandler::nextIndex( unsigned int index ) const
{
	return index + 1;
}
