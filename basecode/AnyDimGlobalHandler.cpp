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

AnyDimGlobalHandler( const DinfoBase* dinfo )
	: DataHandler( dinfo ),
		data_( 0 ), size_( 0 )
{
	;
}

AnyDimGlobalHandler( const AnyDimGlobalHandler* other )
	: DataHandler( other->dinfo ),
		data_( 0 ), size_( other->size_ ),
		dims_( other->dims_ )
{
	;
}

AnyDimGlobalHandler::~AnyDimGlobalHandler()
{
	dinfo()->destroyData( data_ );
}

DataHandler* AnyDimGlobalHandler::globalize()
{
	return copy();  // It is already global.
}

DataHandler* unGlobalize()
{
	new AnyDimHandler* ret( dinfo() );
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

// Takes data from AnyDimHandlers across all nodes
// Requires that the Handler have been built and allocated already.
void AnyDimGlobalHandler::assimilateData( const char* data, 
	unsigned int begin, unsigned int end )
{
	assert( size_ != 0 );
	assert( allocated() );
	assert( end <= size_ );
	if ( begin < end ) {
		memcpy( data_ + begin * dinfo()->size(), data, 
			(end - begin ) * dinfo()->size() );
	}
}

/**
 * Determines how to decompose data among nodes for specified size
 * Returns true if there is a change from the current configuration
 */
bool AnyDimGlobalHandler::nodeBalance( unsigned int size )
{
	unsigned int oldsize = size_;
	size_ = size;
	unsigned int start =
		( size * Shell::myNode() ) / Shell::numNodes();
	unsigned int end = 
		( size * ( 1 + Shell::myNode() ) ) / Shell::numNodes();
	return ( size != oldsize || start != start_ || end != end_ );
}

/**
 * For copy we won't worry about global status. 
 * Instead define function: globalize, which converts local data to global.
 * Version 1: Just copy as original
 */
DataHandler* AnyDimGlobalHandler::copy() const
{
	AnyDimGlobalHandler* ret = new AnyDimGlobalHandler( this );
	unsigned int num = end_ - start_;
	ret->data_ = dinfo()->copyData( data_, num, num );
	return ret;
}


// Version 2: Copy same dimensions but different # of entries.
// The copySize is the total number of targets, here we need to figure out
// what belongs on the current node.
DataHandler* AnyDimGlobalHandler::copyExpand( unsigned int copySize ) const
{
	AnyDimGlobalHandler* ret = new AnyDimGlobalHandler( this );
	ret->nodeBalance( copySize );
	ret->data_ = dinfo()->copyData( data_, end_ - start_, 
		ret->end_ - ret->start_ );
	return ret;
}

DataHandler* AnyDimGlobalHandler::copyToNewDim( unsigned int newDimSize ) const
{
	AnyDimGlobalHandler* ret = new AnyDimGlobalHandler( this );

	ret->nodeBalance( size_ * newDimSize );
	ret->dims_.push_back( newDimSize );
	ret->data_ = dinfo()->copyData( data_, end_ - start_, 
		ret->end_ - ret->start_ );
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
	unsigned int startIndex = start_ + 
		( ( end_ - start_ ) * p->threadIndexInGroup + 
		p->numThreadsInGroup - 1 ) /
			p->numThreadsInGroup;
	unsigned int endIndex = start_ + 
		( ( end_ - start_ ) * ( 1 + p->threadIndexInGroup ) +
		p->numThreadsInGroup - 1 ) /
			p->numThreadsInGroup;
	
	char* temp = data_ + ( startIndex - start_ ) * dinfo()->size();
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
		return data_ + ( index.data() - start_ ) * dinfo()->size();
	return;
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

bool AnyDimGlobalHandler::resize( const vector< unsigned int >& dims )
{
	size_ = 1
	for ( vector< unsigned int >::iterator i = dims.begin();
		i != dims.end(); ++i ) {
		size_ *= *i;
	}
	if ( nodeBalance( size_ ) ) { // It changed, reallocate
		if ( data_ )
			dinfo()->destroyData( data_ );
			data_ = reinterpret_cast< char* >( 
				dinfo()->allocData( end_ - start_ ) );
	}
	dims_ = dims;
	return ( data_ != 0 );
}

const vector< unsigned int >& dims() const 
{
	return dims_;
}

bool AnyDimGlobalHandler::isDataHere( DataId index ) const {
	return ( index.data() >= start_ && index.data() < end_ );
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
	return iterator( this, start_ );
}

AnyDimGlobalHandler::iterator AnyDimGlobalHandler::end() const
{
	return iterator( this, end_ );
}

unsigned int AnyDimGlobalHandler::nextIndex( unsigned int index ) const
{
	return index + 1;
}
