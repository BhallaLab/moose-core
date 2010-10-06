/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "AnyDimHandler.h"

AnyDimHandler( const DinfoBase* dinfo )
	: DataHandler( dinfo ),
		data_( 0 ), size_( 0 ),
		start_( 0 ), end_( 0 )
{
	;
}

AnyDimHandler( const AnyDimHandler* other )
	: DataHandler( other->dinfo() ),
		data_( 0 ), size_( other->size_ ),
		start_( other->start_ ), end_( other->end_ ), dims_( other->dims_ )
{
	;
}

AnyDimHandler::~AnyDimHandler()
{
	dinfo()->destroyData( data_ );
}

// Sequence: The Shell tells all nodes to globalize. The local node
// immediately creates the data handler and assigns its size, and
// then sends out its contents to all other nodes. There needs to be
// a barrier here to guarantee that the input from other nodes doesn't
// come in till the Handler is created. Then the data arrives from other
// nodes and is filled in using assimilateData.
// 
DataHandler* AnyDimHandler::globalize()
{
	AnyDimGlobalHandler* ret = new AnyDimGlobalHandler( dinfo() );
	ret->resize( dims_ );
	ret->assimilateData( data_, start_, end_ );
	return ret; 
	// For now we just pass on this. Will need significant stuff to do
	// with Shell
}

DataHandler* unGlobalize()
{
	return copy(); // It is already unglobal. Do you want a copy?
	nodeBalance( size_ );
	char* newData = dinfo()->copyData( 
		data_ + start_ * dinfo->size(), end_ - start_, end_ - start_ );
	dinfo()->destroyData( data_ );
	data_ = newData;
	isGlobal_ = 0;
	return 1;
}

// Must get data from the AnyDimGlobalHandler
// Data handler must have already been built and allocated.
void AnyDimHandler::assimilateData( const char* data, 
	unsigned int begin, unsigned int end )
{
	assert( begin == 0 );
	assert( end == size_ );
	memcpy( data_, data + start_ * dinfo()->size(), 
		( end_ - start_ ) * dinfo()->size() );

	/*
	nodeBalance( end );
	data_ = dinfo()->copyData( 
		data + start_ * dinfo->size(), end_ - start_, end_ - start_ );
		*/
}

/*
void AnyDimHandler::packageData( PrepackedBuffer& buf ) const
{
	
}
*/


/**
 * Determines how to decompose data among nodes for specified size
 * Returns true if there is a change from the current configuration
 */
bool AnyDimHandler::nodeBalance( unsigned int size )
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
DataHandler* AnyDimHandler::copy() const
{
	AnyDimHandler* ret = new AnyDimHandler( this );
	unsigned int num = end_ - start_;
	ret->data_ = dinfo()->copyData( data_, num, num );
	return ret;
}


// Version 2: Copy same dimensions but different # of entries.
// The copySize is the total number of targets, here we need to figure out
// what belongs on the current node.
DataHandler* AnyDimHandler::copyExpand( unsigned int copySize ) const
{
	AnyDimHandler* ret = new AnyDimHandler( this );
	ret->nodeBalance( copySize );
	ret->data_ = dinfo()->copyData( data_, end_ - start_, 
		ret->end_ - ret->start_ );
	return ret;
}

DataHandler* AnyDimHandler::copyToNewDim( unsigned int newDimSize ) const
{
	AnyDimHandler* ret = new AnyDimHandler( this );

	ret->nodeBalance( size_ * newDimSize );
	ret->dims_.push_back( newDimSize );
	ret->data_ = dinfo()->copyData( data_, end_ - start_, 
		ret->end_ - ret->start_ );
	return ret;
}

void AnyDimHandler::process( const ProcInfo* p, Element* e, FuncId fid ) 
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

char* AnyDimHandler::data( DataId index ) const
{
	if ( isDataHere( index ) )
		return data_ + ( index.data() - start_ ) * dinfo()->size();
	return;
}

unsigned int AnyDimHandler::totalEntries() const
{
	return size_;
}

unsigned int AnyDimHandler::numDimensions() const
{
	return dims_.size();
}

unsigned int AnyDimHandler::sizeOfDim( unsigned int dim ) const
{
	if ( dim < dims_.size() )
		return dims_[dim];
	return 0;
}

bool AnyDimHandler::resize( vector< unsigned int > dims )
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
	return ( data_ != 0 );
}

vector< unsigned int > multiDimIndex( unsigned int index ) const
{
	return dims_.multiDimIndex( index );
}

unsigned int linearIndex( const vector< unsigned int >& index ) const
{
	return dims_.linearIndex( index );
}


bool AnyDimHandler::isDataHere( DataId index ) const {
	return ( index.data() >= start_ && index.data() < end_ );
}

bool AnyDimHandler::isAllocated() const
{
	return ( data_ != 0 );
}

bool AnyDimHandler::isGlobal() const
{
	// return ( Shell::numNodes() <= 1 || ( start_ == 0 && end_ == size_ ) );
	return 0;
}

AnyDimHandler::iterator AnyDimHandler::begin() const
{
	return iterator( this, start_ );
}

AnyDimHandler::iterator AnyDimHandler::end() const
{
	return iterator( this, end_ );
}

unsigned int AnyDimHandler::nextIndex( unsigned int index ) const
{
	return index + 1;
}
