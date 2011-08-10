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

AnyDimHandler::AnyDimHandler( const DinfoBase* dinfo )
	: AnyDimGlobalHandler( dinfo ),
		start_( 0 ), end_( 0 )
{
	;
}

AnyDimHandler::AnyDimHandler( const AnyDimHandler* other )
	: AnyDimGlobalHandler( other->dinfo() ),
		start_( other->start_ ), end_( other->end_ )
{
	unsigned int num = end_ - start_;
	data_ = dinfo()->copyData( data_, num, num );
}

// Sequence: The Shell tells all nodes to globalize. The local node
// immediately creates the data handler and assigns its size, and
// then sends out its contents to all other nodes. There needs to be
// a barrier here to guarantee that the input from other nodes doesn't
// come in till the Handler is created. Then the data arrives from other
// nodes and is filled in using assimilateData.
// 
DataHandler* AnyDimHandler::globalize() const
{
	/*
	AnyDimGlobalHandler* ret = new AnyDimGlobalHandler( dinfo() );
	ret->resize( dims_ );
	ret->assimilateData( data_, start_, end_ );
	return ret; 
	*/
	return 0;
	// For now we just pass on this. Will need significant stuff to do
	// with Shell
}

DataHandler* AnyDimHandler::unGlobalize() const
{
	return copy( 0 ); // It is already unglobal. Do you want a copy?
	/*
	nodeBalance( numData_ );
	char* newData = dinfo()->copyData( 
		data_ + start_ * dinfo->size(), end_ - start_, end_ - start_ );
	dinfo()->destroyData( data_ );
	data_ = newData;
	isGlobal_ = 0;
	return 1;
	*/
}

/**
 * Determines how to decompose data among nodes for specified size
 * Returns true if there is a change from the current configuration
 */
bool AnyDimHandler::innerNodeBalance( unsigned int numData,
	unsigned int myNode, unsigned int numNodes )
{
	unsigned int oldNumData = numData_;
	numData_ = numData;
	unsigned int start =
		( numData * myNode ) / numNodes;
	unsigned int end = 
		( numData * ( 1 + myNode ) ) / numNodes;
	return ( numData != oldNumData || start != start_ || end != end_ );
}

/**
 * For copy we won't worry about global status. 
 * Instead define function: globalize, which converts local data to global.
 * Version 1: Just copy as original
 */
DataHandler* AnyDimHandler::copy( bool toGlobal ) const
{
	assert( !toGlobal ); // Cannot yet handle globalization
	AnyDimHandler* ret = new AnyDimHandler( this );
	return ret;
}

DataHandler* AnyDimHandler::copyUsingNewDinfo( const DinfoBase* dinfo) const
{
	AnyDimHandler* ret = new AnyDimHandler( dinfo );
	ret->start_ = start_;
	ret->end_ = end_;
	ret->data_ = dinfo->allocData( end_ - start_ );
	return ret;
}


// Version 2: Copy same dimensions but different # of entries.
// The copySize is the total number of targets, here we need to figure out
// what belongs on the current node.
DataHandler* AnyDimHandler::copyExpand( 
	unsigned int copySize, bool toGlobal ) const
{
	assert( !toGlobal ); // Cannot yet handle copies to globals.
	AnyDimHandler* ret = new AnyDimHandler( this );
	ret->nodeBalance( copySize );
	ret->data_ = dinfo()->copyData( data_, end_ - start_, 
		ret->end_ - ret->start_ );
	return ret;
}

DataHandler* AnyDimHandler::copyToNewDim(
	unsigned int newDimSize, bool toGlobal ) const
{
	assert( !toGlobal ); // Cannot yet handle copies to globals.
	AnyDimHandler* ret = new AnyDimHandler( this );

	ret->nodeBalance( numData_ * newDimSize );
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
	unsigned int startIndex = start_;
	unsigned int endIndex = end_;
	if ( p->numThreadsInGroup > 1 ) {
		// Note that threadIndexInGroup is indexed from 1 up.
		assert( p->threadIndexInGroup >= 1 );
		startIndex =
			start_ + 
			( ( end_ - start_ ) * ( p->threadIndexInGroup - 1 ) + 
			p->numThreadsInGroup - 1 ) /
				p->numThreadsInGroup;
		endIndex = 
			start_ + 
			( ( end_ - start_ ) * p->threadIndexInGroup +
			p->numThreadsInGroup - 1 ) /
				p->numThreadsInGroup;
	}
	
	assert( startIndex >= start_ && startIndex <= end_ );
	assert( endIndex >= start_ && endIndex < end_ );
	char* temp = data_ + ( startIndex - start_ ) * dinfo()->size();

	const OpFunc* f = e->cinfo()->getOpFunc( fid );
	const ProcOpFuncBase* pf = dynamic_cast< const ProcOpFuncBase* >( f );
	assert( pf );
	for ( unsigned int i = startIndex; i != endIndex; ++i ) {
		pf->proc( temp, Eref( e, i ), p );
		temp += dinfo()->size();
	}
}

unsigned int AnyDimHandler::localEntries() const
{
	return end_ - start_;
}

char* AnyDimHandler::data( DataId index ) const
{
	// Typically data will not be touched when index == any.
	if ( index == DataId::any() )
		return data_; 
	if ( isDataHere( index ) )
		return data_ + ( index.data() - start_ ) * dinfo()->size();
	return 0;
}

bool AnyDimHandler::resize( vector< unsigned int > dims )
{
	numData_ = 1;
	for ( vector< unsigned int >::iterator i = dims.begin();
		i != dims.end(); ++i ) {
		numData_ *= *i;
	}
	if ( nodeBalance( numData_ ) ) { // It changed, reallocate
		if ( data_ )
			dinfo()->destroyData( data_ );
			data_ = reinterpret_cast< char* >( 
				dinfo()->allocData( end_ - start_ ) );
	}
	dims_ = dims;
	return ( data_ != 0 );
}

bool AnyDimHandler::isDataHere( DataId index ) const {
	return ( 
		index == DataId::any() ||
		( index.data() >= start_ && index.data() < end_ )
	);
}

bool AnyDimHandler::isAllocated() const
{
	return ( data_ != 0 );
}

bool AnyDimHandler::isGlobal() const
{
	return 0;
}

DataHandler::iterator AnyDimHandler::begin() const
{
	return DataHandler::iterator( this, start_, start_ );
}

DataHandler::iterator AnyDimHandler::end() const
{
	return DataHandler::iterator( this, end_, end_ );
}

bool AnyDimHandler::setDataBlock( 
	const char* data, unsigned int numData,
	const vector< unsigned int >& startIndex ) const
{

	DataDimensions dd( dims_ );
	unsigned int start = dd.linearIndex( startIndex );
	
	return setDataBlock( data, numData, start );
}

bool AnyDimHandler::setDataBlock( const char* data, 
	unsigned int numEntries,
	DataId startIndex ) const
{
	if ( startIndex.data() + numEntries > totalEntries() )
		return 0;
	unsigned int actualStart = start_;
	if ( start_ < startIndex.data() ) 
		actualStart = startIndex.data();
	unsigned int actualEnd = end_;
	if ( actualEnd > startIndex.data() + numEntries )
		actualEnd = startIndex.data() + numEntries;
	if ( actualEnd > actualStart )
		memcpy( data_ + (actualStart - start_) * dinfo()->size(),
			data + ( actualStart - startIndex.data() ) * dinfo()->size(),
			( actualEnd - actualStart ) * dinfo()->size() );
	return 1;
}

