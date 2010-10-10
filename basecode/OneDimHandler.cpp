/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "AnyDimGlobalHandler.h"
#include "AnyDimHandler.h"

OneDimHandler::OneDimHandler( const DinfoBase* dinfo )
		: OneDimGlobalHandler( dinfo ), 
			start_( 0 ), end_( 0 )
{;}

OneDimHandler::OneDimHandler( const OneDimHandler* other )
	: OneDimGlobalHandler( other ), 
	  start_( other->start_ ),
	  end_( other->end_ )
{
	unsigned int num = end_ - start_;
	data_ = dinfo()->copyData( other->data_, num, num );
}

OneDimHandler::~OneDimHandler() {
	dinfo()->destroyData( data_ );
}

DataHandler* OneDimHandler::copy() const
{
	return ( new OneDimHandler( this ) );
}

DataHandler* OneDimHandler::copyExpand( unsigned int copySize ) const
{
	OneDimHandler* ret = new OneDimHandler( dinfo() );
	vector< unsigned int > dims( 1, copySize );
	ret->resize( dims );
	for ( iterator i = ret->begin(); i != ret->end(); ++i ) {
		char* temp = *i;
		memcpy( temp, data_, dinfo()->size() );
	}
	return ret;
}

DataHandler* OneDimHandler::copyToNewDim( unsigned int newDimSize ) const
{
	AnyDimHandler* ret = new AnyDimHandler( dinfo() );
	vector< unsigned int > dims( 2 );
	dims[1] = newDimSize;
	dims[0] = size_;
	ret->resize( dims );

	for ( unsigned int i = 0; i < newDimSize; ++i ) {
		// setDataBlock( const char* data, unsigned int dataSize, unsigned int dimNum, unsigned int dimIndex )
		ret->setDataBlock( data_, size_, 1, i );
	}
	return ret;
}

/**
 * Handles both the thread and node decomposition
 */
void OneDimHandler::process( const ProcInfo* p, Element* e, FuncId fid ) const
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


char* OneDimHandler::data( DataId index ) const {
	if ( isDataHere( index ) )
		return data_ + ( index.data() - start_ ) * dinfo()->size();
	return 0;
}

bool OneDimHandler::nodeBalance( unsigned int size )
{
	unsigned int oldsize = size_;
	size_ = size;
	unsigned int start = ( size * Shell::myNode() ) / Shell::numNodes();
	unsigned int end = ( size * ( 1 + Shell::myNode() ) ) / Shell::numNodes();
	return ( size != oldsize || start != start_ || end != end_ );
}


/**
 * Resize if size has changed. Can't handle change in #dimensions though.
 */
bool OneDimHandler::resize( vector< unsigned int > dims )
{
	if ( dims.size() != 1 ) {
		cout << "OneDimHandler::Resize: Warning: Attempt to resize wrong # of dims " << dims.size() << "\n";
		return 0;
	}
	if ( nodeBalance( dims[0] ) ) { // It has changed, so reallocate
		if ( data_ )
			dinfo()->destroyData( data_ );
		data_ = reinterpret_cast< char* >( 
			dinfo()->allocData( end_ - start_ ) );
	}
	return ( data_ != 0 );
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

/**
 * Data is an entire data block with numEntries. We pick whatever entries
 * should be put on current node. Return True if assignment happens.
 * Point of uncertainty: We return False even if assignment would happen
 * on some node other than this.
 */
bool OneDimHandler::setDataBlock( const char* data, 
	unsigned int numEntries, 
	unsigned int dimNum, unsigned int dimIndex )
{
	// Here the request is to set the whole array.
	if ( dimNum == 0 && dimIndex == 0 && numEntries == size_ ) {
		memcpy( data_, data + start_ * dinfo()->size(), 
			( end_ - start_ ) * dinfo()->size() );
		return 1;
	}

	// Here the request is to set an individual entry
	if ( dimNum == 0 && numEntries == 1 ) {
		if ( dimIndex >= start_ && dimIndex < end_ ) {
			memcpy( data_ + dimIndex * dinfo()->size(), data, 
				dinfo()->size() );
			return 1;
		}
	}
	return 0;
}
