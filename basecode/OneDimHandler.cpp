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
	: OneDimGlobalHandler( other->dinfo() ), 
	  start_( other->start_ ),
	  end_( other->end_ )
{
    numData_ = other->numData_;
	unsigned int num = end_ - start_;
	data_ = dinfo()->copyData( other->data_, num, num );
}

DataHandler* OneDimHandler::globalize() const
{
	return 0; // Don't know yet how to do this.
}

DataHandler* OneDimHandler::unGlobalize() const
{
	return copy( 0 );
}


DataHandler* OneDimHandler::copy( bool toGlobal ) const
{
	assert( !toGlobal ); // Can't yet handle copies to globals.
	return ( new OneDimHandler( this ) );
}

DataHandler* OneDimHandler::copyUsingNewDinfo( const DinfoBase* dinfo) const
{
	OneDimHandler* ret = new OneDimHandler( dinfo );
	ret->start_ = start_;
	ret->end_ = end_;
	ret->data_ = dinfo->allocData( end_ - start_ );
	return ret;
}

DataHandler* OneDimHandler::copyExpand( 
	unsigned int copySize, bool toGlobal ) const
{
	assert( !toGlobal ); // Can't yet handle copies to globals.
	OneDimHandler* ret = new OneDimHandler( dinfo() );
	vector< unsigned int > dims( 1, copySize );
	ret->resize( dims );
	for ( iterator i = ret->begin(); i != ret->end(); ++i ) {
		char* temp = *i;
		memcpy( temp, data_, dinfo()->size() );
	}
	return ret;
}

DataHandler* OneDimHandler::copyToNewDim( 
	unsigned int newDimSize, bool toGlobal ) const
{
	assert( !toGlobal ); // Can't yet handle copies to globals.
	AnyDimHandler* ret = new AnyDimHandler( dinfo() );
	vector< unsigned int > dims( 2 );
	dims[1] = newDimSize;
	dims[0] = numData_;
	ret->resize( dims );

	for ( unsigned int i = 0; i < newDimSize; ++i ) {
		// setDataBlock( const char* data, unsigned int dataSize, unsigned int dimNum, unsigned int dimIndex )
		ret->setDataBlock( data_, numData_, i * numData_ );
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


unsigned int OneDimHandler::localEntries() const
{
	return end_ - start_;
}

char* OneDimHandler::data( DataId index ) const
{
	// Typically data will not be touched when index == any.
	if ( index == DataId::any() )
		return data_; 
	if ( isDataHere( index ) )
		return data_ + ( index.data() - start_ ) * dinfo()->size();
	return 0;
}

	/*
bool OneDimHandler::nodeBalance( unsigned int size )
{
	return nodeBalance( size, Shell::myNode(), Shell::numNodes() );
	unsigned int oldNumData = numData_;
	unsigned int oldstart = start_;
	unsigned int oldend = end_;
	numData_ = size;
	start_ = ( size * Shell::myNode() ) / Shell::numNodes();
	end_ = ( size * ( 1 + Shell::myNode() ) ) / Shell::numNodes();
	return ( size != oldNumData || oldstart != start_ || oldend != end_ );
}
	*/

bool OneDimHandler::innerNodeBalance( unsigned int numData,
	unsigned int myNode, unsigned int numNodes )
{
	unsigned int oldNumData = numData_;
	unsigned int oldstart = start_;
	unsigned int oldend = end_;
	numData_ = numData;
	start_ = ( numData * myNode ) / numNodes;
	end_ = ( numData * ( 1 + myNode ) ) / numNodes;

	// cout << "OneDimHandler::innerNodeBalance( " << numData_ << ", " << start_ << ", " << end_ << "), fieldDimension = " << getFieldDimension() << "\n";

	return ( numData != oldNumData || oldstart != start_ || oldend != end_ );
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
	if ( !data_ || numData_ == 0 ) {
		nodeBalance( dims[0] );
		if ( start_ < end_ )
		data_ = dinfo()->allocData( end_ - start_ );
		return 1;
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
	return ( index == DataId::any() || (
		index.data() >= start_ && index.data() < end_ ) );
}

bool OneDimHandler::isAllocated() const {
	return data_ != 0;
}

bool OneDimHandler::setDataBlock( 
	const char* data, unsigned int numData,
	const vector< unsigned int >& startIndex ) const
{
	if ( startIndex.size() > 1 ) return 0;
	unsigned int s = 0;

	if ( startIndex.size() == 1 )
		s = startIndex[0];
	
	return setDataBlock( data, numData, s );
}

bool OneDimHandler::setDataBlock( const char* data, unsigned int numData,
			DataId startIndex ) const
{
	if ( !isAllocated() ) return 0;

	// Check if the data block requested goes out of range.
	if ( startIndex.data() + numData > numData_ ) return 0;

	unsigned int actualStart = start_;
	if ( start_ < startIndex.data() ) 
		actualStart = startIndex.data();
	unsigned int actualEnd = end_;
	if ( actualEnd > startIndex.data() + numData )
		actualEnd = startIndex.data() + numData;
	if ( actualEnd > actualStart )
		memcpy( data_ + (actualStart - start_) * dinfo()->size(),
			data + ( actualStart - startIndex.data() ) * dinfo()->size(),
			( actualEnd - actualStart ) * dinfo()->size() );
	return 1;
}
