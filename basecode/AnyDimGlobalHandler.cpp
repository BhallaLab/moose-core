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
		data_( 0 ), numData_( 0 )
{
	;
}

AnyDimGlobalHandler::AnyDimGlobalHandler( const AnyDimGlobalHandler* other )
	: DataHandler( other->dinfo() ),
		data_( other->dinfo()->copyData( other->data_, other->numData_, other->numData_ ) ),
		numData_( other->numData_ ),
		dims_( other->dims_ )
{
	;
}

AnyDimGlobalHandler::~AnyDimGlobalHandler()
{
	if ( data_ )
		dinfo()->destroyData( data_ );
}

DataHandler* AnyDimGlobalHandler::globalize() const
{
	return copy();  // It is already global.
}

DataHandler* AnyDimGlobalHandler::unGlobalize() const
{
	AnyDimHandler* ret = new AnyDimHandler( dinfo() );
	// ret->nodeBalance( numData_ );
	ret->resize( dims_ );
	ret->setDataBlock( data_, numData_, 0 );
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
bool AnyDimGlobalHandler::innerNodeBalance( unsigned int numData,
	unsigned int myNode, unsigned int numNodes )
{
	unsigned int oldNumData = numData_;
	numData_ = numData;
	return ( numData != oldNumData );
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

DataHandler* AnyDimGlobalHandler::copyUsingNewDinfo( 
	const DinfoBase* dinfo) const
{
	AnyDimGlobalHandler* ret = new AnyDimGlobalHandler( dinfo );
	ret->data_ = dinfo->allocData( numData_ );
	ret->numData_ = numData_;
	ret->dims_ = dims_;
	return ret;
}



/**
 * Version 2: Copy same dimensions but different # of entries.
 * This is not possible to do cleanly in multiple dimensions, as
 * any increase must be a multiple of one slice. Specifically, each
 * expansion must be a multiple of the size of all the dimensions except
 * the last (zeroth) one.
 * The system returns zero if this constraint fails.
 */
DataHandler* AnyDimGlobalHandler::copyExpand( unsigned int copySize ) const
{
	unsigned int quantum = 1;
	if ( dims_.size() > 1 ) {
		quantum = numData_ / dims_[0];
		if ( ( copySize % quantum ) != 0 )
			return 0;
	}
	AnyDimGlobalHandler* ret = new AnyDimGlobalHandler( this );
	vector< unsigned int > temp( dims_ );
	temp[0] = copySize / quantum;
	ret->resize( temp );
	unsigned int oldDim0 = dims_[0];
	unsigned int newDim0 = temp[0];

	/* The key formula is (using ints):
	Oldindex = 
		( newIndex % newDim0 ) % oldDim0 + oldDim0 * (newIndex / newDim0)
	
	Need to match ranges of old to new indices for memcpy. I neeed to
	advance newIndex by newDim0 at first level loop. Then I can copy 
	oldDim0 entries over in blocks till I complete newDim0.

	Does this work for multiple dims? Consider 2x3x4
	0	1	2	3				12	13	14	15
	4	5	6	7				16	17	18	19
	8	9	10	11				20	21	22	23

	Now extend to 2x3x5

	0	1	2	3	0			12	13	14	15	12
	4	5	6	7	4			16	17	18	19	16
	8	9	10	11	8			20	21	22	23	20

	Which has native indices

	0	1	2	3	4			15	16	17	18	19
	5	6	7	8	9			20	21	22	23	24
	10	11	12	13	14			25	26	27	28	29

	Let's try newindex 24. This gives 
	( 24 % 5 ) % 4 + 4 * ( 24 / 5 ) = 16. OK.

	Try newindex 9. This gives
	( 9 % 5 ) % 4 + 4 * ( 9 / 5 ) = 4. OK

	Try newindex 11. This gives
	( 11 % 5 ) % 4 + 4 * (11 / 5 ) = 9. OK

	Try newindex 29. This gives
	( 29 % 5 ) % 4 + 4 * (29 / 5 ) = 20. OK

	*/

	for ( unsigned int i = 0; i < copySize; i+= newDim0 ) {
		unsigned int s = oldDim0 * dinfo()->size();
		for ( unsigned int j = 0; j < newDim0; j += oldDim0 ) {
			if ( newDim0 - j < oldDim0 )
				s = ( newDim0 - j ) * dinfo()->size();
			memcpy( ret->data_ + ( i + j ) * dinfo()->size(),
				data_ + oldDim0 * ( i / newDim0 ) * dinfo()->size(), 
				s );
		}
	}
	return ret;
}

DataHandler* AnyDimGlobalHandler::copyToNewDim( unsigned int newDimSize ) const
{
	AnyDimGlobalHandler* ret = new AnyDimGlobalHandler( this );
	vector< unsigned int > newdims = dims_;
	newdims.push_back( newDimSize );
	ret->resize( newdims );
	// ret->nodeBalance( numData_ * newDimSize );
	unsigned int temp = numData_ * dinfo()->size();
	for ( unsigned int i = 0; i < newDimSize; ++i ) {
		memcpy( ret->data_ + i * temp, data_, temp );
	}
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
		( ( numData_ ) * p->threadIndexInGroup + 
		p->numThreadsInGroup - 1 ) /
			p->numThreadsInGroup;
	unsigned int endIndex = 
		( ( numData_ ) * ( 1 + p->threadIndexInGroup ) +
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
	return numData_;
}

unsigned int AnyDimGlobalHandler::localEntries() const
{
	return numData_;
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
	unsigned int numData = 1;
	for ( vector< unsigned int >::iterator i = dims.begin();
		i != dims.end(); ++i ) {
		numData *= *i;
	}
	if ( nodeBalance( numData ) ) { // It changed, reallocate
		if ( data_ )
			dinfo()->destroyData( data_ );

		data_ = reinterpret_cast< char* >( dinfo()->allocData( numData_ ) );
	}
	dims_ = dims;
	return ( data_ != 0 );
}

vector< unsigned int > AnyDimGlobalHandler::dims() const 
{
	return dims_;
}

bool AnyDimGlobalHandler::isDataHere( DataId index ) const {
	return ( index.data() >= 0 && index.data() < numData_ );
}

bool AnyDimGlobalHandler::isAllocated() const
{
	return ( data_ != 0 );
}

bool AnyDimGlobalHandler::isGlobal() const
{
	// return ( Shell::numNodes() <= 1 || ( start_ == 0 && end_ == numData_ ) );
	return 0;
}

AnyDimGlobalHandler::iterator AnyDimGlobalHandler::begin() const
{
	return iterator( this, 0, 0 );
}

AnyDimGlobalHandler::iterator AnyDimGlobalHandler::end() const
{
	return iterator( this, numData_, numData_ );
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
	const vector< unsigned int >& startIndex ) const
{

	DataDimensions dd( dims_ );
	unsigned int start = dd.linearIndex( startIndex );
	
	return setDataBlock( data, numData, start );
}

bool AnyDimGlobalHandler::setDataBlock( 
	const char* data, unsigned int numData,
	DataId startIndex ) const
{
	if ( startIndex.data() + numData > numData_ )
		return 0;
	memcpy( data_ + startIndex.data() * dinfo()->size(), data, 
		numData * dinfo()->size() );
	return 1;
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

	if ( slice.size() == 0 && numData <= numData_ ) {
		memcpy( data_ + begin * dinfo()->size(), data, 
			(end - begin ) * dinfo()->size() );
		return 1;
	}

	if ( dims_.size() == 0 )
		return 0;

	if ( dims_[ dims.size() -1 ] <= slice[0] ) // bad index on slice.
		return 0;

	unsigned int num = numData_ / dims_[ dims.size() - 1 ];
	if ( slice.size() == 1 && numData <= num ) {
		memcpy( data_ + ( begin + num * slice_[0] ) * dinfo()->size(), 
			data, numData * dinfo()->size() );
		return 1;
	}

	// some checks here.

	num = numData_ / dims_[ dims.size() - 1 ];
	if ( slice.size() == 2 && numData <= num ) {
		unsigned int num = numData_ / dims_[ dims.size() - 1 ];
		memcpy( data_ + ( begin + num * slice_[0] ) * dinfo()->size(), 
			data, numData * dinfo()->size() );
		return 1;
	}


	

	assert( numData_ != 0 );
	assert( isAllocated() );
	assert( end <= numData_ );
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


void AnyDimGlobalHandler::nextIndex( 
	DataId& index, unsigned int& linearIndex ) const
{
	index.incrementDataIndex();
	++linearIndex;
}
