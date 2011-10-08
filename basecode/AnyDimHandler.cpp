/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "../shell/Shell.h"

AnyDimHandler::AnyDimHandler( const DinfoBase* dinfo, bool isGlobal,
	const vector< int >& dims )
		: DataHandler( dinfo, isGlobal ), 
			start_( 0 ), end_( 0 )
{
	dims_.resize( dims.size() );
	totalEntries_ = 1;
	for ( unsigned int i = 0; i < dims.size(); ++i ) {
		assert( dims[i] > 0 );
		dims_[i] = dims[i];
		totalEntries_ *= dims_[i];
	}
	innerNodeBalance( totalEntries_, Shell::myNode(), Shell::numNodes() );
	data_ = dinfo->allocData( end_ - start_ );
	/*
	double numBits = log( totalEntries_ ) / log( 2.0 );
	bitMask_ = ~( (~0) << static_cast< unsigned int >( ceil( numBits ) ) );
	*/
}

AnyDimHandler::AnyDimHandler( const AnyDimHandler* other )
	: DataHandler( other->dinfo(), other->isGlobal_ ), 
	  start_( other->start_ ),
	  end_( other->end_ ),
	  totalEntries_( other->totalEntries_ ),
	  dims_( other->dims_ )
{
	/*
	double numBits = log( totalEntries_ ) / log( 2.0 );
	bitMask_ = ~( (~0) << static_cast< unsigned int >( ceil( numBits ) ) );
	*/
	unsigned int num = end_ - start_;
	data_ = dinfo()->copyData( other->data_, num, num );
}

AnyDimHandler::~AnyDimHandler() {
	if ( data_ )
		dinfo()->destroyData( data_ );
}
////////////////////////////////////////////////////////////////////////
// Information functions
////////////////////////////////////////////////////////////////////////

char* AnyDimHandler::data( DataId index ) const
{
	// unsigned int i = index & bitMask_;
	return data_ + ( index.value() + start_ ) * dinfo()->size();
}

unsigned int AnyDimHandler::totalEntries() const
{
	return totalEntries_;
}

unsigned int AnyDimHandler::localEntries() const
{
	return end_ - start_;
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

vector< unsigned int > AnyDimHandler::dims() const
{
	return dims_;
}

/**
 * Returns true if the node decomposition has the data on the
 * current node
 */
bool AnyDimHandler::isDataHere( DataId index ) const {
	return ( isGlobal_ || 
		index == DataId::any || index == DataId::globalField ||
		( index.value() >= start_ && index.value() < end_ ) );
}

bool AnyDimHandler::isAllocated() const {
	return data_ != 0;
}

////////////////////////////////////////////////////////////////////////
// Load balancing
////////////////////////////////////////////////////////////////////////

bool AnyDimHandler::innerNodeBalance( unsigned int numData,
	unsigned int myNode, unsigned int numNodes )
{
	if ( isGlobal_ ) {
		if ( totalEntries_ == numData )
			return 0;
		start_ = 0;
		end_ = totalEntries_ = numData;
		return 1;
	} else {
		unsigned int oldNumData = totalEntries_;
		unsigned int oldstart = start_;
		unsigned int oldend = end_;
		totalEntries_ = numData;
		start_ = ( numData * myNode ) / numNodes;
		end_ = ( numData * ( 1 + myNode ) ) / numNodes;
		return ( numData != oldNumData || oldstart != start_ || 
			oldend != end_ );
	}
	// bitMask_ = ~( (~0) << static_cast< unsigned int >( ceil( numBits ) ) );

	// cout << "AnyDimHandler::innerNodeBalance( " << numData_ << ", " << start_ << ", " << end_ << "), fieldDimension = " << getFieldDimension() << "\n";

}

////////////////////////////////////////////////////////////////////////
// Process function
////////////////////////////////////////////////////////////////////////

/**
 * Handles both the thread and node decomposition
 */
void AnyDimHandler::process( const ProcInfo* p, Element* e, FuncId fid ) const
{
	/**
	 * This is the variant with threads in a block.
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
	*/
	
	unsigned int startIndex = threadStart_[ p->threadIndexInGroup ];
	unsigned int endIndex = threadStart_[ p->threadIndexInGroup +1 ];
	assert( startIndex >= start_ && startIndex <= end_ );
	assert( endIndex >= start_ && endIndex <= end_ );
	char* temp = data_ + ( startIndex - start_ ) * dinfo()->size();

	const OpFunc* f = e->cinfo()->getOpFunc( fid );
	const ProcOpFuncBase* pf = dynamic_cast< const ProcOpFuncBase* >( f );
	assert( pf );
	for ( unsigned int i = startIndex; i != endIndex; ++i ) {
		pf->proc( temp, Eref( e, i ), p );
		temp += dinfo()->size();
	}
}

void AnyDimHandler::foreach( const OpFunc* f, Element* e, const Qinfo* q,
	const double* arg, unsigned int argIncrement ) const
{
	assert( q->threadNum() < threadStart_.size() );
	unsigned int end = threadStart_[ q->threadNum() + 1 ];
	for( unsigned int i = threadStart_[ q->threadNum() ]; i != end; ++i) {
		f->op( Eref( e, i ), q, arg );
		arg += argIncrement;
	}
}

unsigned int AnyDimHandler::getAllData( vector< char* >& dataVec ) const
{
	dataVec.resize( 0 );
	char* temp = data_;
	for ( unsigned int i = start_; i < end_; ++i ) {
		dataVec.push_back( temp );
		temp += dinfo()->size();
	}
	return dataVec.size();
}

////////////////////////////////////////////////////////////////////////
// Data Reallocation functions.
////////////////////////////////////////////////////////////////////////

void AnyDimHandler::globalize( const char* data, unsigned int numEntries )
{
	if ( isGlobal_ )
		return;
	isGlobal_ = true;

	assert( numEntries == totalEntries_ );

	dinfo()->destroyData( data_ );
	data_ = dinfo()->copyData( data, numEntries, numEntries );
	start_ = 0;
	end_ = numEntries;

}

void AnyDimHandler::unGlobalize()
{
	if ( !isGlobal_ ) return;
	isGlobal_ = false;

	if ( innerNodeBalance( totalEntries_, Shell::myNode(), 
		Shell::numNodes() ) ) {
		char* temp = data_;
		unsigned int n = end_ - start_;
		data_ = dinfo()->copyData( temp + start_ * dinfo()->size(), n, n );
		dinfo()->destroyData( temp );
	}
}

DataHandler* AnyDimHandler::copy( bool toGlobal, unsigned int n ) const
{
	if ( toGlobal ) {
		if ( !isGlobal() ) {
			cout << "Warning: AnyDimHandler::copy: Cannot copy from nonGlob    al to global\n";
			return 0;
		}
	}
	if ( n > 1 ) {
		vector< int > dims( dims_.size() + 1);
		for ( unsigned int i = 0; i < dims_.size(); ++i )
			dims[i] = dims_[i];
		dims[ dims.size() ] = n;

		AnyDimHandler* ret = new AnyDimHandler( dinfo(), toGlobal, dims );
		if ( data_ )  {
			ret->assign( data_, end_ - start_ );
		}
		return ret;
	} else {
		return new AnyDimHandler( this );
	}
	return 0;
}

DataHandler* AnyDimHandler::copyUsingNewDinfo( const DinfoBase* dinfo) const
{
	vector< int > temp( dims_.size() );
	for ( unsigned int i = 0; i < dims_.size(); ++i ) {
		temp[i] = dims_[i];
	}
	return new AnyDimHandler( dinfo, isGlobal_, temp );
}

/**
 * Resize if size has changed in any one of its dimensions, in this case
 * only dim zero. Does NOT alter # of dimensions.
 * In the best case, we would leave the old data alone. This isn't
 * possible if the data starts out as non-Global, as the index allocation
 * gets shuffled around. So I deal with it only in the isGlobal case.
 * Returns 1 if the size has changed.
 */
bool AnyDimHandler::resize( unsigned int dimension, unsigned int numEntries)
{
	if ( dimension < dims_.size() && data_ != 0 && totalEntries_ > 0 &&
		numEntries > 0 ) {
		if ( dims_[ dimension ] == numEntries )
			return 0;
		unsigned int oldN = dims_[ dimension ];
		unsigned int oldTotalEntries = totalEntries_;
		dims_[ dimension ] = numEntries;
		totalEntries_ = 1;
		for ( unsigned int i = 0; i < dims_.size(); ++i ) {
			totalEntries_ *= dims_[i];
		}
		
		if ( dimension == 0 ) {
			// go from 1 2 3 : 4 5 6 to 1 2 3 .. : 4 5 6 ..
			// Try to preserve original data, possible if it is global.
			char* temp = data_;
			innerNodeBalance( totalEntries_, 
				Shell::myNode(), Shell::numNodes() );
			unsigned int newLocalEntries = end_ - start_;
			data_ = dinfo()->allocData( newLocalEntries );
			if ( isGlobal_ ) {
				unsigned int newBlockSize = dims_[0] * dinfo()->size();
				unsigned int oldBlockSize = oldN * dinfo()->size();
				unsigned int j = totalEntries_ / dims_[0];
				for ( unsigned int i = 0; i < j; ++i ) {
					memcpy( data_ + i * newBlockSize, temp + i * oldBlockSize, oldBlockSize );
				}
			} 
			dinfo()->destroyData( temp );
		} else {
			char* temp = data_;
			innerNodeBalance( totalEntries_, 
				Shell::myNode(), Shell::numNodes() );
			unsigned int newLocalEntries = end_ - start_;
			if ( isGlobal_ ) {
				assert( newLocalEntries == totalEntries_ );
				data_ = dinfo()->copyData( temp, oldTotalEntries,
					totalEntries_ );
			} else {
				data_ = dinfo()->allocData( newLocalEntries );
			}
			dinfo()->destroyData( temp );
		}
	}
	return 0;
}

void AnyDimHandler::assign( const char* orig, unsigned int numOrig )
{
	if ( data_ && numOrig > 0 ) {
		if ( isGlobal() ) {
			dinfo()->assignData( data_, totalEntries_, orig, numOrig );
		} else {
			unsigned int n = end_ - start_;
			if ( numOrig >= end_ ) {
				dinfo()->assignData( data_, n, 
					orig + start_ * dinfo()->size(), n );
			} else {
				char* temp = dinfo()->
					copyData( orig, numOrig, totalEntries_ );
				dinfo()->assignData( data_, n, 
					temp + start_ * dinfo()->size(), n );
				dinfo()->destroyData( temp );
			}
		}
	}
}

/*
////////////////////////////////////////////////////////////////////////
// Iterators
////////////////////////////////////////////////////////////////////////

DataHandler::iterator AnyDimHandler::begin( ThreadId threadNum ) const
{
	unsigned int startIndex = start_;
	if ( Shell::numProcessThreads() > 1 ) {
		// Note that threadNum is indexed from 1 up.
		assert( threadNum >= 1 );
		startIndex =
			start_ + 
			( ( end_ - start_ ) * ( threadNum - 1 ) + 
			Shell::numProcessThreads() - 1 ) /
				Shell::numProcessThreads();
	}
	
	assert( startIndex >= start_ && startIndex <= end_ );
	return iterator( this, startIndex );
	
}

DataHandler::iterator AnyDimHandler::end( ThreadId threadNum ) const
{
	unsigned int endIndex = end_;
	if ( Shell::numProcessThreads() > 1 ) {
		// Note that threadNum is indexed from 1 up.
		assert( threadNum >= 1 );
		endIndex = 
			start_ + 
			( ( end_ - start_ ) * threadNum +
			Shell::numProcessThreads() - 1 ) /
				Shell::numProcessThreads();
	}
	assert( endIndex >= start_ && endIndex <= end_ );
	return iterator( this, endIndex );
}

void AnyDimHandler::rolloverIncrement( DataHandler::iterator* i ) const
{
	i->setData( i->data() + dinfo()->size() );
}

*/
