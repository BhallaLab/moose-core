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

OneDimHandler::OneDimHandler( const DinfoBase* dinfo, bool isGlobal,
	unsigned int size )
		: DataHandler( dinfo, isGlobal ), 
			totalEntries_( size ),
			start_( 0 ), end_( 0 )
{
	innerNodeBalance( size, Shell::myNode(), Shell::numNodes() );
	data_ = dinfo->allocData( end_ - start_ );
	/*
	double numBits = log( totalEntries_ ) / log( 2.0 );
	bitMask_ = ~( (~0) << static_cast< unsigned int >( ceil( numBits ) ) );
	*/
}

OneDimHandler::OneDimHandler( const OneDimHandler* other )
	: DataHandler( other->dinfo(), other->isGlobal_ ), 
	  start_( other->start_ ),
	  end_( other->end_ )
{
    totalEntries_ = other->totalEntries_;
	/*
	double numBits = log( totalEntries_ ) / log( 2.0 );
	bitMask_ = ~( (~0) << static_cast< unsigned int >( ceil( numBits ) ) );
	*/
	unsigned int num = end_ - start_;
	data_ = dinfo()->copyData( other->data_, num, num );
	innerNodeBalance( totalEntries_, Shell::myNode(), Shell::numNodes() );
}

OneDimHandler::~OneDimHandler() {
	if ( data_ )
		dinfo()->destroyData( data_ );
}
////////////////////////////////////////////////////////////////////////
// Information functions
////////////////////////////////////////////////////////////////////////

char* OneDimHandler::data( DataId index ) const
{
	// unsigned int i = index & bitMask_;
	return data_ + ( index.value() + start_ ) * dinfo()->size();
}

unsigned int OneDimHandler::totalEntries() const
{
	return totalEntries_;
}

unsigned int OneDimHandler::localEntries() const
{
	return end_ - start_;
}

unsigned int OneDimHandler::sizeOfDim( unsigned int dim ) const
{
	if ( dim == 0 )
		return totalEntries_;
	return 0;
}

vector< unsigned int > OneDimHandler::dims() const
{
	vector< unsigned int > ret( 1, totalEntries_ );
	return ret;
}

/**
 * Returns true if the node decomposition has the data on the
 * current node
 */
bool OneDimHandler::isDataHere( DataId index ) const {
	return ( isGlobal_ || 
		index == DataId::any || index == DataId::globalField ||
		( index.value() >= start_ && index.value() < end_ ) );
}

bool OneDimHandler::isAllocated() const {
	return data_ != 0;
}

////////////////////////////////////////////////////////////////////////
// Load balancing
////////////////////////////////////////////////////////////////////////

bool OneDimHandler::innerNodeBalance( unsigned int numData,
	unsigned int myNode, unsigned int numNodes )
{
	bool ret = 0;
	if ( isGlobal_ ) {
		start_ = 0;
		ret = ( totalEntries_ != numData );
		end_ = totalEntries_ = numData;
	} else {
		unsigned int oldNumData = totalEntries_;
		unsigned int oldstart = start_;
		unsigned int oldend = end_;
		totalEntries_ = numData;
		start_ = ( numData * myNode ) / numNodes;
		end_ = ( numData * ( 1 + myNode ) ) / numNodes;
		ret = ( numData != oldNumData || oldstart != start_ || 
			oldend != end_ );
	}
	if ( Shell::numProcessThreads() == 0 ) { // Single thread mode.
		threadStart_.resize( 2 );
		threadStart_[0] = start_;
		threadStart_[1] = end_;
	} else {
		threadStart_.resize( Shell::numProcessThreads() + 1 );
		for ( unsigned int i = 0; i <= Shell::numProcessThreads(); ++i ) {
			threadStart_[i] = start_ + 
			( ( end_ - start_ ) * i + Shell::numProcessThreads() - 1 ) /
				Shell::numProcessThreads();
		}
	}
	return ret;
	// cout << "OneDimHandler::innerNodeBalance( " << numData_ << ", " << start_ << ", " << end_ << "), fieldDimension = " << getFieldDimension() << "\n";

}

////////////////////////////////////////////////////////////////////////
// Process function
////////////////////////////////////////////////////////////////////////

/**
 * Handles both the thread and node decomposition
 */
void OneDimHandler::process( const ProcInfo* p, Element* e, FuncId fid ) const
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
	// unsigned int endIndex = end_;
	
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

void OneDimHandler:: foreach( const OpFunc* f, Element* e, const Qinfo* q,
			const double* arg, unsigned int argIncrement ) const
{
	assert( q->threadNum() < threadStart_.size() );
	unsigned int end = threadStart_[ q->threadNum() + 1 ];
	for( unsigned int i = threadStart_[ q->threadNum() ]; i != end; ++i) {
		f->op( Eref( e, i ), q, arg );
		arg += argIncrement;
	}
}

unsigned int OneDimHandler::getAllData( vector< char* >& dataVec ) const
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

void OneDimHandler::globalize( const char* data, unsigned int numEntries )
{
	if ( isGlobal_ )
		return;
	isGlobal_ = true;

	dinfo()->destroyData( data_ );
	data_ = dinfo()->copyData( data, numEntries, numEntries );
	start_ = 0;
	end_ = totalEntries_ = numEntries;

}

void OneDimHandler::unGlobalize()
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

DataHandler* OneDimHandler::copy( bool toGlobal, unsigned int n ) const
{
	if ( toGlobal ) {
		if ( !isGlobal() ) {
			cout << "Warning: OneDimHandler::copy: Cannot copy from nonGlob    al to global\n";
			return 0;
		}
	}
	if ( n > 1 ) { 
		// Note that we expand into ny, rather than nx. The current array
		// size is going to remain the lowest level index.
		TwoDimHandler* ret = new TwoDimHandler( dinfo(), toGlobal, n, totalEntries_ );
		if ( data_ )  {
			if ( isGlobal() ) {
				ret->assign( data_, totalEntries_ );
			} else {
				ret->assign( data_, end_ - start_ );
			}
		}
		return ret;
	} else {
		return new OneDimHandler( this );
	}
	return 0;
}

DataHandler* OneDimHandler::copyUsingNewDinfo( const DinfoBase* dinfo) const
{
	return new OneDimHandler( dinfo, isGlobal_, totalEntries_ );
}

/**
 * Resize if size has changed in any one of its dimensions, in this case
 * only dim zero. Does NOT alter # of dimensions.
 */
bool OneDimHandler::resize( unsigned int dimension, unsigned int numEntries)
{
	if ( data_ != 0 && dimension == 0 && 
		totalEntries_ > 0 && numEntries > 0 ) {
		if ( numEntries == totalEntries_ ) {
			return 1;
		}
		char* temp = data_;
		unsigned int n = end_ - start_;
		innerNodeBalance( numEntries, 
			Shell::myNode(), Shell::numNodes() );
		unsigned int newN = end_ - start_;
		data_ = dinfo()->copyData( temp, n, newN );
		return 1;
	}
	return 0;
}

void OneDimHandler::assign( const char* orig, unsigned int numOrig )
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

DataHandler::iterator OneDimHandler::begin( ThreadId threadNum ) const
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

DataHandler::iterator OneDimHandler::end( ThreadId threadNum ) const
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

void OneDimHandler::rolloverIncrement( DataHandler::iterator* i ) const
{
	// *i = iterator( this, end_ );
	i->setData( i->data() + dinfo()->size() );
}

*/
