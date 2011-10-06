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

TwoDimHandler::TwoDimHandler( const DinfoBase* dinfo, bool isGlobal,
	unsigned int nx, unsigned int ny )
		: DataHandler( dinfo, isGlobal ), 
			nx_( nx ), ny_( ny ),
			start_( 0 ), end_( 0 )
{
	innerNodeBalance( nx * ny, Shell::myNode(), Shell::numNodes() );
	data_ = dinfo->allocData( end_ - start_ );
	/*
	double numBits = log( totalEntries_ ) / log( 2.0 );
	bitMask_ = ~( (~0) << static_cast< unsigned int >( ceil( numBits ) ) );
	*/
}

TwoDimHandler::TwoDimHandler( const TwoDimHandler* other )
	: DataHandler( other->dinfo(), other->isGlobal_ ), 
	  nx_( other->nx_ ),
	  ny_( other->ny_ ),
	  start_( other->start_ ),
	  end_( other->end_ )
{
	/*
	double numBits = log( totalEntries_ ) / log( 2.0 );
	bitMask_ = ~( (~0) << static_cast< unsigned int >( ceil( numBits ) ) );
	*/
	unsigned int num = end_ - start_;
	data_ = dinfo()->copyData( other->data_, num, num );
}

TwoDimHandler::~TwoDimHandler() {
	if ( data_ )
		dinfo()->destroyData( data_ );
}
////////////////////////////////////////////////////////////////////////
// Information functions
////////////////////////////////////////////////////////////////////////

char* TwoDimHandler::data( DataId index ) const
{
	// unsigned int i = index & bitMask_;
	return data_ + ( index.value() + start_ ) * dinfo()->size();
}

unsigned int TwoDimHandler::totalEntries() const
{
	return nx_ * ny_;
}

unsigned int TwoDimHandler::localEntries() const
{
	return end_ - start_;
}

unsigned int TwoDimHandler::sizeOfDim( unsigned int dim ) const
{
	if ( dim == 0 )
		return nx_;
	if ( dim == 1 )
		return ny_;
	return 0;
}

vector< unsigned int > TwoDimHandler::dims() const
{
	vector< unsigned int > ret( 2 );
	ret[0] = nx_;
	ret[1] = ny_;
	return ret;
}

/**
 * Returns true if the node decomposition has the data on the
 * current node
 */
bool TwoDimHandler::isDataHere( DataId index ) const {
	return ( isGlobal_ || 
		index == DataId::any || index == DataId::globalField ||
		( index.value() >= start_ && index.value() < end_ ) );
}

bool TwoDimHandler::isAllocated() const {
	return data_ != 0;
}

////////////////////////////////////////////////////////////////////////
// Load balancing
////////////////////////////////////////////////////////////////////////

bool TwoDimHandler::innerNodeBalance( unsigned int numData,
	unsigned int myNode, unsigned int numNodes )
{
	unsigned int totalEntries = nx_ * ny_;
	if ( isGlobal_ ) {
		start_ = 0;
		bool ret = ( totalEntries != numData );
		end_ = totalEntries = numData;
		return ret;
	} else {
		unsigned int oldNumData = totalEntries;
		unsigned int oldstart = start_;
		unsigned int oldend = end_;
		start_ = ( numData * myNode ) / numNodes;
		end_ = ( numData * ( 1 + myNode ) ) / numNodes;
		return ( numData != oldNumData || oldstart != start_ || 
			oldend != end_ );
	}
	// bitMask_ = ~( (~0) << static_cast< unsigned int >( ceil( numBits ) ) );

	// cout << "TwoDimHandler::innerNodeBalance( " << numData_ << ", " << start_ << ", " << end_ << "), fieldDimension = " << getFieldDimension() << "\n";

}

////////////////////////////////////////////////////////////////////////
// Process function
////////////////////////////////////////////////////////////////////////

/**
 * Handles both the thread and node decomposition
 */
void TwoDimHandler::process( const ProcInfo* p, Element* e, FuncId fid ) const
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

void TwoDimHandler::foreach( const OpFunc* f, Element* e, const Qinfo* q,
	const double* arg, unsigned int argIncrement ) const
{
	assert( q->threadNum() < threadStart_.size() );
	unsigned int end = threadStart_[ q->threadNum() + 1 ];
	for( unsigned int i = threadStart_[ q->threadNum() ]; i != end; ++i) {
		f->op( Eref( e, i ), q, arg );
		arg += argIncrement;
	}
}

////////////////////////////////////////////////////////////////////////
// Data Reallocation functions.
////////////////////////////////////////////////////////////////////////

void TwoDimHandler::globalize( const char* data, unsigned int numEntries )
{
	if ( isGlobal_ )
		return;
	isGlobal_ = true;

	assert( numEntries == nx_ * ny_ );

	dinfo()->destroyData( data_ );
	data_ = dinfo()->copyData( data, numEntries, numEntries );
	start_ = 0;
	end_ = numEntries;

}

void TwoDimHandler::unGlobalize()
{
	if ( !isGlobal_ ) return;
	isGlobal_ = false;

	if ( innerNodeBalance( nx_ * ny_, Shell::myNode(), 
		Shell::numNodes() ) ) {
		char* temp = data_;
		unsigned int n = end_ - start_;
		data_ = dinfo()->copyData( temp + start_ * dinfo()->size(), n, n );
		dinfo()->destroyData( temp );
	}
}

DataHandler* TwoDimHandler::copy( bool toGlobal, unsigned int n ) const
{
	if ( toGlobal ) {
		if ( !isGlobal() ) {
			cout << "Warning: TwoDimHandler::copy: Cannot copy from nonGlob    al to global\n";
			return 0;
		}
	}
	if ( n > 1 ) {
		vector< int > dims(3);
		dims[0] = nx_;
		dims[1] = ny_;
		dims[2] = n;
		AnyDimHandler* ret = new AnyDimHandler( dinfo(), toGlobal, dims );
		if ( data_ )  {
			ret->assign( data_, end_ - start_ );
		}
		return ret;
	} else {
		return new TwoDimHandler( this );
	}
	return 0;
}

DataHandler* TwoDimHandler::copyUsingNewDinfo( const DinfoBase* dinfo) const
{
	return new TwoDimHandler( dinfo, isGlobal_, nx_, ny_ );
}

/**
 * Resize if size has changed in any one of its dimensions, in this case
 * only dim zero. Does NOT alter # of dimensions.
 * In the best case, we would leave the old data alone. This isn't
 * possible if the data starts out as non-Global, as the index allocation
 * gets shuffled around. So I deal with it only in the isGlobal case.
 */
bool TwoDimHandler::resize( unsigned int dimension, unsigned int numEntries)
{
	if ( data_ != 0 && nx_ * ny_ > 0 && numEntries > 0 ) {
		if ( dimension == 0 ) {
			// go from 1 2 3 : 4 5 6 to 1 2 3 .. : 4 5 6 ..
			// Try to preserve original data, possible if it is global.
			if ( numEntries == nx_ )
				return 0;
			char* temp = data_;
			unsigned int oldNx = nx_;
			nx_ = numEntries;
			innerNodeBalance( nx_ * ny_, 
				Shell::myNode(), Shell::numNodes() );
			unsigned int newN = end_ - start_;
			data_ = dinfo()->allocData( newN );
			if ( isGlobal_ ) {
				unsigned int newBlockSize = nx_ * dinfo()->size();
				unsigned int oldBlockSize = oldNx * dinfo()->size();
				for ( unsigned int i = 0; i < ny_; ++i ) {
					memcpy( data_ + i * newBlockSize, temp + i * oldBlockSize, oldBlockSize );
				}
			} 
			dinfo()->destroyData( temp );
		} else if ( dimension == 1 ) {
			// go from 1 2 3 : 4 5 6 to 1 2 3 : 4 5 6 : 7 8 9 : ....
			// Try to preserve original data, possible if it is global.
			if ( numEntries == ny_ )
				return 0;
			char* temp = data_;
			unsigned int oldNy = ny_;
			ny_ = numEntries;
			innerNodeBalance( nx_ * ny_, 
				Shell::myNode(), Shell::numNodes() );
			unsigned int newN = end_ - start_;
			if ( isGlobal_ ) {
				assert( newN == nx_ * ny_ );
				data_ = dinfo()->copyData( temp, nx_ * oldNy, nx_ * ny_ );
			} else {
				data_ = dinfo()->allocData( newN );
			}
			dinfo()->destroyData( temp );
		}
	}
	return 0;
}

void TwoDimHandler::assign( const char* orig, unsigned int numOrig )
{
	if ( data_ && numOrig > 0 ) {
		if ( isGlobal() ) {
			dinfo()->assignData( data_, nx_ * ny_, orig, numOrig );
		} else {
			unsigned int n = end_ - start_;
			if ( numOrig >= end_ ) {
				dinfo()->assignData( data_, n, 
					orig + start_ * dinfo()->size(), n );
			} else {
				char* temp = dinfo()->
					copyData( orig, numOrig, nx_ * ny_ );
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

DataHandler::iterator TwoDimHandler::begin( ThreadId threadNum ) const
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

DataHandler::iterator TwoDimHandler::end( ThreadId threadNum ) const
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

void TwoDimHandler::rolloverIncrement( DataHandler::iterator* i ) const
{
	i->setData( i->data() + dinfo()->size() );
}

*/
