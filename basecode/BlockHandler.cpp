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

BlockHandler::BlockHandler( const DinfoBase* dinfo, 
	const vector< DimInfo >& dims,
	unsigned short pathDepth, bool isGlobal )
		: DataHandler( dinfo, dims, pathDepth, isGlobal ), 
			start_( 0 ), end_( 0 )
{
	assert( totalEntries_ > 0 );
	innerNodeBalance( totalEntries_, Shell::myNode(), Shell::numNodes() );
	data_ = dinfo->allocData( end_ - start_ );
}

BlockHandler::BlockHandler( const BlockHandler* other )
	: DataHandler( other->dinfo(), other->dims(), 
			other->pathDepth(),  other->isGlobal_ ), 
	  	start_( other->start_ ),
	  	end_( other->end_ )
{
	unsigned int num = end_ - start_;
	data_ = dinfo()->copyData( other->data_, num, num );
	innerNodeBalance( totalEntries_, Shell::myNode(), Shell::numNodes() );
}

BlockHandler::~BlockHandler() {
	if ( data_ )
		dinfo()->destroyData( data_ );
}
////////////////////////////////////////////////////////////////////////
// Information functions
////////////////////////////////////////////////////////////////////////

char* BlockHandler::data( DataId index ) const
{
	// unsigned int i = index & bitMask_;
	assert( index.value() >= start_ && index.value() < end_ );
	return data_ + ( index.value() - start_ ) * dinfo()->size();
}

unsigned int BlockHandler::totalEntries() const
{
	return totalEntries_;
}

unsigned int BlockHandler::localEntries() const
{
	return end_ - start_;
}

/**
 * Returns true if the node decomposition has the data on the
 * current node
 */
bool BlockHandler::isDataHere( DataId index ) const {
	return ( isGlobal_ || 
		index == DataId::any || index == DataId::globalField ||
		( index.value() >= start_ && index.value() < end_ ) );
}

bool BlockHandler::isAllocated() const {
	return data_ != 0;
}

unsigned int BlockHandler::linearIndex( DataId di ) const 
{
	return di.value();
}

vector< vector< unsigned int > > BlockHandler::pathIndices( DataId di ) 
	const
{
	/*
	unsigned int j = 0;
	vector< vector< unsigned int > > ret( pathDepth_ + 1 );
	unsigned long long index = di.value();

	for( unsigned int i = 0; i <= pathDepth_; ++i ) {
		if ( j < dims_.size() && dims_[j].depth == i ) {
			unsigned int q = index % dims_[j].size;
			ret[i].push_back( q );
			index /= dims_[j].size;
			j++
		}
	}
	return ret;
	*/

	vector< vector< unsigned int > > ret( pathDepth_ + 1 );
	int j = dims_.size() - 1;
	unsigned long long index = di.value();
	for ( unsigned int i = pathDepth_; i > 0;  --i ) {
		vector< unsigned int > temp;
		while ( j >= 0 && dims_[j].depth == i ) {
			unsigned int q = index % dims_[j].size;
			temp.push_back( q );
			index /= dims_[j].size;
			--j;
		}
		ret[i] = temp;
	}
	return ret;
}

DataId BlockHandler::pathDataId( 
	const vector< vector< unsigned int > >& indices) const
{
	if ( indices.size() != static_cast< unsigned int >( pathDepth_ ) + 1 )
		return DataId::bad;

	unsigned short depth = 0;
	unsigned long long linearIndex = 0;
	unsigned int j = 0;
	for ( unsigned int i = 0; i < dims_.size(); ++i ) {
		if ( depth != dims_[i].depth ) {
			j = 0;
			depth = dims_[i].depth;
		}
		assert( indices.size() > depth );
		assert( indices[ depth ].size() > j );
		linearIndex *= dims_[i].size;
		linearIndex += indices[depth][j++];
	}

	return DataId( linearIndex );
}

////////////////////////////////////////////////////////////////////////
// Load balancing
////////////////////////////////////////////////////////////////////////

bool BlockHandler::innerNodeBalance( unsigned int numData,
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
		threadEnd_.resize( 2 );
		threadStart_[0] = start_;
		threadEnd_[0] = end_;
		threadStart_[1] = start_;
		threadEnd_[1] = end_;
	} else {
		threadStart_.resize( Shell::numProcessThreads() + 1 );
		threadEnd_.resize( Shell::numProcessThreads() + 1 );
		threadStart_[0] = start_;
		threadEnd_[0] = end_;
		unsigned int n = end_ - start_;
		for ( unsigned int i = 0; i < Shell::numProcessThreads(); ++i ) {
			threadStart_[i + 1 ] = start_ + 
			( n * i + Shell::numProcessThreads() - 1 ) /
				Shell::numProcessThreads();

			threadEnd_[i + 1 ] = start_ + 
			( n * ( i + 1 ) + Shell::numProcessThreads() - 1 ) /
				Shell::numProcessThreads();
		}
	}
	return ret;
	// cout << "BlockHandler::innerNodeBalance( " << numData_ << ", " << start_ << ", " << end_ << "), fieldDimension = " << getFieldDimension() << "\n";

}

bool BlockHandler::execThread( ThreadId thread, DataId di ) const
{
	assert( thread < threadStart_.size() );
	assert( thread < threadEnd_.size() );
	return ( 
		( di == DataId::globalField && thread <= 1 ) ||
		( threadStart_[ thread ] <= di.value() && 
		di.value() < threadEnd_[ thread ] )
	);
}

////////////////////////////////////////////////////////////////////////
// Process function
////////////////////////////////////////////////////////////////////////

/**
 * Handles both the thread and node decomposition
 */
void BlockHandler::process( const ProcInfo* p, Element* e, FuncId fid ) const
{
	unsigned int startIndex = threadStart_[ p->threadIndexInGroup ];
	unsigned int endIndex = threadEnd_[ p->threadIndexInGroup ];
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

void BlockHandler:: foreach( const OpFunc* f, Element* e, const Qinfo* q,
	const double* arg, unsigned int argSize, unsigned int numArgs ) const
{
	assert( q->threadNum() < threadStart_.size() );
	unsigned int end = threadEnd_[ q->threadNum() ];
	unsigned int start = threadStart_[ q->threadNum() ];
	if ( numArgs <= 1 ) {
		for( unsigned int i = start; i != end; ++i)
			f->op( Eref( e, i ), q, arg );
	} else {
		unsigned int argOffset = argSize * start;
		unsigned int maxOffset = argSize * numArgs;
		for( unsigned int i = start; i != end; ++i) {
			f->op( Eref( e, i ), q, arg + argOffset );
			argOffset += argSize;
			if ( argOffset >= maxOffset )
				argOffset = 0;
		}
	}
}

unsigned int BlockHandler::getAllData( vector< char* >& dataVec ) const
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

void BlockHandler::globalize( const char* data, unsigned int numEntries )
{
	if ( isGlobal_ )
		return;
	isGlobal_ = true;

	assert( numEntries == totalEntries_ );
	dinfo()->destroyData( data_ );
	data_ = dinfo()->copyData( data, numEntries, numEntries );
	start_ = 0;
	end_ = totalEntries_ = numEntries;

}

void BlockHandler::unGlobalize()
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


/// The copy function is defined in the derived classes.
/// The copyUsingNewDinfo function is defined in the derived classes.
/// The resize function is defined in the derived classes.

void BlockHandler::assign( const char* orig, unsigned int numOrig )
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
