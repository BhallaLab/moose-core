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

AnyDimHandler::AnyDimHandler( const DinfoBase* dinfo, 
	const vector< DimInfo >& dims,
	unsigned short pathDepth,
	bool isGlobal )
		: BlockHandler( dinfo, dims, pathDepth, isGlobal )
{;}

AnyDimHandler::AnyDimHandler( const AnyDimHandler* other )
	: BlockHandler( other )
{;}

AnyDimHandler::~AnyDimHandler()
{;}

////////////////////////////////////////////////////////////////////////
// Information functions
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// Load balancing
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// Process function
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// Data Reallocation functions.
////////////////////////////////////////////////////////////////////////

DataHandler* AnyDimHandler::copy( unsigned short copyDepth,
	bool toGlobal, unsigned int n ) const
{
	if ( toGlobal ) {
		if ( !isGlobal() ) {
			cout << "Warning: AnyDimHandler::copy: Cannot copy from nonGlob    al to global\n";
			return 0;
		}
	}
	if ( n > 1 ) {
		DimInfo temp = {n, copyDepth, 0 };
		vector< DimInfo > newDims;
		newDims.push_back( temp );
		for ( unsigned int i = 0; i < dims_.size(); ++i ) {
			newDims.push_back( dims_[0] );
			newDims.back().depth += copyDepth - pathDepth_;
		}
		AnyDimHandler* ret = new AnyDimHandler( dinfo(), 
			newDims, copyDepth, toGlobal );
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
	return new AnyDimHandler( dinfo, dims_, pathDepth_, isGlobal_ );
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
		if ( dims_[ dimension ].size == numEntries )
			return 0;
		unsigned int oldN = dims_[ dimension ].size;
		unsigned int oldTotalEntries = totalEntries_;
		dims_[ dimension ].size = numEntries;
		totalEntries_ = 1;
		for ( unsigned int i = 0; i < dims_.size(); ++i ) {
			totalEntries_ *= dims_[i].size;
		}
		
		if ( dimension == 0 ) {
			// go from 1 2 3 : 4 5 6 to 1 2 3 .. : 4 5 6 ..
			// Try to preserve original data, possible if it is global.
			char* temp = data_;
			innerNodeBalance( totalEntries_, 
				Shell::myNode(), Shell::numNodes() );
			dims_[0].size = numEntries;
			unsigned int newLocalEntries = end_ - start_;
			data_ = dinfo()->allocData( newLocalEntries );
			if ( isGlobal_ ) {
				assert ( totalEntries_ == newLocalEntries);
				unsigned int newBlockSize = dims_[0].size * dinfo()->size();
				unsigned int oldBlockSize = oldN * dinfo()->size();
				unsigned int j = totalEntries_ / dims_[0].size;
				for ( unsigned int i = 0; i < j; ++i ) {
					dinfo()->assignData( data_ + i * newBlockSize, 
						dims_[0].size, temp + i * oldBlockSize, oldN );
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
			dims_[dimension].size = numEntries;
			dinfo()->destroyData( temp );
		}
	}
	return 0;
}
