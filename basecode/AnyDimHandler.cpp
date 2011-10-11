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
		: BlockHandler( dinfo, isGlobal )
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
}

AnyDimHandler::AnyDimHandler( const AnyDimHandler* other )
	: BlockHandler( other ), 
	  dims_( other->dims_ )
{;}

AnyDimHandler::~AnyDimHandler()
{;}

////////////////////////////////////////////////////////////////////////
// Information functions
////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////
// Load balancing
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// Process function
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// Data Reallocation functions.
////////////////////////////////////////////////////////////////////////

DataHandler* AnyDimHandler::copy( bool toGlobal, unsigned int n ) const
{
	if ( toGlobal ) {
		if ( !isGlobal() ) {
			cout << "Warning: AnyDimHandler::copy: Cannot copy from nonGlob    al to global\n";
			return 0;
		}
	}
	if ( n > 1 ) {
		vector< int > newDims( dims_.size() + 1);
		for ( unsigned int i = 0; i < dims_.size(); ++i )
			newDims[i] = dims_[i];
		newDims[ dims_.size() ] = n;

		AnyDimHandler* ret = new AnyDimHandler( dinfo(), toGlobal, newDims);
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
				assert ( totalEntries_ == newLocalEntries);
				unsigned int newBlockSize = dims_[0] * dinfo()->size();
				unsigned int oldBlockSize = oldN * dinfo()->size();
				unsigned int j = totalEntries_ / dims_[0];
				for ( unsigned int i = 0; i < j; ++i ) {
					dinfo()->assignData( data_ + i * newBlockSize, dims_[0],
						temp + i * oldBlockSize, oldN );
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
