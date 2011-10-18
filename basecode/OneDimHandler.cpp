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

OneDimHandler::OneDimHandler( const DinfoBase* dinfo, 
	const vector< DimInfo >& dims,
	unsigned short pathDepth, bool isGlobal )
		: BlockHandler( dinfo, dims, pathDepth, isGlobal )
{;}

OneDimHandler::OneDimHandler( const OneDimHandler* other )
	: BlockHandler( other )
{;}

OneDimHandler::~OneDimHandler()
{;}

////////////////////////////////////////////////////////////////////////
// Information functions
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
// Load balancing
/// Defined in base class BlockHandler.
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
// Process function
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
// Data Reallocation functions.
////////////////////////////////////////////////////////////////////////

DataHandler* OneDimHandler::copy( unsigned short newParentDepth,
	unsigned short copyRootDepth,
	bool toGlobal, unsigned int n ) const
{
	if ( toGlobal ) {
		if ( !isGlobal() ) {
			cout << "Warning: OneDimHandler::copy: Cannot copy from nonGlob    al to global\n";
			return 0;
		}
	}

	// don't allow copying that would kill the array.
	// Later we can put in options to copy part of the array.
	if ( copyRootDepth > dims_[0].depth )
		return 0;

	if ( n > 1 ) {
		// Note that we expand into ny, rather than nx. The current array
		// size is going to become the fastest varying one, highest index.
		// The copyRoot is going to be multiplied.
		DimInfo temp = { n, newParentDepth + 1, 0 };
		vector< DimInfo > newDims;
		newDims.push_back( temp ); // new dim is closest to root.
		newDims.push_back( dims_[0] ); // Old index is fastest varying.

		// depth of array portion adjusted.
		newDims.back().depth += 1 + newParentDepth - copyRootDepth;

		TwoDimHandler* ret = new TwoDimHandler( dinfo(), newDims, 
			1 + pathDepth() + newParentDepth - copyRootDepth, toGlobal );
		if ( data_ )  {
			if ( isGlobal() ) {
				ret->assign( data_, totalEntries_ );
			} else {
				ret->assign( data_, end_ - start_ );
			}
		}
		return ret;
	} else {
		OneDimHandler* ret = new OneDimHandler( this );
		if ( !ret->changeDepth( pathDepth() + 1 + newParentDepth - copyRootDepth ) ) {
			delete ret;
			return 0;
		}
		return ret;
	}
	return 0;
}

DataHandler* OneDimHandler::copyUsingNewDinfo( const DinfoBase* dinfo) const
{
	return new OneDimHandler( dinfo, dims_, pathDepth_, isGlobal_ );
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
		dims_[0].size = numEntries;
		innerNodeBalance( numEntries, 
			Shell::myNode(), Shell::numNodes() );
		unsigned int newN = end_ - start_;
		data_ = dinfo()->copyData( temp, n, newN );
		return 1;
	}
	return 0;
}
