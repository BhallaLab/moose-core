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
		: BlockHandler( dinfo, isGlobal, nx * ny ), 
			nx_( nx ), ny_( ny )
{;}

TwoDimHandler::TwoDimHandler( const TwoDimHandler* other )
	: BlockHandler( other ),
	  nx_( other->nx_ ),
	  ny_( other->ny_ )
{;}

TwoDimHandler::~TwoDimHandler()
{;}

////////////////////////////////////////////////////////////////////////
// Information functions
////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////
// Load balancing defined in base class BlockHandler
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
// Process, foreach functions defined in base class.
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
// Data Reallocation functions. Defined in BlockHandler
////////////////////////////////////////////////////////////////////////

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

// Assign is defined in BlockHandler class.
