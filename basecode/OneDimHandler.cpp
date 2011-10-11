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
		: BlockHandler( dinfo, isGlobal, size )
{;}

OneDimHandler::OneDimHandler( const OneDimHandler* other )
	: BlockHandler( other )
{;}

OneDimHandler::~OneDimHandler()
{;}

////////////////////////////////////////////////////////////////////////
// Information functions
////////////////////////////////////////////////////////////////////////

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
		// ny is the last argument.
		TwoDimHandler* ret = new TwoDimHandler( dinfo(), toGlobal, totalEntries_, n );
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

