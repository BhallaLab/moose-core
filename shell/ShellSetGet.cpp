/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Shell.h"

////////////////////////////////////////////////////////////////////////
// Functions for handling field set/get and func calls
////////////////////////////////////////////////////////////////////////

void Shell::expectVector( bool flag )
{
	gettingVector_ = flag;
}

void Shell::recvGet( const Eref& e, const Qinfo* q, PrepackedBuffer pb )
{
	if ( myNode_ == 0 ) {
		if ( gettingVector_ ) {
			ObjId tgt = q->src();
			unsigned int linearIndex = q->src().eref().linearIndex();
			if ( linearIndex >= getBuf_.size() ) {
				if ( linearIndex >= getBuf_.capacity() )
					getBuf_.reserve( linearIndex * 2 );
				getBuf_.resize( linearIndex + 1 );
			}
			assert ( linearIndex < getBuf_.size() );
			double*& c = getBuf_[ linearIndex ];
			c = new double[ pb.dataSize() ];
			memcpy( c, pb.data(), pb.dataSize() * sizeof( double ) );
			// cout << myNode_ << ": Shell::recvGet[" << tgt.linearIndex() << "]= (" << pb.dataSize() << ", " <<  *reinterpret_cast< const double* >( c ) << ")\n";
		} else  {
			assert ( getBuf_.size() == 1 );
			double*& c = getBuf_[ 0 ];
			c = new double[ pb.dataSize() ];
			memcpy( c, pb.data(), pb.dataSize() * sizeof( double ) );
			handleAck( 0, OkStatus );
		}
		++numGetVecReturns_;
	}
}

////////////////////////////////////////////////////////////////////////

void Shell::clearGetBuf()
{
	for ( vector< double* >::iterator i = getBuf_.begin(); 
		i != getBuf_.end(); ++i )
	{
		if ( *i != 0 ) {
			delete[] *i;
			*i = 0;
		}
	}
	getBuf_.resize( 1, 0 );
}
