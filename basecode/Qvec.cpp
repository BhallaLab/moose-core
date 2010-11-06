/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"

const unsigned int Qvec::threadQreserve = 128;

// CPU and OS specific, just pick a biggish value to handle most cases.
// Also it has to be big enough to put in a Qinfo for doing the 
// stitching between blocks.
const unsigned int Qvec::threadOverlapProtection = sizeof( Qinfo );

Qvec::Qvec( unsigned int numThreads )
	: numThreads_( numThreads )
{
	data_.resize( numThreads * threadQreserve, 0 );
	threadBlockStart_.resize( numThreads + 1 );
	threadBlockEnd_.resize( numThreads );

	for ( unsigned int i = 0; i < numThreads_; ++i )
		threadBlockStart_[i] = threadBlockEnd_[i] = i * threadQreserve;
	threadBlockStart_[ numThreads] = numThreads * threadQreserve;
}

void Qvec::push_back( unsigned int thread, const Qinfo* q, const char* arg )
{
	unsigned int tbe = threadBlockEnd_[ thread ];
	unsigned int tbs = threadBlockStart_[ thread + 1 ];
	unsigned int dataSize = sizeof( Qinfo ) + q->size();
	if ( tbe + dataSize + threadOverlapProtection > tbs )
	{
		// Do some clever reallocation here. The goal is to have the
		// blocks reasonably full.
		unsigned int extraSize = dataSize + threadQreserve;
		// insert( pos, n, val );
		data_.insert( data_.begin() + tbe, extraSize, 0 );
		for ( unsigned int i = thread + 1; i < numThreads_; ++i ) {
			threadBlockStart_[i] += extraSize;
			threadBlockEnd_[i] += extraSize;
		}
		threadBlockStart_[ numThreads_ ] += extraSize;
		assert( threadBlockStart_[ numThreads_ ] == data_.size() );
	}
	// Need to figure out pos
	char* pos = &data_[ tbe ];
	memcpy( pos, q, sizeof( Qinfo ) );
	memcpy( pos + sizeof( Qinfo ), arg, q->size() );
	threadBlockEnd_[ thread ] += dataSize;
}

void Qvec::clear()
{
	for ( unsigned int i = 0; i < numThreads_; ++i )
		threadBlockEnd_[i] = threadBlockStart_[i];
}

void Qvec::stitch()
{
	for ( unsigned int i = 0; i < numThreads_; ++i ) {
		// Qinfo( srcIndex, size, useSendTo )
		unsigned int remainingSize = 
			threadBlockStart_[ i + 1 ] - threadBlockEnd_[i];
		assert( remainingSize >= sizeof( Qinfo ) );
		Qinfo q( 0, remainingSize, 0 );
		memcpy( &data_[ threadBlockEnd_[i] ], &q, remainingSize );
	}
}

const char* Qvec::data() const
{
	assert( data_.size() > threadQreserve );
	return &data_[0];
}

char* Qvec::writableData()
{
	assert( data_.size() > threadQreserve );
	return &data_[0];
}

unsigned int Qvec::usedSize() const
{
	assert( numThreads_ > 0 );
	return threadBlockEnd_[ numThreads_ - 1 ];
}

unsigned int Qvec::allocatedSize() const
{
	return data_.size();
}

// static function
void Qvec::testQvec()
{
}
