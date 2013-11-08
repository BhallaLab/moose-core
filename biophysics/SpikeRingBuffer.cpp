/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <math.h>
#include <vector>
#include <cassert>
using namespace std;
#include "SpikeRingBuffer.h"

SpikeRingBuffer::SpikeRingBuffer()
		: dt_( 1e-4 ), currentBin_( 0 ), weightSum_( 20, 0.0 )
{;}

void SpikeRingBuffer::reinit( double dt )
{
	dt_ = dt;
	currentBin_ = 0;
	weightSum_.assign( weightSum_.size(), 0.0 );
}

void SpikeRingBuffer::addSpike( double t, double w )
{
	static const unsigned int MAXBIN = 1000;
	assert( t > 0.0 );
	unsigned int bin = round( t / dt_ );
	// Replace this with a catch-throw
	if ( bin >= weightSum_.size() ) {
		assert( bin < MAXBIN );
		weightSum_.resize( bin + 1 );
	}
	// Replace the % with a bitwise operation.
	weightSum_[ ( bin + currentBin_ ) % weightSum_.size() ] += w;
}

double SpikeRingBuffer::pop()
{
	double ret = weightSum_[ currentBin_ ];
	weightSum_[ currentBin_++ ] = 0.0;
	return ret;
}
