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
#include <iostream>
using namespace std;
#include "SpikeRingBuffer.h"

const unsigned int SpikeRingBuffer::MAXBIN = 128;

SpikeRingBuffer::SpikeRingBuffer()
		: dt_( 1e-4 ), 
		currTime_( 0 ),
		currentBin_( 0 ), 
		weightSum_( 20, 0.0 )
{;}

void SpikeRingBuffer::reinit( double dt, double bufferTime )
{
	dt_ = dt;
	currTime_ = 0.0;
	currentBin_ = 0;
	unsigned int newsize = bufferTime / dt;
	if ( newsize > MAXBIN ) {
		cout << "Warning: SpikeRingBuffer::reinit: buffer size too big: " <<
			newsize << " = " << bufferTime << " / " << 
				dt << ", using " << MAXBIN << endl;
		newsize = MAXBIN;
	}
	if ( newsize == 0 ) {
		cout << "Warning: SpikeRingBuffer::reinit: buffer size too small: " <<
			newsize << " = " << bufferTime << " / " << 
				dt << ", using " << 10 << endl;
		newsize = 10;
	}
	weightSum_.clear();
	weightSum_.resize( newsize, 0.0 );
}

void SpikeRingBuffer::addSpike( double t, double w )
{
	unsigned int bin = round( ( t - currTime_ ) / dt_ );
	// Should do catch-throw here
	if ( bin < 0 ) {
		cout << "Warning: SpikeRingBuffer: handling spike too late: " <<
			t << " <  " << currTime_ << ", using currTime\n";
		bin = 0;
	} else if ( bin >= MAXBIN ) {
		cout << "Warning: SpikeRingBuffer: bin number exceeds limit: " <<
			bin << " >=  " << MAXBIN << ", terminating\n";
			assert( 0 );
	} 
	if ( bin > weightSum_.size() ) {
		weightSum_.resize( bin + 1 );
	}

	// Replace the % with a bitwise operation.
	weightSum_[ ( bin + currentBin_ ) % weightSum_.size() ] += w;
}

double SpikeRingBuffer::pop( double currTime )
{
	currTime_ = currTime;
	double ret = weightSum_[ currentBin_ ];
	weightSum_[ currentBin_++ ] = 0.0;
	return ret;
}
