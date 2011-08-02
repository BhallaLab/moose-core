/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <cstring>
#include "PrepackedBuffer.h"

PrepackedBuffer::PrepackedBuffer( 
	const double* data, unsigned int dataSize, unsigned int numEntries )
	: dataSize_( dataSize ), numEntries_( numEntries )
{
	if ( numEntries == 0 )
		individualDataSize_ = dataSize_;
	else
		individualDataSize_ = dataSize_ / numEntries_;
	
	data_ = new double[ dataSize ];
	memcpy( data_, data, dataSize * sizeof( double ) );
}

PrepackedBuffer::PrepackedBuffer( const PrepackedBuffer& other )
	: dataSize_( other.dataSize_ ), 
		numEntries_( other.numEntries_ )
{
	if ( numEntries_ == 0 )
		individualDataSize_ = dataSize_;
	else
		individualDataSize_ = dataSize_ / numEntries_;
	data_ = new double[ dataSize_ ];
	memcpy( data_, other.data_, dataSize_ * sizeof( double ) );
}

PrepackedBuffer::PrepackedBuffer( const double* buf )
	: 
		dataSize_( buf[0] ),
		numEntries_( buf[1] )
{
	if ( numEntries_ == 0 )
		individualDataSize_ = dataSize_;
	else
		individualDataSize_ = dataSize_ / numEntries_;
	data_ = new double[ dataSize_ ];
	memcpy( data_, buf + 2, dataSize_ * sizeof( double ) );
}

PrepackedBuffer::PrepackedBuffer() // Used to make StrSet happy
	: dataSize_( 0 ), numEntries_( 0 ), individualDataSize_( 0 )
{
	data_ = new double[1];
	data_[0] = 0;
}

PrepackedBuffer::~PrepackedBuffer()
{
	delete[] data_;
}

const double* PrepackedBuffer::operator[]( unsigned int index ) const
{
	if ( numEntries_ == 0 )
		return data_ ;
	return data_ + ( index % numEntries_ ) * individualDataSize_;
}

unsigned int PrepackedBuffer::conv2buf( double* buf ) const
{
	buf[0] = dataSize_;
	buf[1] = numEntries_;
	memcpy( buf + 2, data_, dataSize_ * sizeof( double ) );
	return 2 + dataSize_;
}
