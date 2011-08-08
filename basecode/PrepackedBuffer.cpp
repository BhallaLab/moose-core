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
{
	data_ = new double[ dataSize + 2 ];
	data_[0] = dataSize;
	data_[1] = numEntries;
	memcpy( data_ + 2, data, dataSize * sizeof( double ) );
}

PrepackedBuffer::PrepackedBuffer( const PrepackedBuffer& other )
{
	unsigned int dataSize = other.dataSize();
	data_ = new double[ dataSize + 2 ];
	memcpy( data_, other.data_, ( 2 + dataSize ) * sizeof( double ) );
}

PrepackedBuffer::PrepackedBuffer( const double* buf )
{
	unsigned int dataSize = buf[0];
	data_ = new double[ dataSize + 2 ];
	memcpy( data_, buf, ( 2 + dataSize ) * sizeof( double ) );
}

PrepackedBuffer::PrepackedBuffer() // Used to make StrSet happy
{
	data_ = new double[2];
	data_[0] = 0;
	data_[1] = 0;
}

PrepackedBuffer::~PrepackedBuffer()
{
	delete[] data_;
}

const double* PrepackedBuffer::operator[]( unsigned int index ) const
{
	unsigned int dataSize = data_[0];
	unsigned int numEntries = data_[1];
	unsigned int individualDataSize = dataSize / numEntries;
	if ( numEntries == 0 )
		return data_ + 2 ;
	return data_ + ( index % numEntries ) * individualDataSize;
}

unsigned int PrepackedBuffer::conv2buf( double* buf ) const
{
	unsigned int dataSize = data_[0];

	memcpy( buf, data_, ( 2 + dataSize ) * sizeof( double ) );
	return 2 + dataSize;
}
