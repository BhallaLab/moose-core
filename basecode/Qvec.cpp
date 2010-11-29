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
const unsigned int Qvec::HeaderSize = sizeof( unsigned int );
static const unsigned int BLOCKSIZE = 20000;

Qvec::Qvec()
{
	;
}

Qvec::Qvec( unsigned int numThreads )
{
	data_.resize( numThreads );
	for ( unsigned int i = 0; i < numThreads; ++i )
		data_[i].reserve( Qvec::threadQreserve );
}

void Qvec::push_back( unsigned int thread, const Qinfo* q, const char* arg )
{
	assert( thread < data_.size() );
	vector< char >& d = data_[thread];
	unsigned int lastsize = d.size();
	d.resize( lastsize + sizeof ( Qinfo ) + q->size() );
	char* pos = &d[ lastsize ];
	memcpy( pos, q, sizeof( Qinfo ) );
	memcpy( pos + sizeof( Qinfo ), arg, q->size() );
}

void Qvec::clear()
{
	for ( unsigned int i = 0; i < data_.size(); ++i )
		data_[i].resize( 0 );
	linearData_.resize( 0 );
}

void Qvec::stitch()
{
	unsigned int totSize = 0;
	for ( unsigned int i = 0; i < data_.size(); ++i )
		totSize += data_[i].size();
	
	unsigned int tgtSize = BLOCKSIZE;
	if ( ( totSize + HeaderSize ) > BLOCKSIZE )
		tgtSize = totSize + HeaderSize;
	if ( linearData_.size() != tgtSize )
		linearData_.resize( tgtSize );

	setMpiDataSize( totSize + HeaderSize );
	if ( totSize == 0 ) {
		return;
	}

	totSize = HeaderSize;
	for ( unsigned int i = 0; i < data_.size(); ++i ) {
		char* pos = &linearData_[ totSize ];
		memcpy( pos, &( data_[i][0] ), data_[i].size() );
		totSize += data_[i].size();
	}
}

// This has to be protected by a check on the data size.
const char* Qvec::data() const
{
	assert( linearData_.size() > HeaderSize );
	return &linearData_[ HeaderSize ];
}

char* Qvec::writableData()
{
	assert( linearData_.size() > HeaderSize );
	return &linearData_[0];
}

unsigned int Qvec::dataQsize() const
{
	if ( linearData_.size() >= HeaderSize )
		return *reinterpret_cast< const unsigned int* >( &linearData_[0] ) -
			HeaderSize;
	return 0;
}

unsigned int Qvec::allocatedSize() const
{
	return linearData_.size();
}

unsigned int Qvec::numThreads() const
{
	return data_.size();
}

unsigned int Qvec::numEntries( unsigned int threadNum ) const
{
	assert( threadNum < data_.size() );
	const char* pos = &( data_[ threadNum ][0] );
	const char* end = pos + data_[ threadNum ].size();

	unsigned int ret = 0;
	while ( pos < end ) {
		const Qinfo* q = reinterpret_cast< const Qinfo* >( pos );
		pos += sizeof( Qinfo ) + q->size();
		++ret;
	}
	return ret;
}

unsigned int Qvec::totalNumEntries() const
{
	unsigned int ret = 0;
	for ( unsigned int i = 0; i < data_.size(); ++i )
		ret += numEntries( i );
	return ret;
}

bool Qvec::isBigBlock() const
{
	assert( linearData_.size() > HeaderSize );
	const unsigned int *mpiDataSize = 
		reinterpret_cast< const unsigned int* >( &linearData_[0] );
	return ( mpiDataSize[1] == 0 );
}

unsigned int  Qvec::mpiArrivedDataSize() const
{
	assert( linearData_.size() > HeaderSize );
	const unsigned int *mpiDataSize = 
		reinterpret_cast< const unsigned int* >( &linearData_[0] );
	return ( *mpiDataSize );
}

void Qvec::resizeLinearData( unsigned int size )
{
	linearData_.resize( size );
	if ( size > HeaderSize )
		setMpiDataSize( HeaderSize );
}

void Qvec::setMpiDataSize( unsigned int arrived )
{
	if ( arrived < HeaderSize )
		arrived = HeaderSize; // Always put in header size for data xfer.
	assert( linearData_.size() > HeaderSize );
	*( reinterpret_cast< unsigned int* >( &linearData_[ 0 ] ) ) = arrived;
}

// static function
void Qvec::testQvec()
{
	static const unsigned int bigDataSize = 1234;
	static const unsigned int smallDataSize = 12;
	Qvec q( 4 );
	assert( q.dataQsize() == 0 );
	assert( q.allocatedSize() == 0 );
	
	char *buf = new char[5000];
	Qinfo qi( 0, bigDataSize, buf );
	unsigned int datasize = sizeof( Qinfo ) + bigDataSize;

	//// Put a big chunk of data on thread 0.
	assert( q.numEntries( 0 ) == 0 );
	assert( q.totalNumEntries() == 0 );
	q.push_back( 0, &qi, buf );
	////
	assert( q.numEntries( 0 ) == 1 );
	assert( q.totalNumEntries() == 1 );
	assert( q.dataQsize() == 0 ); // Isn't assigned till 'stitch'.
	assert( q.allocatedSize() == 0 );
	q.stitch();
	assert( q.dataQsize() == datasize );
	assert( q.allocatedSize() == BLOCKSIZE );

	assert( q.data() == q.writableData() + HeaderSize );
	assert( q.data() == &( q.linearData_[ HeaderSize ] ) );

	//// Put some more data (small) on thread 0.
	Qinfo qiSmall( 0, smallDataSize, buf );
	unsigned int datasizeSmall = sizeof( Qinfo ) + smallDataSize;
	q.push_back( 0, &qiSmall, buf );
	////
	assert( q.numEntries( 0 ) == 2 );
	assert( q.totalNumEntries() == 2 );
	assert( q.dataQsize() == datasize );
	assert( q.allocatedSize() == BLOCKSIZE );
	q.stitch();
	assert( q.dataQsize() == datasize + datasizeSmall );
	assert( q.allocatedSize() == BLOCKSIZE );

	//// Put some data on thread 2
	q.push_back( 2, &qiSmall, buf );
	////
	assert( q.numEntries( 0 ) == 2 );
	assert( q.numEntries( 2 ) == 1 );
	assert( q.totalNumEntries() == 3 );
	assert( q.dataQsize() == datasize + datasizeSmall );
	assert( q.allocatedSize() == BLOCKSIZE );
	q.stitch();
	assert( q.dataQsize() == datasize + 2 * datasizeSmall );
	assert( q.allocatedSize() == BLOCKSIZE );

	const char* data = q.data();
	const Qinfo* temp = reinterpret_cast< const Qinfo* >( data );
	assert( temp->size() == bigDataSize );
	temp = reinterpret_cast< const Qinfo* >( data + datasize );
	assert( temp->size() == smallDataSize );

	temp = reinterpret_cast< const Qinfo* >( data + datasize + datasizeSmall );
	assert( temp->size() == smallDataSize );
	/////// Put some data on thread 1, requiring that the data on thread2
	// shift over.
	q.push_back( 1, &qi, buf );
	q.stitch();
	assert( q.numEntries( 0 ) == 2 );
	assert( q.numEntries( 1 ) == 1 );
	assert( q.numEntries( 2 ) == 1 );
	assert( q.totalNumEntries() == 4 );
	assert( q.dataQsize() == 2 * datasize + 2 * datasizeSmall );
	assert( q.allocatedSize() == BLOCKSIZE );

	data = q.data();
	temp = reinterpret_cast< const Qinfo* >( data );
	assert( temp->size() == bigDataSize );
	temp = reinterpret_cast< const Qinfo* >( data + datasize );
	assert( temp->size() == smallDataSize );

	/////// Now check function of the 'stitch' command.
	q.stitch();

	// start data for thread 0.
	data = q.data();
	temp = reinterpret_cast< const Qinfo* >( data );
	assert( temp->size() == bigDataSize );

	data += sizeof( Qinfo ) + temp->size();
	temp = reinterpret_cast< const Qinfo* >( data );
	assert( temp->size() == smallDataSize );
	
	// start data for thread 1.
	data += sizeof( Qinfo ) + temp->size();
	temp = reinterpret_cast< const Qinfo* >( data );
	assert( temp->size() == bigDataSize );

	// start data for thread 2.
	data += sizeof( Qinfo ) + temp->size();
	temp = reinterpret_cast< const Qinfo* >( data );
	assert( temp->size() == smallDataSize );

	delete[] buf;

	cout << "." << flush;
}
