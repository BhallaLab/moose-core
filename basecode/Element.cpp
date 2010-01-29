/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

Element::Element( const Cinfo* c, 
	char* d, unsigned int numData, unsigned int dataSize, 
		unsigned int numFuncIndex, unsigned int numConn )
	: d_( d ), numData_( numData ), dataSize_( dataSize ), 
	sendBuf_( 0 ), cinfo_( c ), c_( numConn )
{ 
	targetFunc_.resize( numFuncIndex, 0 );
}

Element::Element( const Cinfo* c, const Element* other )
	: 	d_( other->d_ ), 
		numData_( other->numData_ ), 
		dataSize_( other->dataSize_),
		sendBuf_( 0 ), cinfo_( c ), c_( c->numConn() )
{
	targetFunc_.resize( c->numFuncIndex(), 0 );
}

Element::~Element()
{
	delete[] sendBuf_;
	cinfo_->destroy( d_ );
	cinfo_ = 0; // A flag that the Element is doomed, used to avoid lookups when deleting Msgs.
	for ( vector< Conn >::iterator i = c_.begin(); i != c_.end(); ++i )
		i->clearConn(); // Get rid of Msgs on them.
	for ( vector< MsgId >::iterator i = m_.begin(); i != m_.end(); ++i )
		if ( *i ) // Dropped Msgs set this pointer to zero, so skip them.
			Msg::deleteMsg( *i );
}

/**
 * The indices handled by each thread are in blocks
 * Thread0 handles the first (numData_ / numThreads ) indices
 * Thread1 handles ( numData_ / numThreads ) to (numData_*2 / numThreads)
 * and so on.
 */
void Element::process( const ProcInfo* p )
{
	char* data = d_;
	unsigned int start =
		( numData_ * p->threadIndexInGroup ) / p->numThreadsInGroup;
	unsigned int end =
		( numData_ * ( p->threadIndexInGroup + 1) ) / p->numThreadsInGroup;
	data += start * dataSize_;
	for ( unsigned int i = start; i < end; ++i ) {
		reinterpret_cast< Data* >( data )->process( p, Eref( this, i ) );
		data += dataSize_;
	}
}


double Element::sumBuf( SyncId slot, unsigned int i ) const
{
	vector< unsigned int >::const_iterator offset = 
		procBufRange_.begin() + slot + i * numRecvSlots_;
	vector< double* >::const_iterator begin = 
		procBuf_.begin() + *offset++;
	vector< double* >::const_iterator end = 
		procBuf_.begin() + *offset;
	double ret = 0.0;
	for ( vector< double* >::const_iterator i = begin; 
		i != end; ++i )
		ret += **i;
	return ret;
}

double Element::prdBuf( SyncId slot, unsigned int i, double v )
	const
{
	vector< unsigned int >::const_iterator offset = 
		procBufRange_.begin() + slot + i * numRecvSlots_;
	vector< double* >::const_iterator begin = 
		procBuf_.begin() + *offset++;
	vector< double* >::const_iterator end = 
		procBuf_.begin() + *offset;
	for ( vector< double* >::const_iterator i = begin;
		i != end; ++i )
		v *= **i;
	return v;
}

double Element::oneBuf( SyncId slot, unsigned int i ) const
{
	// unsigned int offset = i * numData_ + slot;
	unsigned int offset = slot + i * numRecvSlots_;
	assert( offset + 1 < procBufRange_.size() );
	return *procBuf_[ procBufRange_[ offset ] ];
}

double* Element::getBufPtr( SyncId slot, unsigned int i )
{
	// unsigned int offset = i * numData_ + slot;
	unsigned int offset = slot + i * numRecvSlots_;
	assert( offset + 1 < procBufRange_.size() );
	return procBuf_[ procBufRange_[ offset ] ];
}

void Element::ssend1( SyncId slot, unsigned int i, double v )
{
	sendBuf_[ slot + i * numSendSlots_ ] = v;
}

void Element::ssend2( SyncId slot, unsigned int i, double v1, double v2 )
{
	double* sb = sendBuf_ + slot + i * numSendSlots_;
	*sb++ = v1;
	*sb = v2;
}

char* Element::data( DataId index )
{
	assert( index.data() < numData_ );
	return d_ + index.data() * dataSize_;
}

char* Element::data1( DataId index )
{
	assert( index.data() < numData_ );
	return d_ + index.data() * dataSize_;
}

unsigned int Element::numData() const
{
	return numData_;
}

unsigned int Element::numData1() const
{
	return numData_;
}

unsigned int Element::numData2( unsigned int index1 ) const
{
	return 1;
}

unsigned int Element::numDimensions() const
{
	return 1;
}

void Element::setArraySizes( const vector< unsigned int >& sizes )
{;}

void Element::getArraySizes( vector< unsigned int >& sizes ) const
{
	sizes.clear();
	sizes.resize( numData_, 1 );
}

const Conn* Element::conn( ConnId c ) const {
	assert( c < c_.size() );
	return &( c_[c] );
}

void Element::addMsg( MsgId m )
{
	while ( m_.size() > 0 ) {
		if ( m_.back() == Msg::Null )
			m_.pop_back();
		else
			break;
	}
	m_.push_back( m );
}

/**
 * Called from ~Msg. This requires the usual scan through all msgs,
 * and could get inefficient.
 */
void Element::dropMsg( MsgId mid )
{
	if ( cinfo_ == 0 ) // Don't need to clear if Element itself is going.
		return;
	// Here we have the spectacularly ugly C++ erase-remove idiot.
	m_.erase( remove( m_.begin(), m_.end(), mid ), m_.end() );

	for ( vector< Conn >::iterator i = c_.begin(); i != c_.end(); ++i )
		i->drop( mid ); // Get rid of specific Msg, if present, on Conn
}

void Element::addMsgToConn( MsgId mid, ConnId cid )
{
	if ( c_.size() < cid + 1 )
		c_.resize( cid + 1 );
	c_[cid].add( mid );
}

void Element::clearConn( ConnId cid )
{
	assert( cid < c_.size() );
	c_[cid].clearConn();
}

const Cinfo* Element::cinfo() const
{
	return cinfo_;
}

void Element::addTargetFunc( FuncId fid, unsigned int funcIndex )
{
	if ( targetFunc_.size() < funcIndex + 1 )
		targetFunc_.resize( funcIndex + 1 );
	targetFunc_[ funcIndex ] = fid;
}

FuncId Element::getTargetFunc( unsigned int funcIndex ) const
{
	assert ( targetFunc_.size() > funcIndex );
	return targetFunc_[ funcIndex ];
}
