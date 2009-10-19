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
	Data* d, unsigned int numData, unsigned int dataSize, 
		unsigned int numFuncIndex, unsigned int numConn )
	: d_( d ), numData_( numData ), dataSize_( dataSize ), 
	sendBuf_( 0 ), cinfo_( c ), c_( numConn )
{ 
	targetFunc_.resize( numFuncIndex, 0 );
}

/*
Element::Element( vector< Data* >& d, 
	unsigned int numSendSlots, unsigned int numRecvSlots )
	: d_( d ), 
	finfo_( d_[0]->initClassInfo() ), 
	numSendSlots_( numSendSlots ),
	numRecvSlots_( numRecvSlots )
{
	q_.resize( 16, 0 ); // Put in place space for at least one entry.
}
*/

Element::~Element()
{
	delete[] sendBuf_;
	cinfo_->destroy( d_ );
	for ( vector< Conn >::iterator i = c_.begin(); i != c_.end(); ++i )
		i->clearConn(); // Get rid of Msgs on them.
	/*
	for ( vector< Data* >::iterator i = d_.begin(); i != d_.end(); ++i )
		delete *i;
	*/
	for ( vector< Msg* >::iterator i = m_.begin(); i != m_.end(); ++i )
		if ( *i ) // Dropped Msgs set this pointer to zero, so skip them.
			delete *i;
}

void Element::process( const ProcInfo* p )
{
	char* data = reinterpret_cast< char* >( d_ );
	for ( unsigned int i = 0; i < numData_; ++i ) {
		reinterpret_cast< Data* >( data )->process( p, Eref( this, i ) );
		data += dataSize_;
	}

	/*
	for ( unsigned int i = 0; i < d_.size(); ++i )
		d_[i]->process( p, Eref( this, i ) );
		*/
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

Data* Element::data( unsigned int index )
{
	assert( index < numData_ );
	return reinterpret_cast< Data* >( 
		reinterpret_cast< char* >( d_ ) + index * dataSize_ );
}

/*
const vector< Msg* >& Element::msg( SyncId slot ) const
{
	assert( msg_.size() > slot );
	return msg_[ slot ];
}
*/

const Conn& Element::conn( ConnId c ) const {
	assert( c < c_.size() );
	return c_[c];
}

const Msg* Element::getMsg( MsgId mid ) const {
	assert( mid < m_.size() );
	return m_[ mid ];
}

/**
 * Parses the buffer and executes the func in all specified Data
 * objects on the Element.
 * Returns new position on buffer.
 * The buffer looks like this:
 * uint FuncId, uint MsgId, Args
 *
 * The Msg does the iteration, and as it is a virtual base class
 * it can do all sorts of optimizations depending on how the mapping looks.
 *
 */
const char* Element::execFunc( const char* buf )
{
	assert( buf != 0 );

	Qinfo q = *( reinterpret_cast < const Qinfo * >( buf ) );

	if ( q.useSendTo() ) {
		// const char* temp = buf + sizeof( Qinfo );
		unsigned int tgtIndex =
			*reinterpret_cast< const unsigned int* >( 
			buf + sizeof( Qinfo ) + q.size() - sizeof( unsigned int ) );
		if ( tgtIndex < numData_ ) {
			// checks for valid func
			const OpFunc* func = cinfo_->getOpFunc( q.fid() );
			func->op( Eref( this, tgtIndex ), buf );
		} else {
			cout << "Warning: Message to nonexistent Element index " << 
				tgtIndex << " on " << this << endl;
		}
	} else {
		const Msg* m = getMsg( q.mid() ); // Runtime check for Msg identity.
		if ( m )
			m->exec( this, buf );
	}


/*
	buf += sizeof( Qinfo );
	const OpFunc* func = cinfo_->getOpFunc( q.fid() ); // checks for valid func
	const Msg* m = getMsg( q.mid() ); // Runtime check for Msg identity.

	if ( q.useSendTo() ) {
		unsigned int tgtIndex =
			*reinterpret_cast< const unsigned int* >( buf + q.size() - sizeof( unsigned int ) );
		if ( tgtIndex < numData_ ) {
			func->op( Eref( this, tgtIndex ), buf );
		} else {
			cout << "Warning: Message to nonexistent Element index " << 
				tgtIndex << " on " << this << endl;
		}
	} else if ( func && m ) {
		m->exec( this, func, q.srcIndex(), buf );
	}
*/

	return buf + sizeof( Qinfo) + q.size();;
}

/**
 * clearQ: goes through async function request queue and carries out
 * operations on it. 
 *
 * Node decomposition of Element: Two stage process. First, the Queue
 * request itself must reach all target nodes. This will have to be 
 * managed by the Connection and set up with the Element when the
 * Connection is created. Second, the Element must provide the Msg
 * with range info. Msg will use efficiently to choose what to call.
 *
 * Thread decomposition of Element:	Incoming data is subdivided into
 * per-thread buffers. It would be nice to clearQ these per-thread too.
 * Need some way to shift stuff around for balancing. Must ensure that
 * only one Msg deals with any given index.
 *
 */
void Element::clearQ( )
{
	const char* buf = &(q_[0]);
//	while ( buf && *reinterpret_cast< const FuncId* >(buf) != ENDFUNC )
	while ( buf && buf < &q_.back() )
	{
		buf = execFunc( buf );
	}
	q_.resize( 0 );
}

/**
 * This function pushes a function event onto the queue.
 * It should be extended to provide thread safety, which can be done
 * if each thread has its own queue.
 */
void Element::addToQ( const Qinfo& qi, const char* arg )
{
	qi.addToQ( q_, arg );
}

MsgId Element::addMsg( Msg* m )
{
	m_.push_back( m );
	return m_.size() - 1;
}

/**
 * Called from ~Msg.
 */
void Element::dropMsg( const Msg* m, MsgId mid )
{
	assert ( mid < m_.size() );
	assert( m == m_[mid] );

	m_[mid] = 0; // To clean up later, if at all.
	for ( vector< Conn >::iterator i = c_.begin(); i != c_.end(); ++i )
		i->drop( m ); // Get rid of specific Msg, if present, on Conn
}

void Element::addMsgToConn( Msg* m, ConnId cid )
{
	if ( c_.size() < cid + 1 )
		c_.resize( cid + 1 );
	c_[cid].add( m );
}

void Element::clearConn( ConnId cid )
{
	assert( cid < c_.size() );
	// if ( c_.size() > cid )
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
