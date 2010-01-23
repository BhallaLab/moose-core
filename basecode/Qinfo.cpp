/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

// Declaration of static field
vector< vector< char > > Qinfo::q_;
vector< SimGroup > Qinfo::g_;

Qinfo::Qinfo( FuncId f, DataId srcIndex, 
	unsigned int size, bool useSendTo, bool isForward )
	:	m_( 0 ), 
		useSendTo_( useSendTo ), 
		isForward_( isForward ), 
		f_( f ), 
		srcIndex_( srcIndex ),
		size_( size )
{;}

Qinfo::Qinfo( FuncId f, DataId srcIndex, unsigned int size )
	:	m_( 0 ), 
		useSendTo_( 0 ), 
		isForward_( 1 ), 
		f_( f ), 
		srcIndex_( srcIndex ),
		size_( size )
{;}

Qinfo::Qinfo()
	:	m_( 0 ), 
		useSendTo_( 0 ), 
		isForward_( 1 ), 
		f_( 0 ), 
		srcIndex_( 0 ),
		size_( 0 )
{;}

void Qinfo::expandSize()
{
	size_ += sizeof( DataId );
}

// Static func: deprecated
/*
void Qinfo::setNumQs( unsigned int n, unsigned int reserve )
{
	q_.resize( n );
	for ( unsigned int i = 0; i < n; ++i ) {
		q_[i].reserve( reserve );
	}
}
*/ 

/**
 * Static func: Sets up a SimGroup to keep track of thread and node
 * grouping info. This is used by the Qinfo to manage assignment of
 * threads and queues.
 * numThreads is the number of threads present in this group on this node.
 * Returns the group number of the new group.
 * have that because it will require updates to messages.
 */
unsigned int Qinfo::addSimGroup( unsigned short numThreads )
{
	unsigned short ng = g_.size();
	unsigned short si = 0;
	if ( ng > 0 )
		si = g_[ng - 1].startThread + g_[ng - 1].numThreads;
	SimGroup sg( numThreads, si );
	g_.push_back( sg );
	q_.resize( si + numThreads + 1 );
	for ( unsigned int i = 0; i <= numThreads; ++i ) {
		q_[i + si].reserve( 1024 );
	}
	return ng;
}

unsigned int Qinfo::numSimGroup()
{
	return g_.size();
}

const SimGroup* Qinfo::simGroup( unsigned int index )
{
	assert( index < g_.size() );
	return &( g_[index] );
}

// static func
// Note that it does not advance the buffer.
void Qinfo::hackForSendTo( const Qinfo* q, const char* buf )
{
	const DataId* tgtIndex = 
		reinterpret_cast< const DataId* >( buf + sizeof( Qinfo ) +
		q->size() - sizeof( DataId ) );

	Element* tgt;
	if ( q->isForward() )
		tgt = Msg::getMsg( q->m_ )->e2();
	else 
		tgt = Msg::getMsg( q->m_ )->e1();
	const OpFunc* func = tgt->cinfo()->getOpFunc( q->fid() );
	func->op( Eref( tgt, *tgtIndex ), buf );
}

void Qinfo::clearQ( const ProcInfo* proc )
{
	readQ( proc );
	zeroQ( proc->qId );
}

/** 
 * Static func
 * In this variant we just go through the specified queue. 
 * The job of thread safety is left to the calling function.
 * Thread safe as it is readonly in the Queue.
 */ 
void Qinfo::readQ( const ProcInfo* proc )
{
	assert( proc );
	assert( proc->qId < q_.size() );
	vector< char >& q = q_[ proc->qId ];
	const char* buf = &q[0];
	while ( buf && buf < &q.back() )
	{
		const Qinfo *q = reinterpret_cast< const Qinfo* >( buf );
		if ( q->useSendTo() ) {
			hackForSendTo( q, buf );
		} else {
			const Msg* m = Msg::getMsg( q->m_ );
			m->exec( buf, proc );
		}
		buf += sizeof( Qinfo ) + q->size();
	}
}

/**
 * Zeroes out contentsof specified queue.
 */
void Qinfo::zeroQ( Qid qId )
{
	assert( qId < q_.size() );
	q_[ qId ].resize( 0 );
	/*
	vector< char >& temp = q_[ qId ];
	for ( unsigned int i = 0; i < temp.size(); ++i )
		temp.resize( 0 );
	*/
}

/**
 * Static func. Not thread safe.
 * Merge out all outQs from a group into its inQ. This clears out inQ
 * before filling it, and clears out the outQs after putting them into inQ.
 */
void Qinfo::mergeQ( unsigned int groupId )
{
	assert( groupId < g_.size() );
	SimGroup& g = g_[ groupId ];
	unsigned int j = g.startThread;
	assert( j + g.numThreads < q_.size() );

	unsigned int totSize = 0;
	for ( unsigned int i = 0; i < g.numThreads; ++i )
		totSize += q_[ j++ ].size();

	vector< char >& inQ = q_[ groupId ];
	inQ.resize( totSize );
	j = g.startThread;
	char* buf = &inQ[0];
	for ( unsigned int i = 0; i < g.numThreads; ++i ) {
		memcpy( buf, &q_[ j ], q_[ j ].size() );
		buf += q_[ j ].size();
		j++;
		q_[ j ].resize( 0 );
	}
}

/**
 * Static func. Not thread safe. Catenates data from a buffer into queue.
 * May resize it in the process, so iterators have to watch out.
 */
void Qinfo::loadQ( Qid qId, const char* buf, unsigned int length )
{
	assert( qId < q_.size() );
	vector< char >& q = q_[qId];
	q.insert( q.end(), buf, buf + length );
}

/**
 * Static func. Not thread safe. Catenates data from a queue into buffer.
 * Does not touch the queue. Returns data size.
 */
unsigned int Qinfo::dumpQ( Qid qId, char* buf )
{
	assert( qId < q_.size() );
	vector< char >& q = q_[qId];
	memcpy( buf, &q[0], q.size() );
	return q.size();
}


// Non-static: copies itself onto queue.
// qid specifies which queue to use.
// mid assigns the msgId.
void Qinfo::addToQ( Qid qId, MsgId mid, bool isForward, const char* arg )
{
	assert( qId < q_.size() );

	vector< char >& q = q_[qId];
	unsigned int origSize = q.size();
	m_ = mid;
	isForward_ = isForward;
	q.resize( origSize + sizeof( Qinfo ) + size_ );
	char* pos = &( q[origSize] );
	memcpy( pos, this, sizeof( Qinfo ) );
	// ( reinterpret_cast< Qinfo* >( pos ) )->setForward( isForward );
	memcpy( pos + sizeof( Qinfo ), arg, size_ );
}
