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
vector< vector< char > > Qinfo::inQ_;
vector< vector< char > > Qinfo::outQ_;
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
	inQ_.resize( g_.size() );
	outQ_.resize( si + numThreads );
	for ( unsigned int i = 0; i < numThreads; ++i ) {
		outQ_[i + si].reserve( 1024 );
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
	mergeQ( proc->groupId );
	readQ( proc );
	inQ_[ proc->groupId ].resize( 0 );
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
	assert( proc->groupId < inQ_.size() );
	vector< char >& q = inQ_[ proc->groupId ];
	const char* buf = &q[0];
	while ( buf && buf < &q.back() )
	{
		const Qinfo *qi = reinterpret_cast< const Qinfo* >( buf );
		if ( qi->useSendTo() ) {
			hackForSendTo( qi, buf );
		} else {
			const Msg* m = Msg::getMsg( qi->m_ );
			m->exec( buf, proc );
		}
		buf += sizeof( Qinfo ) + qi->size();
	}
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
	assert( j + g.numThreads <= outQ_.size() );

	unsigned int totSize = 0;
	for ( unsigned int i = 0; i < g.numThreads; ++i )
		totSize += outQ_[ j++ ].size();

	vector< char >& inQ = inQ_[ groupId ];
	inQ.resize( totSize );
	j = g.startThread;
	char* buf = &inQ[0];
	for ( unsigned int i = 0; i < g.numThreads; ++i ) {
		memcpy( buf, &(outQ_[ j ][0]), outQ_[ j ].size() );
		buf += outQ_[ j ].size();
		outQ_[ j ].resize( 0 );
		j++;
	}
}

/**
 * Static func. Not thread safe. Catenates data from a buffer into 
 * specified inQ.
 * May resize it in the process, so iterators have to watch out.
 */
void Qinfo::loadQ( Qid qid, const char* buf, unsigned int length )
{
	assert( qid < inQ_.size() );
	vector< char >& q = inQ_[qid];
	q.insert( q.end(), buf, buf + length );
}

/**
 * Static func. Not thread safe. Catenates data from a outQ into buffer.
 * Does not touch the queue. Returns data size.
 * Should perhaps replace qid with the proc or groupid so it can dump
 * the whole set.
 */
unsigned int Qinfo::dumpQ( Qid qId, char* buf )
{
	assert( qId < outQ_.size() );
	vector< char >& q = outQ_[qId];
	memcpy( buf, &q[0], q.size() );
	return q.size();
}

/**
 * Static func. readonly, so it is thread safe
 */
void Qinfo::reportQ()
{
	cout << "	inQ: ";
	for ( unsigned int i = 0; i < inQ_.size(); ++i )
		cout << "[" << i << "]=" << inQ_[i].size() << "	";
	cout << "outQ: ";
	for ( unsigned int i = 0; i < outQ_.size(); ++i )
		cout << "[" << i << "]=" << outQ_[i].size() << "	";
	cout << endl;
}


// Non-static: copies itself onto queue.
// qid specifies which queue to use. Must be an outQ.
// mid assigns the msgId.
void Qinfo::addToQ( Qid qId, MsgId mid, bool isForward, const char* arg )
{
	assert( qId < outQ_.size() );

	vector< char >& q = outQ_[qId];
	unsigned int origSize = q.size();
	m_ = mid;
	isForward_ = isForward;
	q.resize( origSize + sizeof( Qinfo ) + size_ );
	char* pos = &( q[origSize] );
	memcpy( pos, this, sizeof( Qinfo ) );
	// ( reinterpret_cast< Qinfo* >( pos ) )->setForward( isForward );
	memcpy( pos + sizeof( Qinfo ), arg, size_ );
}
