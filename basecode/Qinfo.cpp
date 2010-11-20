/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#ifdef USE_MPI
#include <mpi.h>
#endif

// Declaration of static fields
bool Qinfo::isSafeForStructuralOps_ = 0;
vector< Qvec > Qinfo::q1_;
vector< Qvec > Qinfo::q2_;
vector< Qvec > Qinfo::mpiQ_;
vector< Qvec >* Qinfo::inQ_ = &Qinfo::q1_;
vector< Qvec >* Qinfo::outQ_ = &Qinfo::q2_;
vector< SimGroup > Qinfo::g_;
vector< vector< QueueBlock > > Qinfo::qBlock_;
vector< const char* > Qinfo::structuralQ_;

void hackForSendTo( const Qinfo* q, const char* buf );
static const unsigned int BLOCKSIZE = 20000;

Qinfo::Qinfo( FuncId f, DataId srcIndex, unsigned int size, bool useSendTo )
	:	
		useSendTo_( useSendTo ), 
		isForward_( 1 ), 
		isDummy_( 0 ), 
		m_( 0 ), 
		f_( f ), 
		srcIndex_( srcIndex ),
		size_( size )
{;}

Qinfo::Qinfo( DataId srcIndex, unsigned int size, bool useSendTo )
	:	
		useSendTo_( useSendTo ), 
		isForward_( 1 ), 
		isDummy_( 0 ), 
		m_( 0 ), 
		f_( 0 ), 
		srcIndex_( srcIndex ),
		size_( size )
{;}

Qinfo::Qinfo()
	:	
		useSendTo_( 0 ), 
		isForward_( 1 ), 
		isDummy_( 0 ), 
		m_( 0 ), 
		f_( 0 ), 
		srcIndex_( 0 ),
		size_( 0 )
{;}

/// Static function
Qinfo Qinfo::makeDummy( unsigned int size )
{
	Qinfo ret( 0, size, 0 ) ;
	ret.isDummy_ = 1;
	return ret;
}

/**
 * Static func: Sets up a SimGroup to keep track of thread and node
 * grouping info. This is used by the Qinfo to manage assignment of
 * threads and queues.
 * numThreads is the number of threads present in this group on this node.
 * Returns the group number of the new group.
 * have that because it will require updates to messages.
 */
unsigned int Qinfo::addSimGroup( unsigned short numThreads, 
	unsigned short numNodes )
{
	unsigned short ng = g_.size();
	unsigned short si = 0;
	if ( ng > 0 )
		si = g_[ng - 1].startThread + g_[ng - 1].numThreads;
	SimGroup sg( numThreads, si, numNodes );
	g_.push_back( sg );

	q1_.push_back( Qvec( numThreads ) );
	q2_.push_back( Qvec( numThreads ) );

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

/**
 * This is called within barrier1 of the ProcessLoop. It isn't
 * thread-safe, relies on the location of the call to achieve safety.
 */
void Qinfo::clearStructuralQ()
{
	isSafeForStructuralOps_ = 1;
	for ( vector< const char* >::iterator i = structuralQ_.begin(); 
		i != structuralQ_.end(); ++i ) {
		const char* buf = *i;
		const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
		const Msg *m = Msg::getMsg( q->mid() );
		if ( m ) { // may be 0 if model restructured before msg delivery.
			Element* tgt;
			if ( q->isForward() )
				tgt = m->e2();
			else 
				tgt = m->e1();
			assert( tgt == Id()() ); // The tgt should be the shell
			const OpFunc* func = tgt->cinfo()->getOpFunc( q->fid() );
			func->op( Eref( tgt, 0 ), buf );
		}
	}
	isSafeForStructuralOps_ = 0;
	structuralQ_.resize( 0 );
}

// local func
// Note that it does not advance the buffer.
void hackForSendTo( const Qinfo* q, const char* buf )
{
	const Msg *m = Msg::getMsg( q->mid() );
	if ( m ) { // may be 0 if model restructured before msg delivery.
		const DataId* tgtIndex = 
			reinterpret_cast< const DataId* >( buf + sizeof( Qinfo ) +
			q->size() - sizeof( DataId ) );
	
		Element* tgt;
		if ( q->isForward() )
			tgt = m->e2();
		else 
			tgt = m->e1();
		const OpFunc* func = tgt->cinfo()->getOpFunc( q->fid() );
		func->op( Eref( tgt, *tgtIndex ), buf );
	}
}

void readBuf(const Qvec& qv, const ProcInfo* proc )
{
	const char* buf = qv.data();
	// unsigned int bufsize = *reinterpret_cast< const unsigned int* >( buf );
	const char* end = buf + qv.dataQsize();
	// buf += sizeof( unsigned int );
	while ( buf < end )
	{
		const Qinfo *qi = reinterpret_cast< const Qinfo* >( buf );
		if ( !qi->isDummy() ) {
			if ( qi->useSendTo() ) {
				hackForSendTo( qi, buf );
			} else {
				const Msg* m = Msg::getMsg( qi->mid() );
				// m may be 0 if a Msg or Element has been cleared between
				// the sending of the Msg and its receipt.
				if ( m )
					m->exec( buf, proc );
			}
		}
		buf += sizeof( Qinfo ) + qi->size();
	}
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
	assert( proc->groupId < inQ_->size() );
	for ( unsigned int i = 0; i < inQ_->size(); ++i )
		readBuf( ( *inQ_ )[ i ], proc );
}

/**
 * Static func. 
 * Deliver the contents of the mpiQ to target objects
 * Assumes that the Msgs invoked by readBuf handle the thread safety.
 */
void Qinfo::readMpiQ( const ProcInfo* proc )
{
	/*
	assert( proc );
	assert( proc->groupId < mpiQ_.size() );
	const Qvec& q = mpiQ_[ proc->groupId ];
	for ( unsigned int i = 0; i < proc->numNodesInGroup; ++i ) {
		if ( i != proc->nodeIndexInGroup ) {
			const char* buf = q.data() + BLOCKSIZE * i;
			assert( q.allocatedSize() >= sizeof( unsigned int ) + BLOCKSIZE * i );
			const unsigned int *bufsize = 
				reinterpret_cast< const unsigned int* >( buf);
			if ( *bufsize > BLOCKSIZE ) { // Giant message
				// Do something.
				exit( 0 );
			} else {
				readBuf( buf, proc );
			}
		}
	}
	mpiQ_[proc->groupId].clear();
	*/
}

/**
 * Exchanges inQs and outQs. Also stitches together thread blocks on
 * the inQ so that the readBuf function will go through as a single 
 * unit.
 * Static func. Not thread safe. Must be done on single thread,
 * protected by barrier so that it happens at a defined point for all
 * threads.
 */
void Qinfo::swapQ()
{
	// This happens here protected by the barrier, so that operations that
	// change the structure of the model can occur without risk of 
	// affecting ongoing messaging.
	clearStructuralQ(); // static function.

	if ( inQ_ == &q2_ ) {
		inQ_ = &q1_;
		outQ_ = &q2_;
	} else {
		inQ_ = &q2_;
		outQ_ = &q1_;
	}
	for ( vector< Qvec >::iterator i = inQ_->begin(); 
		i != inQ_->end(); ++i )
	{
		i->stitch();
	}

	for ( vector< Qvec >::iterator i = outQ_->begin(); 
		i != outQ_->end(); ++i ) {
		i->clear();
	}
}

void Qinfo::swapMpiQ()
{
	// dummy func for now.
}

/**
 * Static func.
 * Used for simulation time data transfer. Symmetric across all nodes.
 *
 * the MPI::Alltoall function doesn't work here because it partitions out
 * the send buffer into pieces targetted for each other node. 
 * The Scatter fucntion does something similar, but it is one-way.
 * The Broadcast function is good. Sends just the one datum from source
 * to all other nodes.
 * For return we need the Gather function: the root node collects responses
 * from each of the other nodes.
 */
void Qinfo::sendAllToAll( const ProcInfo* proc )
{
	if ( proc->numNodesInGroup == 1 )
		return;
	// cout << proc->nodeIndexInGroup << ", " << proc->threadId << ": Qinfo::sendAllToAll\n";
	assert( mpiQ_[proc->groupId].allocatedSize() >= 
		BLOCKSIZE * proc->numNodesInGroup );
#ifdef USE_MPI
	const char* sendbuf = inQ_->data();
	char* recvbuf = mpiQ_.writableData();
	//assert ( inQ_[ proc->groupId ].size() == BLOCKSIZE );

	MPI_Barrier( MPI_COMM_WORLD );

	// Recieve data into recvbuf of all nodes from sendbuf of all nodes
	MPI_Allgather( 
		sendbuf, BLOCKSIZE, MPI_CHAR, 
		recvbuf, BLOCKSIZE, MPI_CHAR, 
		MPI_COMM_WORLD );
	// cout << "\n\nGathered stuff via mpi, on node = " << proc->nodeIndexInGroup << ", size = " << *reinterpret_cast< unsigned int* >( recvbuf ) << "\n";
#endif
}

void innerReportQ( const Qvec& qv, const string& name )
{
	if ( qv.totalNumEntries() == 0 )
		return;
	cout << endl << Shell::myNode() << ": Reporting " << name << 
		". threads=" << qv.numThreads() << ", sizes: "; 
	for ( unsigned int i = 0; i < qv.numThreads(); ++i ) {
		if ( i > 0 )
			cout << ", ";
		cout << qv.numEntries( i );
	}
	cout << endl;
	
	const char* buf = qv.data();
	const char* end = qv.data() + qv.dataQsize();
	while ( buf < end ) {
		const Qinfo *q = reinterpret_cast< const Qinfo* >( buf );
		if ( !q->isDummy() ) {
			const Msg *m = Msg::safeGetMsg( q->mid() );
			if ( m ) {
				cout << "Q::MsgId = " << q->mid() << 
					", FuncId = " << q->fid() <<
					", srcIndex = " << q->srcIndex() << 
					", size = " << q->size() <<
					", isForward = " << q->isForward() << 
					", e1 = " << m->e1()->getName() << 
					", e2 = " << m->e2()->getName() << endl;
			} else {
				cout << "Q::MsgId = " << q->mid() << " (points to bad Msg)" <<
					", FuncId = " << q->fid() <<
					", srcIndex = " << q->srcIndex() << 
					", size = " << q->size() << endl;
			}
		}
		buf += q->size() + sizeof( Qinfo );
	}
}

/**
 * Static func. readonly, so it is thread safe
 */
void Qinfo::reportQ()
{
	cout << Shell::myNode() << ":	inQ: ";
	for ( unsigned int i = 0; i < inQ_->size(); ++i )
		cout << "[" << i << "]=" << ( *inQ_ )[i].totalNumEntries() << "	";
	cout << "outQ: ";
	for ( unsigned int i = 0; i < outQ_->size(); ++i )
		cout << "[" << i << "]=" << (*outQ_ )[i].totalNumEntries() << "	";
	cout << "mpiQ: ";
	for ( unsigned int i = 0; i < mpiQ_.size(); ++i ) {
		unsigned int size = mpiQ_[i].totalNumEntries();
		cout << "[" << i << "]=" << size << "	";
	}
	cout << endl;

	if ( inQ_->size() > 0 ) innerReportQ( (*inQ_)[0], "inQ[0]" );
	if ( inQ_->size() > 1 ) innerReportQ( (*inQ_)[1], "inQ[1]" );
	if ( outQ_->size() > 0 ) innerReportQ( (*outQ_)[0], "outQ[0]" );
	if ( outQ_->size() > 1 ) innerReportQ( (*outQ_)[1], "outQ[1]" );
	if ( mpiQ_.size() > 0 ) innerReportQ( mpiQ_[0], "mpiQ[0]" );
	if ( mpiQ_.size() > 1 ) innerReportQ( mpiQ_[1], "mpiQ[1]" );
}

void Qinfo::addToQforward( const ProcInfo* p, MsgFuncBinding b, 
	const char* arg )
{
	m_ = b.mid;
	f_ = b.fid;
	isForward_ = 1;
	(*outQ_)[p->groupId].push_back( p->threadIndexInGroup, this, arg );
}

void Qinfo::addToQbackward( const ProcInfo* p, MsgFuncBinding b, 
	const char* arg )
{
	m_ = b.mid;
	f_ = b.fid;
	isForward_ = 0;
	(*outQ_)[p->groupId].push_back( p->threadIndexInGroup, this, arg );
}

/**
 * This is called by the 'handle<op>' functions in Shell. So it is always
 * either in phase2/3, or within clearStructuralQ() which is on a single 
 * thread inside Barrier1.
 * Note that the 'isSafeForStructuralOps' flag is only ever touched in 
 * clearStructuralQ().
 */
bool Qinfo::addToStructuralQ() const
{
	if ( isSafeForStructuralOps_ )
		return 0;
	structuralQ_.push_back( reinterpret_cast< const char* >( this ) );
	return 1;
}

void Qinfo::addSpecificTargetToQ( const ProcInfo* p, MsgFuncBinding b, 
	const char* arg, const DataId& target, bool isForward )
{
	m_ = b.mid;
	f_ = b.fid;
	isForward_ = isForward;
	char* temp = new char[ size_ + sizeof( DataId ) ];
	memcpy( temp, arg, size_ );
	memcpy( temp + size_, &target, sizeof( DataId ) );
	size_ += sizeof( DataId );
	(*outQ_)[p->groupId].push_back( p->threadIndexInGroup, this, temp );
	delete[] temp;
}

void Qinfo::emptyAllQs()
{
	for ( vector< Qvec >::iterator i = q1_.begin(); i != q1_.end(); ++i )
		i->clear();
	for ( vector< Qvec >::iterator i = q2_.begin(); i != q2_.end(); ++i )
		i->clear();
}

// Static function. Used only during single-thread tests, when the
// main thread-handling loop is inactive.
// Called after all the 'send' functions have been done. Typically just
// one is pending.
void Qinfo::clearQ( const ProcInfo* p )
{
	swapQ();
	readQ( p );
}
