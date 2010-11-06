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

// Declaration of static field
vector< Qvec > Qinfo::q1_;
vector< Qvec > Qinfo::q2_;
vector< Qvec > Qinfo::mpiQ_;
vector< Qvec >* Qinfo::inQ_ = &Qinfo::q1_;
vector< Qvec >* Qinfo::outQ_ = &Qinfo::q2_;
vector< SimGroup > Qinfo::g_;
vector< vector< QueueBlock > > Qinfo::qBlock_;

void hackForSendTo( const Qinfo* q, const char* buf );
static const unsigned int BLOCKSIZE = 20000;

Qinfo::Qinfo( FuncId f, DataId srcIndex, unsigned int size, bool useSendTo )
	:	
		useSendTo_( useSendTo ), 
		isForward_( 1 ), 
		m_( 0 ), 
		f_( f ), 
		srcIndex_( srcIndex ),
		size_( size )
{;}

Qinfo::Qinfo( DataId srcIndex, unsigned int size, bool useSendTo )
	:	
		useSendTo_( useSendTo ), 
		isForward_( 1 ), 
		m_( 0 ), 
		f_( 0 ), 
		srcIndex_( srcIndex ),
		size_( size )
{;}

Qinfo::Qinfo()
	:	
		useSendTo_( 0 ), 
		isForward_( 1 ), 
		m_( 0 ), 
		f_( 0 ), 
		srcIndex_( 0 ),
		size_( 0 )
{;}

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

	/*
	inQ_.resize( g_.size() );

	mpiQ_.resize( g_.size() );
	mpiQ_.back().resize( BLOCKSIZE * numNodes );

	outQ_.resize( si + numThreads );
	qBlock_.resize( si + numThreads );
	for ( unsigned int i = 0; i < numThreads; ++i ) {
		outQ_[i + si].reserve( BLOCKSIZE );
	}
	*/
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

// local func
// Note that it does not advance the buffer.
void hackForSendTo( const Qinfo* q, const char* buf )
{
	const DataId* tgtIndex = 
		reinterpret_cast< const DataId* >( buf + sizeof( Qinfo ) +
		q->size() - sizeof( DataId ) );

	Element* tgt;
	if ( q->isForward() )
		tgt = Msg::getMsg( q->mid() )->e2();
	else 
		tgt = Msg::getMsg( q->mid() )->e1();
	const OpFunc* func = tgt->cinfo()->getOpFunc( q->fid() );
	func->op( Eref( tgt, *tgtIndex ), buf );
}

void readBuf(const char* begin, const ProcInfo* proc )
{
	const char* buf = begin;
	unsigned int bufsize = *reinterpret_cast< const unsigned int* >( buf );
	/*
	if ( bufsize != 36 && proc->numNodesInGroup > 1 && proc->groupId == 0 )
		cout << "In readBuf on " << proc->nodeIndexInGroup << ", bufsize = " << bufsize << endl;
		*/
	const char* end = buf + bufsize;
	buf += sizeof( unsigned int );
	while ( buf < end )
	{
		const Qinfo *qi = reinterpret_cast< const Qinfo* >( buf );
		if ( qi->useSendTo() ) {
			hackForSendTo( qi, buf );
		} else {
			const Msg* m = Msg::getMsg( qi->mid() );
			assert( m );
			m->exec( buf, proc );
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
	readBuf( ( *inQ_ )[ proc->groupId ].data(), proc );
}

/**
 * Static func. 
 * Deliver the contents of the mpiQ to target objects
 * Assumes that the Msgs invoked by readBuf handle the thread safety.
 */
void Qinfo::readMpiQ( const ProcInfo* proc )
{
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
}

/**
 * Static func. 
 * Deliver the contents of the mpiQ to target objects,
 * used in sending data between roots.
 * Not thread safe. To run multithreaded, requires that
 * the messages have been subdivided on a per-thread basis to avoid 
 * overwriting each others targets.
void Qinfo::readRootQ( const ProcInfo* proc )
{
	assert( proc );
	assert( proc->groupId < mpiQ_.size() );
	vector< char >& q = mpiQ_[ 0 ];

	unsigned int *bufsize = reinterpret_cast< unsigned int* >( &q[0] );
	readBuf( &q[0], proc );
	*bufsize = 0;
	// cout << Shell::myNode() << ": readRootQ\n";
	// Qinfo::reportQ();
	return;
}
 */

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

	/*
	assert( groupId < g_.size() );
	SimGroup& g = g_[ groupId ];
	// unsigned int j = g.startThread;
	// assert( j + g.numThreads <= outQ_.size() );
	const Qvec& inQ = (*inQ_)[ groupId ];

	inQ.resize( sizeof( unsigned int ) );
	*( reinterpret_cast< unsigned int* >( &inQ[0] ) ) = 0;
	localQ_.resize( sizeof( unsigned int ) );
	*( reinterpret_cast< unsigned int* >( &localQ_[0] ) ) = 0;
	for ( unsigned int i = 0; i < g.numThreads; ++i ) {
	//	unsigned int outQindex = i + g.startThread;
		unsigned int outQindex = i;
		vector< QueueBlock >& qb = qBlock_[outQindex];
		vector< char >::iterator begin = outQ_[outQindex].begin();
		for ( unsigned int j = 0; j < qb.size(); ++j ) {
			if ( qb[j].whichQ == 0 ) {
				inQ.insert( inQ.end(), begin + qb[j].startOffset, 
					begin + qb[j].startOffset + qb[j].size );
			} else {
				localQ_.insert( localQ_.end(), begin + qb[j].startOffset, 
					begin + qb[j].startOffset + qb[j].size );
			}
		}
		outQ_[outQindex].resize( 0 );
		qb.resize( 0 );
	}
	*reinterpret_cast< unsigned int* >( &inQ[0] ) = inQ.size();
	*reinterpret_cast< unsigned int* >( &localQ_[0] ) = localQ_.size();
	*/
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

/**
 * Static func.
 * Here the root node tells all other nodes what to do, using a Bcast.
 * It then reads back all their responses. The function is meant to
 * be run on all nodes, and it partitions out the work according to
 * node#.
 * The function is used only by the Shell thread.
 * the MPI::Alltoall function doesn't work here because it partitions out
 * the send buffer into pieces targetted for each other node. 
void Qinfo::sendRootToAll( const ProcInfo* proc )
{
	if ( proc->numNodesInGroup == 1 )
		return;
	// cout << proc->nodeIndexInGroup << ", " << proc->threadId << ": Qinfo::sendRootToAll\n";
	// cout << "ng = " << g_.size() << ", ninQ= " << inQ_[0].size() << ", nmpiQ = " << mpiQ_[0].size() << " proc->groupId =  " << proc->groupId  << " s1 = " << mpiQ_[ proc->groupId ].size() << " s2 = " << BLOCKSIZE * proc->numNodesInGroup;
	assert( mpiQ_[ proc->groupId ].size() >= BLOCKSIZE * proc->numNodesInGroup );
#ifdef USE_MPI
	char* sendbuf = &inQ_[ proc->groupId ][0];
	char* recvbuf = &mpiQ_[ proc->groupId ][0];
//	assert ( inQ_[ proc->groupId ].size() == BLOCKSIZE );
	assert ( inQ_[ proc->groupId ].size() >= sizeof( unsigned int ) );
	// Send out data from master node.
		// cout << "\n\nEntering sendRootToAll barrier, on node = " << proc->nodeIndexInGroup << endl;
	MPI_Barrier( MPI_COMM_WORLD );
		// cout << "Exiting sendRootToAll barrier, on node = " << proc->nodeIndexInGroup << endl;
	if ( proc->nodeIndexInGroup == 0 ) {
		// cout << "\n\nSending stuff via mpi, on node = " << proc->nodeIndexInGroup << ", size = " << *reinterpret_cast< unsigned int* >( sendbuf ) << "\n";
		MPI_Bcast( 
			sendbuf, BLOCKSIZE, MPI_CHAR, 0, MPI_COMM_WORLD );
		// cout << "\n\nSent stuff via mpi, on node = " << proc->nodeIndexInGroup << ", ret = " << ret << endl;
		unsigned int bufsize = *( reinterpret_cast< unsigned int* >( sendbuf ) );
		if ( bufsize > BLOCKSIZE ) {
			// cout << Shell::myNode() << "." << proc->threadIndexInGroup << ": Sending Large MPI_Bcast of size = " << bufsize << endl;
			MPI_Bcast( 
				sendbuf, bufsize, MPI_CHAR, 0, MPI_COMM_WORLD );
		}
	} else {
		// cout << "\n\nStarting Recv via mpi, on node = " << proc->nodeIndexInGroup << endl;
		MPI_Bcast( 
			recvbuf, BLOCKSIZE, MPI_CHAR, 0, MPI_COMM_WORLD );
		// cout << "\n\nRecvd stuff via mpi, on node = " << proc->nodeIndexInGroup << ", size = " << *reinterpret_cast< unsigned int* >( recvbuf ) << "\n";
		unsigned int bufsize = *( reinterpret_cast< unsigned int* >( recvbuf ) );
		if ( bufsize > BLOCKSIZE ) {
			// cout << Shell::myNode() << "." << proc->threadIndexInGroup << ": Recv Large MPI_Bcast of size = " << bufsize << " on group " << proc->groupId << endl;
			mpiQ_[ proc->groupId ].resize( bufsize );
			recvbuf = &mpiQ_[ proc->groupId ][0];
			MPI_Bcast( 
				recvbuf, bufsize, MPI_CHAR, 0, MPI_COMM_WORLD );
		}
	}

//	unsigned int* sendbufSize = reinterpret_cast< unsigned int* >( sendbuf);
//	unsigned int* recvbufSize = reinterpret_cast< unsigned int* >( recvbuf);

	// Recieve data into recvbuf of node0 from sendbuf of all other nodes
	// cout << Shell::myNode() << "." << proc->threadIndexInGroup << ": About to Gather stuff via mpi, recvbufsize = " << *recvbufSize << ", sendbufsize = " << *sendbufSize << "\n";
	if ( proc->nodeIndexInGroup == 0 ) {
		static char dummySendbuf[BLOCKSIZE];
		*reinterpret_cast< unsigned int* >( dummySendbuf ) = 4;
		MPI_Gather( 
			dummySendbuf, BLOCKSIZE, MPI_CHAR, 
			recvbuf, BLOCKSIZE, MPI_CHAR, 0, MPI_COMM_WORLD );
	} else {
		MPI_Gather( 
			sendbuf, BLOCKSIZE, MPI_CHAR, 
			recvbuf, BLOCKSIZE, MPI_CHAR, 0, MPI_COMM_WORLD );
	}
	// cout << Shell::myNode() << "." << proc->threadIndexInGroup << ": Gathered stuff via mpi, recvbufsize = " << *recvbufSize << ", sendbufsize = " << *sendbufSize << "\n";
#endif
}
*/

void innerReportQ( const Qvec& qv, const string& name )
{
	cout << Shell::myNode() << ": Reporting " << name << ". size=" <<
		qv.dataQsize() << endl;
	
	const char* buf = qv.data();
	const char* end = qv.data() + qv.dataQsize();
	while ( buf < end ) {
		const Qinfo *q = reinterpret_cast< const Qinfo* >( buf );
		const Msg *m = Msg::safeGetMsg( q->mid() );
		if ( m ) {
			cout << "Q::MsgId = " << q->mid() << 
				", FuncId = " << q->fid() <<
				", srcIndex = " << q->srcIndex() << 
				", size = " << q->size() <<
				", src = " << m->e1()->getName() << 
				", dest = " << m->e2()->getName() << endl;
		} else {
			cout << "Q::MsgId = " << q->mid() << " (points to bad Msg)" <<
				", FuncId = " << q->fid() <<
				", srcIndex = " << q->srcIndex() << 
				", size = " << q->size() << endl;
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
		cout << "[" << i << "]=" << ( *inQ_ )[i].dataQsize() << "	";
	cout << "outQ: ";
	for ( unsigned int i = 0; i < outQ_->size(); ++i )
		cout << "[" << i << "]=" << (*outQ_ )[i].dataQsize() << "	";
	cout << "mpiQ: ";
	for ( unsigned int i = 0; i < mpiQ_.size(); ++i ) {
		unsigned int size = mpiQ_[i].dataQsize();
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

void Qinfo::addToQ( const ProcInfo* p, MsgFuncBinding b, 
	const char* arg )
{
	m_ = b.mid;
	f_ = b.fid;
	(*outQ_)[p->groupId].push_back( p->threadIndexInGroup, this, arg );
}

void Qinfo::addSpecificTargetToQ( const ProcInfo* p, MsgFuncBinding b, 
	const char* arg, const DataId& target )
{
	m_ = b.mid;
	f_ = b.fid;
	char* temp = new char[ size_ + sizeof( DataId ) ];
	memcpy( temp, arg, size_ );
	memcpy( temp + size_, &target, sizeof( DataId ) );
	size_ += sizeof( DataId );
	(*outQ_)[p->groupId].push_back( p->threadIndexInGroup, this, arg );
	delete[] temp;
}

/**
 * Three blocks: 
 * 0 is inQ, going to all nodes in group
 * 1 is local, going only to current node.
 * 2 and higher are to other simGroups. Don't worry about yet
void Qinfo::assignQblock( const Msg* m, const ProcInfo* p )
{
	unsigned int threadIndex = p->threadId;
	unsigned int offset = outQ_[ threadIndex ].size();
	vector< QueueBlock >& qb = qBlock_[ threadIndex ];
	if ( // Figure out if msg should go in local queue.
		m->mid() == Msg::setMsg ||
		( 
			( m->mid() != 2 ) && // mid of 2 is between shells on diff nodes
			(
				( isForward_ && m->e2()->dataHandler()->isGlobal() )  ||
				( !isForward_ && m->e1()->dataHandler()->isGlobal() )
			)
		)
	) { // Put in queue 1, which is localQ.
		if ( qb.size() > 0 && qb.back().whichQ == 1 ) { // Extend qb.back
			qb.back().size += size_ + sizeof( Qinfo );
		} else {
			qb.push_back( QueueBlock( 1, offset, size_ + sizeof( Qinfo ) ));
		}
	} else { // Put in queue 0, which is inQ, which goes to other nodes.
		if ( qb.size() > 0 && qb.back().whichQ == 0 ) { // Extend qb.back
			qb.back().size += size_ + sizeof( Qinfo );
		} else {
			qb.push_back( QueueBlock( 0, offset, size_ + sizeof( Qinfo ) ));
		}
	}
}
 */

void Qinfo::emptyAllQs()
{
	for ( vector< Qvec >::iterator i = q1_.begin(); i != q1_.end(); ++i )
		i->clear();
	for ( vector< Qvec >::iterator i = q2_.begin(); i != q2_.end(); ++i )
		i->clear();
}

/*
void Qinfo::assembleOntoQ( const MsgFuncBinding& i, 
	const Element* e, const ProcInfo *p, const char* arg )
{
	const Msg* m = Msg::getMsg( i.mid );
	isForward_ = m->isForward( e );
	if ( m->isMsgHere( *this ) ) {
		assignQblock( m, p );
		addToQ( p->threadId, i, arg );
	}
}
*/

// Static function. Dummy for now.
void Qinfo::clearQ( const ProcInfo* p )
{
	;
}

// Static function. Deprecated. Dummy for now.
void Qinfo::mpiClearQ( const ProcInfo* p )
{
	;
}
