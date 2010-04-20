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
vector< vector< char > > Qinfo::inQ_;
vector< vector< char > > Qinfo::outQ_;
vector< vector< char > > Qinfo::mpiQ_;
vector< SimGroup > Qinfo::g_;

void hackForSendTo( const Qinfo* q, const char* buf );
static const unsigned int BLOCKSIZE = 65536;

Qinfo::Qinfo( FuncId f, DataId srcIndex, 
	unsigned int size, bool useSendTo, bool isForward )
	:	
		useSendTo_( useSendTo ), 
		isForward_( isForward ), 
		m_( 0 ), 
		f_( f ), 
		srcIndex_( srcIndex ),
		size_( size )
{;}

Qinfo::Qinfo( FuncId f, DataId srcIndex, unsigned int size )
	:	
		useSendTo_( 0 ), 
		isForward_( 1 ), 
		m_( 0 ), 
		f_( f ), 
		srcIndex_( srcIndex ),
		size_( size )
{;}

Qinfo::Qinfo( bool useSendTo, bool isForward,
	DataId srcIndex, unsigned int size )
	:	
		useSendTo_( useSendTo ), 
		isForward_( isForward ), 
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

	inQ_.resize( g_.size() );

	mpiQ_.resize( g_.size() );
	mpiQ_.back().resize( BLOCKSIZE * numNodes );

	outQ_.resize( si + numThreads );
	for ( unsigned int i = 0; i < numThreads; ++i ) {
		outQ_[i + si].reserve( BLOCKSIZE );
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

void Qinfo::clearQ( const ProcInfo* proc )
{
	mergeQ( proc->groupId );
	if ( 0 ) {
		sendRootToAll( proc );
		readQ( proc );
		readMpiQ( proc );
	} else {
		readQ( proc );
	}
	inQ_[ proc->groupId ].resize( BLOCKSIZE );
	*reinterpret_cast< unsigned int* >( &inQ_[ proc->groupId ][0] ) = 0;
}

void Qinfo::mpiClearQ( const ProcInfo* proc )
{
	// cout << proc->nodeIndexInGroup << ": Qinfo::mpiClearQ: numNodes= " << proc->numNodesInGroup << "\n";
	mergeQ( proc->groupId );
	if ( proc->numNodesInGroup > 1 ) {
		sendRootToAll( proc );
		readQ( proc );
		readMpiQ( proc );
	} else {
		readQ( proc );
	}
	inQ_[ proc->groupId ].resize( BLOCKSIZE );
	*reinterpret_cast< unsigned int* >( &inQ_[ proc->groupId ][0] ) = 0;
}

void readBuf(const char* begin, const ProcInfo* proc )
{
	const char* buf = begin;
	// cout << "1";
	unsigned int bufsize = *reinterpret_cast< const unsigned int* >( buf );
	// cout << "2";
	/*
	if ( bufsize != 36 && proc->numNodesInGroup > 1 && proc->groupId == 0 )
		cout << "In readBuf on " << proc->nodeIndexInGroup << ", bufsize = " << bufsize << endl;
		*/
	const char* end = buf + bufsize;
	// cout << "3";
	buf += sizeof( unsigned int );
	// cout << "4" << flush;
	while ( buf && buf < end )
	{
		// cout << "5" << flush;
		const Qinfo *qi = reinterpret_cast< const Qinfo* >( buf );
		// cout << "6" << flush;
		if ( qi->useSendTo() ) {
			hackForSendTo( qi, buf );
		} else {
		// cout << "7: msg=" << qi->mid() << " " << flush;
			const Msg* m = Msg::getMsg( qi->mid() );
		// cout << "8" << flush;
			m->exec( buf, proc );
		// cout << "9" << flush;
		}
		buf += sizeof( Qinfo ) + qi->size();
		// cout << "a";
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
	assert( proc->groupId < inQ_.size() );
	vector< char >& q = inQ_[ proc->groupId ];
	assert( q.size() >= sizeof( unsigned int ) );
	readBuf( &q[0], proc );
}

/**
 * Static func. 
 * Deliver the contents of the mpiQ to target objects
 * Not thread safe. To run multithreaded, requires that
 * the messages have been subdivided on a per-thread basis to avoid 
 * overwriting each others targets.
 */
void Qinfo::readMpiQ( const ProcInfo* proc )
{
	assert( proc );
	assert( proc->groupId < mpiQ_.size() );
	vector< char >& q = mpiQ_[ proc->groupId ];
	/*
	cout << "in Qinfo::readMpiQ on node " << proc->nodeIndexInGroup <<
		", qsize = " << q.size() << endl;
	*/
	for ( unsigned int i = 0; i < proc->numNodesInGroup; ++i ) {
		if ( i != proc->nodeIndexInGroup ) {
			char* buf = &q[0] + BLOCKSIZE * i;
			assert( q.size() >= sizeof( unsigned int ) + BLOCKSIZE * i );
			readBuf( buf, proc );
			unsigned int *bufsize = reinterpret_cast< unsigned int* >( buf);
			/*
			if ( *bufsize > 0 ) {
				cout << "On (" << proc->nodeIndexInGroup << 
					", " << proc->threadIndexInGroup << 
					"): got msg of size " << *bufsize << endl;

			}
			*/
			*bufsize = 0;
		}
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
	assert( totSize + sizeof( unsigned int ) <= BLOCKSIZE );
	// inQ.resize( totSize + sizeof( unsigned int ) );
	inQ.resize( BLOCKSIZE );
	j = g.startThread;
	char* buf = &inQ[0];
	unsigned int *bufsize = reinterpret_cast< unsigned int* >( buf );
	*bufsize = 0;
//	*bufsize = inQ.size();
	buf += sizeof( unsigned int );
	for ( unsigned int i = 0; i < g.numThreads; ++i ) {
		memcpy( buf, &(outQ_[ j ][0]), outQ_[ j ].size() );
		buf += outQ_[ j ].size();
		*bufsize += outQ_[ j ].size();
		outQ_[ j ].resize( 0 );
		j++;
	}
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
	assert( mpiQ_[ proc->groupId ].size() >= BLOCKSIZE * proc->numNodesInGroup );
#ifdef USE_MPI
	char* sendbuf = &inQ_[ proc->groupId ][0];
	char* recvbuf = &mpiQ_[ proc->groupId ][0];
	assert ( inQ_[ proc->groupId ].size() == BLOCKSIZE );

	MPI_Barrier( MPI_COMM_WORLD );
	/*
	if ( proc->nodeIndexInGroup == 0 ) {
		int ret = MPI_Bcast( 
			sendbuf, BLOCKSIZE, MPI_CHAR, 0, MPI_COMM_WORLD );
	} else {
		MPI_Bcast( 
			recvbuf, BLOCKSIZE, MPI_CHAR, 0, MPI_COMM_WORLD );
	}
	*/

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
 */
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
	assert ( inQ_[ proc->groupId ].size() == BLOCKSIZE );
	/*
	MPI::COMM_WORLD.Alltoall( 
		sendbuf, BLOCKSIZE, MPI::CHAR,
		recvbuf, BLOCKSIZE, MPI::CHAR );
		*/
	// Send out data from master node.
		// cout << "\n\nEntering sendRootToAll barrier, on node = " << proc->nodeIndexInGroup << endl;
	MPI_Barrier( MPI_COMM_WORLD );
		// cout << "Exiting sendRootToAll barrier, on node = " << proc->nodeIndexInGroup << endl;
	if ( proc->nodeIndexInGroup == 0 ) {
		// cout << "\n\nSending stuff via mpi, on node = " << proc->nodeIndexInGroup << ", size = " << *reinterpret_cast< unsigned int* >( sendbuf ) << "\n";
		MPI_Bcast( 
			sendbuf, BLOCKSIZE, MPI_CHAR, 0, MPI_COMM_WORLD );
		// cout << "\n\nSent stuff via mpi, on node = " << proc->nodeIndexInGroup << ", ret = " << ret << endl;
	} else {
		// cout << "\n\nStarting Recv via mpi, on node = " << proc->nodeIndexInGroup << endl;
		MPI_Bcast( 
			recvbuf, BLOCKSIZE, MPI_CHAR, 0, MPI_COMM_WORLD );
		// cout << "\n\nRecvd stuff via mpi, on node = " << proc->nodeIndexInGroup << ", size = " << *reinterpret_cast< unsigned int* >( recvbuf ) << "\n";
	}

	// Recieve data into recvbuf of node0 from sendbuf of all other nodes
	MPI_Gather( 
		sendbuf, BLOCKSIZE, MPI_CHAR, 
		recvbuf, BLOCKSIZE, MPI_CHAR, 0, MPI_COMM_WORLD );
	// cout << "\n\nGathered stuff via mpi, on node = " << proc->nodeIndexInGroup << ", size = " << *reinterpret_cast< unsigned int* >( recvbuf ) << "\n";
#endif
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

	cout << "Reporting inQ[0]\n";
	const char* buf = &(inQ_[0][0]);
	unsigned int bufsize = *reinterpret_cast< const unsigned int* >( buf );
	const char* end = buf + bufsize;
	buf += sizeof( unsigned int );
	while ( buf < end ) {
		const Qinfo *q = reinterpret_cast< const Qinfo* >( buf );
		const Msg *m = Msg::getMsg( q->m_ );
		cout << "Q::MsgId = " << q->m_ << ", FuncId = " << q->f_ <<
			", srcIndex = " << q->srcIndex_ << ", size = " << q->size_ <<
			", src = " << m->e1()->name() << ", dest = " << m->e2()->name() << endl;
		buf += q->size() + sizeof( Qinfo );
	}
}

void Qinfo::addToQ( Qid qId, MsgFuncBinding b, const char* arg )
{
	assert( qId < outQ_.size() );

	vector< char >& q = outQ_[qId];
	unsigned int origSize = q.size();
	m_ = b.mid;
	f_ = b.fid;
	q.resize( origSize + sizeof( Qinfo ) + size_ );
	char* pos = &( q[origSize] );
	memcpy( pos, this, sizeof( Qinfo ) );
	// ( reinterpret_cast< Qinfo* >( pos ) )->setForward( isForward );
	// cout << Shell::myNode() << ": Qinfo::addToQ: size_ = " << size_ << endl;
	memcpy( pos + sizeof( Qinfo ), arg, size_ );
}

void Qinfo::addSpecificTargetToQ( Qid qId, MsgFuncBinding b, 
	const char* arg, const DataId& target )
{
	assert( qId < outQ_.size() );

	unsigned int temp = size_;
	// Expand local size to accommodate FullId for target of msg.
	size_ += sizeof( DataId );

	vector< char >& q = outQ_[qId];
	unsigned int origSize = q.size();
	m_ = b.mid;
	f_ = b.fid;
	q.resize( origSize + sizeof( Qinfo ) + size_ );
	char* pos = &( q[origSize] );
	memcpy( pos, this, sizeof( Qinfo ) );
	pos += sizeof( Qinfo );
	memcpy( pos, arg, temp );
	pos += temp;
	memcpy( pos, &target, sizeof( DataId ) );
}
