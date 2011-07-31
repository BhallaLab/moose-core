/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
// #include "ReduceFinfo.h"
#include "../shell/Shell.h"
/*
#ifdef USE_MPI
#include <mpi.h>
#endif
*/

// Declaration of static fields
bool Qinfo::isSafeForStructuralOps_ = 0;
vector< double > Qinfo::q0_( 1000, 0.0 );
vector< double > Qinfo::q1_( 1000, 0.0 );
double* Qinfo::inQ_ = &Qinfo::q0_[0];
double* Qinfo::outQ_ = &Qinfo::q1_[0];
vector< vector< ReduceBase* > > Qinfo::reduceQ_;

vector< double > Qinfo::mpiQ1_( 1000, 0.0 );
vector< double > Qinfo::mpiQ2_( 1000, 0.0 );
double* Qinfo::mpiInQ_ = &mpiQ1_[0];
double* Qinfo::mpiRecvQ_ = &mpiQ2_[0];

// vector< SimGroup > Qinfo::g_;
vector< Qinfo > Qinfo::structuralQinfo_;
vector< double > Qinfo::structuralQdata_;

// void hackForSendTo( const Qinfo* q, const char* buf );
static const unsigned int BLOCKSIZE = 20000;


/// Default dummy Qinfo creation.
Qinfo::Qinfo()
	:	
		src_(),
		msgBindIndex_(0),
		threadNum_( 0 ),
		dataIndex_( 0 )
{;}

Qinfo::Qinfo( const ObjId& src, 
	BindIndex bindIndex, unsigned short threadNum,
	unsigned int dataIndex )
	:	
		src_( src ),
		msgBindIndex_( bindIndex ),
		threadNum_( threadNum ),
		dataIndex_( dataIndex )
{;}

/**
 * Deprecated
 * Static func: Sets up a SimGroup to keep track of thread and node
 * grouping info. This is used by the Qinfo to manage assignment of
 * threads and queues.
 * numThreads is the number of threads present in this group on this node.
 * Returns the group number of the new group.
 * have that because it will require updates to messages.
 * Returns number of new sim group.
unsigned int Qinfo::addSimGroup( unsigned short numThreads, 
	unsigned short numNodes )
{
	const unsigned int DefaultMpiBufSize = 1000; // bytes.
	// For starters, just put this group on all nodes.
	SimGroup sg( 0, numNodes, DefaultMpiBufSize );
	g_.push_back( sg );

	Qvec qv( numThreads );

	q1_.push_back( qv );
	q2_.push_back( qv );

	return g_.size() - 1;
}
 */

/*
// Static function, deprecated
unsigned int Qinfo::numSimGroup()
{
	return g_.size();
}

// Static function, deprecated
const SimGroup* Qinfo::simGroup( unsigned int index )
{
	assert( index < g_.size() );
	return &( g_[index] );
}

// Static function, deprecated
void Qinfo::clearSimGroups()
{
	g_.clear();
}
*/

/**
 * This is called within barrier1 of the ProcessLoop. It isn't
 * thread-safe, relies on the location of the call to achieve safety.
 */
void Qinfo::clearStructuralQ()
{
	isSafeForStructuralOps_ = 1;
	double* buf = &structuralQdata_[0];
	for ( unsigned int i = 0; i < structuralQinfo_.size(); ++i ) {
		const Qinfo* qi = &structuralQinfo_[i];
		if ( !qi->isDummy() ) {
			const Element* e = qi->src_.element();
			e->exec( qi, Shell::procInfo(), buf + qi->dataIndex_ );
		}
	}
	/*
	const double* begin = &( structuralQ_[0] );
	const double* end = begin + structuralQ_.size();
	const double* buf = begin;
	while ( buf < end ) {
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
		buf += sizeof( Qinfo ) + q->size();
	}
	*/

	isSafeForStructuralOps_ = 0;
	structuralQinfo_.resize( 0 );
	structuralQdata_.resize( 0 );
}

/*
// local func. Deprecated.
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
*/

void readBuf(const double* buf, const ProcInfo* proc )
{
	/*
	assert( qv.allocatedSize() >= Qvec::HeaderSize &&
		qv.mpiArrivedDataSize() >= Qvec::HeaderSize );
	const char* buf = qv.data();
	unsigned int bufsize = qv.mpiArrivedDataSize() - Qvec::HeaderSize;
	*/
	unsigned int bufsize = static_cast< unsigned int >( buf[0] );
	unsigned int numQinfo = static_cast< unsigned int >( buf[1] );
	assert( bufsize > numQinfo * 3 );

	const double* qptr = buf + 2;

	for ( unsigned int i = 0; i < numQinfo; ++i ) {
		const Qinfo* qi = reinterpret_cast< const Qinfo* >( qptr );
		qptr += 3;
		if ( !qi->isDummy() ) {
			const Element* e = qi->src().element();
			e->exec( qi, proc, buf + qi->dataIndex() );
		}
	}

	

	////////////////////////////////////////////////////////////////////

	/*
	const char* end = buf + bufsize;
	Qinfo::doMpiStats( bufsize, proc ); // Need zone info too.

	// If on current node, then check bufsize
	// assert( srcNode != Shell::myNode() || bufsize == qv.dataQsize() );

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
		if ( qi->size() % 2 != 0 || qi->size() > 2000000 || qi->mid() == Msg::badMsg ) {
			// Qinfo::reportQ();
			cout << proc->nodeIndexInGroup << "." << 
				proc->threadIndexInGroup << ": readBuf is odd sized: " <<
				qi->size() << ", MsgId = " << qi->mid() << endl;
			break;
		}
	}
	*/
}

/**
 * Static func.  Placeholder for now.
 * Will want to analyze data traffic and periodically tweak buffer
 * sizes
 */
void Qinfo::doMpiStats( unsigned int bufsize, const ProcInfo* proc )
{
}

/** 
 * Static func
 * In this variant we just go through the specified queue. 
 * Thread safety is fragile here. We require that all the exec functions
 * on the Element or on the Msg, should internally guarantee that they
 * will dispatch calls only to the ObjIds that are allowed for the
 * running thread.
 */ 
void Qinfo::readQ( const ProcInfo* proc )
{
	assert( proc );
	readBuf( inQ_, proc );
	/*
	assert( proc->groupId < inQ_->size() );
	for ( unsigned int i = 0; i < inQ_->size(); ++i )
		readBuf( ( *inQ_ )[ i ], proc );
		*/
}

/**
 * Static func. 
 * Deliver the contents of the mpiQ to target objects
 * Assumes that the Msgs invoked by readBuf handle the thread safety.
 * Checks the data block size in case it is over the regular
 * blocksize allocated for the current node.
 */
void Qinfo::readMpiQ( const ProcInfo* proc, unsigned int node )
{
	assert( proc );
	// assert( proc->groupId < mpiQ_.size() );
	// if ( mpiInQ_->mpiArrivedDataSize() > Qvec::HeaderSize )
		// cout << Shell::myNode() << ":" << proc->threadIndexInGroup << "	mpi data transf = " << mpiInQ_->mpiArrivedDataSize() << endl;

	readBuf( mpiInQ_, proc );

	/*
	if ( proc->nodeIndexInGroup == 0 && mpiInQ_->isBigBlock() ) 
		// Clock::requestBigBlock( proc->groupId, node );
		;
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
	/**
	 * clearStructuralQ happens here protected by the barrier, so that 
	 * operations that
	 * change the structure of the model can occur without risk of 
	 * affecting ongoing messaging.
	 */
	clearStructuralQ(); // static function.

	/**
	 * Here we just deposit all the data from the data and Q vectors into
	 * the inQ.
	 */
	unsigned int bufsize = 0;
	unsigned int datasize = 0;
	unsigned int numQinfo = 0;
	for ( unsigned int i = 0; i < Shell::numCores(); ++i ) {
		numQinfo += qBuf_[i].size();
		datasize += dBuf_[i].size();
	}

	bufsize = numQinfo * 3 + datasize + 2;
	if ( bufsize > q0_.capacity() )
		q0_.reserve( bufsize + bufsize / 2 );
	q0_.resize( bufsize );
	inQ_ = &q0_[0];
	q0_[0] = bufsize;
	q0_[1] = numQinfo;
	double* qptr = &q0_[2];
	unsigned int prevQueueDataIndex = numQinfo * 3 + 2;
	for ( unsigned int i = 0; i < Shell::numCores(); ++i ) {
		for ( vector< Qinfo >::iterator j = qBuf_[i].begin(); 
			j != qBuf_[i].end(); ++j ) {
			Qinfo* qi = reinterpret_cast< Qinfo* >( qptr );
			*qi = *j;
			qi->dataIndex_ += prevQueueDataIndex;
			qptr += 3;
		}
		prevQueueDataIndex += dBuf_[i].size();
	}
	assert( prevQueueDataIndex == bufsize );

	/*
	if ( inQ_ == &q2_[0] ) {
		inQ_ = &q1_[0];
		outQ_ = &q2_[0];
	} else {
		inQ_ = &q2_[0];
		outQ_ = &q1_[0];
	}

	// Clear outQ: set all entries to dummies, as the least-effort way.
	unsigned int bufsize = static_cast< unsigned int >( buf[0] );
	unsigned int numQinfo = static_cast< unsigned int >( outQ_[1] );
	assert( bufsize > numQinfo * 3 );
	double* qptr = outQ_ + 2;
	for ( unsigned int i = 0; < numQinfo; ++i ) {
		Qinfo* qi = reinterpret_cast< Qinfo* >( qptr );
		qi->setDummy();
		qptr += 3;
	}
	*/

	// Here we also need to do the size adjustment, dealing with overflow,
	// and so on.

	/*
	for ( vector< Qvec >::iterator i = inQ_->begin(); 
		i != inQ_->end(); ++i )
	{
		i->stitch();
	}

	for ( vector< Qvec >::iterator i = outQ_->begin(); 
		i != outQ_->end(); ++i ) {
		i->clear();
	}
	*/
}

/**
 * Need to allocate space for incoming block
 */
void Qinfo::swapMpiQ()
{
	if ( mpiInQ_ == &mpiQ2_[0] ) {
		mpiInQ_ = &mpiQ1_[0];
		mpiRecvQ_ = &mpiQ2_[0];
	} else {
		mpiInQ_ = &mpiQ2_[0];
		mpiRecvQ_ = &mpiQ1_[0];
	}
	// mpiRecvQ_->resizeLinearData( BLOCKSIZE );
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
/*
	if ( proc->numNodesInGroup == 1 )
		return;
	// cout << proc->nodeIndexInGroup << ", " << proc->threadId << ": Qinfo::sendAllToAll\n";
	assert( mpiRecvQ_->allocatedSize() >= BLOCKSIZE );
#ifdef USE_MPI
	assert( inQ_.size() == mpiQ_.size() );
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
*/
}

/*
void innerReportQ( const Qvec& qv, const string& name )
{
	bool isMpi = ( name.substr( 0, 3 ) == "mpi" );
	const char* buf = 0;
	const char* end = 0;
	if ( isMpi ) {
		if ( qv.allocatedSize() > Qvec::HeaderSize ) {
			cout << endl << Shell::myNode() << ": Reporting " << name << 
				". Size = " << qv.dataQsize() << endl; 
			buf = qv.data();
			end = qv.data() + qv.dataQsize();
		} else {
			cout << endl << Shell::myNode() << ": Reporting " << name << 
				". Size = " << 0 << endl; 
		}
	} else {
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
	
		Qvec temp( qv );
		temp.stitch(); // This is a bit of a hack. The qv.data etc are not
		// valid till stitch is called. I can't touch qv, and in any case
		// I should not, since it might invalidate pointers. So we copy it
		// to a temporary.
		buf = temp.data();
		end = temp.data() + temp.dataQsize();
	}

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
*/

void innerReportQ( const double* q, const string& name )
{
	cout << endl << Shell::myNode() << ": Reporting " << name << endl;
	unsigned int numQinfo = q[1];
	const double* qptr = q + 2;
	for ( unsigned int i = 0; i < numQinfo; ++i ) {
		const Qinfo* qi = reinterpret_cast< const Qinfo* >( qptr );
		cout << "src= " << qi->src() << 
			", msgBind=" << qi->bindIndex() <<
			", threadNum=" << qi->threadNum() <<
			", dataIndex=" << qi->dataIndex() << endl;
		qptr += 3;
	}
}

void reportOutQ( const vector< vector< Qinfo > >& q, const string& name )
{
	cout << endl << Shell::myNode() << ": Reporting " << name << endl;
	for ( unsigned int i = 0; i < Shell::numCores(); ++i ) {
		for ( vector< Qinfo >::const_iterator qi = q[i].begin(); 
			qi != q[i].end(); ++qi ) {
			cout << i << ": src= " << qi->src() <<
				", msgBind=" << qi->bindIndex() <<
				", threadNum=" << qi->threadNum() <<
				", dataIndex=" << qi->dataIndex() << endl;
		}
	}
}


/**
 * Static func. readonly, so it is thread safe
 */
void Qinfo::reportQ()
{
	cout << Shell::myNode() << ":	inQ: size =  " << inQ_[0] << 
		", numQinfo = " << inQ_[1] << endl;
	cout << Shell::myNode() << ":	mpiInQ: size =  " << mpiInQ_[0] << 
		", numQinfo = " << mpiInQ_[1] << endl;
	cout << Shell::myNode() << ":	mpiRecvQ: size =  " << mpiRecvQ_[0] << 
		", numQinfo = " << mpiRecvQ_[1] << endl;

	unsigned int datasize = 0;
	unsigned int numQinfo = 0;
	for ( unsigned int i = 0; i < Shell::numCores(); ++i ) {
		numQinfo += qBuf_[i].size();
		datasize += dBuf_[i].size();
	}
	cout << Shell::myNode() << ":	outQ: numCores = " << 
		Shell::numCores() << ", size = " << datasize + 3 * numQinfo << 
		", numQinfo = " << numQinfo << endl;

	if ( inQ_[1] > 0 ) innerReportQ(inQ_, "inQ" );
	if ( numQinfo > 0 ) reportOutQ( qBuf_, "outQ" );
	if ( mpiInQ_[1] > 0 ) innerReportQ( mpiInQ_, "mpiInQ" );
	if ( mpiRecvQ_[1] > 0 ) innerReportQ( mpiRecvQ_, "mpiRecvQ" );
}


/*
pthread_mutex_t* init_mutex()
{
	static pthread_mutex_t* mutex = new pthread_mutex_t;
	pthread_mutex_init( mutex, NULL );
	cout << "inititing mutex\n";

	return mutex;
}
*/

/**
 * Static function.
 * Adds a Queue entry.
 */
void Qinfo::addToQ( const ObjId& oi, 
	BindIndex bindIndex, unsigned short threadNum,
	const double* arg, unsigned int size )
{
	qBuf_[ threadNum ].push_back( 
		Qinfo( oi, bindIndex, threadNum, dBuf_.size() ) );
	if ( size > 0 ) {
		vector< double >& vec = dBuf_[ threadNum ];
		vec.insert( vec.end(), arg, arg + size );
	}
}


/**
 * Static function.
 * Adds a Queue entry. This variant allocates the space and returns
 * the pointer into which the data can be copied. Useful when there are
 * multiple fields to be put in.
 */
double* Qinfo::addToQ( const ObjId& oi, 
	BindIndex bindIndex, unsigned short threadNum,
	unsigned int size )
{
	unsigned int oldSize = dBuf_.size();
	qBuf_[ threadNum ].push_back( 
		Qinfo( oi, bindIndex, threadNum, oldSize ) );
	dBuf_[ threadNum ].resize( oldSize + size );
	return &( dBuf_[threadNum][oldSize] );
}

// Static function
void Qinfo::addDirectToQ( const ObjId& src, const ObjId& dest, 
	unsigned short threadNum,
	FuncId fid, 
	const double* arg, unsigned int size )
{
	static const unsigned int ObjFidSizeInDoubles = 
		1 + ( sizeof( ObjFid ) - 1 ) / sizeof( double );

	qBuf_[ threadNum ].push_back( 
		Qinfo( src, ~0, threadNum, dBuf_.size() ) );
	ObjFid ofid = { dest, fid, 0, 0 };
	const double* ptr = reinterpret_cast< const double* >( &ofid );
	vector< double >& vec = dBuf_[ threadNum ];
	vec.insert( vec.end(), ptr, ptr + ObjFidSizeInDoubles );
	if ( size > 0 ) {
		vec.insert( vec.end(), arg, arg + size );
	}
}

/// Static func.
void Qinfo::disableStructuralQ()
{
	isSafeForStructuralOps_ = 1;
}

/// static func
void Qinfo::enableStructuralQ()
{
	isSafeForStructuralOps_ = 0;
}

/**
 * This is called by the 'handle<op>' functions in Shell. So it is always
 * either in phase2/3, or within clearStructuralQ() which is on a single 
 * thread inside Barrier1.
 * Note that the 'isSafeForStructuralOps' flag is only ever touched in 
 * clearStructuralQ(), except in the unit tests.
 */
bool Qinfo::addToStructuralQ( const double* data, unsigned int size ) const
{
	if ( isSafeForStructuralOps_ )
		return 0;
	structuralQinfo_.push_back( *this );
	// unsigned int di = structuralQinfo_.back().dataIndex();
	structuralQinfo_.back().dataIndex_ = structuralQdata_.size();
	structuralQdata_.insert( structuralQdata_.end(), data, data + size );

	/*
	const char* begin = reinterpret_cast< const char* >( this );
	const char* end = begin + sizeof( Qinfo ) + size_;
	structuralQ_.insert( structuralQ_.end(), begin, end );
	*/
	// structuralQ_.push_back( reinterpret_cast< const char* >( this ) );
	return 1;
}

/*
void Qinfo::addSpecificTargetToQ( const ProcInfo* p, MsgFuncBinding b, 
	const char* arg, const DataId& target, bool isForward )
{
	// if ( !isForward )
		cout << p->groupId << ":" << p->threadIndexInGroup << " " << 
			b.mid << ", " << size_ << endl << flush;
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
*/

/// Static func
void Qinfo::emptyAllQs()
{
	inQ_[0] = 0;
	inQ_[1] = 0;
	for ( unsigned int i = 0; i < Shell::numCores(); ++i ) {
		qBuf_[i].resize( 0 );
		dBuf_[i].resize( 0 );
	}

	/*
	for ( vector< Qvec >::iterator i = q1_.begin(); i != q1_.end(); ++i )
		i->clear();
	for ( vector< Qvec >::iterator i = q2_.begin(); i != q2_.end(); ++i )
		i->clear();
	*/
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

/*
// static func
char* Qinfo::inQ( unsigned int group )
{
	assert( group < inQ_->size() );
	return (*inQ_)[ group ].writableData();
}

// static func
unsigned int Qinfo::inQdataSize( unsigned int group )
{
	assert( group < inQ_->size() );
	return (*inQ_)[ group ].mpiArrivedDataSize();
}

// static func
char* Qinfo::mpiRecvQbuf()
{
	return mpiRecvQ_->writableData();
}

// static func
void Qinfo::expandMpiRecvQbuf( unsigned int size )
{
	mpiRecvQ_->resizeLinearData( size );
}
*/

// static func. Typically only called during setup in Shell::setHardware.
void Qinfo::initMpiQs()
{
	mpiQ1_.resize( BLOCKSIZE );
	mpiQ2_.resize( BLOCKSIZE );
}

/////////////////////////////////////////////////////////////////////
// Stuff for ReduceQ
/////////////////////////////////////////////////////////////////////

		/**
		 * Adds an entry to the ReduceQ. If the Eref is a new one it 
		 * creates a new Q entry with slots for each thread, otherwise
		 * just fills in the appropriate thread on an existing entry.
		 * I expect that these will be quite rare.
		 */
// static func:
void Qinfo::addToReduceQ( ReduceBase* r, unsigned int threadIndex )
	/*
	const Eref& er, const ReduceFinfoBase* rfb, ReduceBase* r, 
	unsigned int threadIndex )
	*/
{
	reduceQ_[ threadIndex ].push_back( r );
}

/**
 * Utility function used below
 */
const ReduceBase* findMatchingReduceEntry( 
	const ReduceBase* start, vector< ReduceBase* >& vec, unsigned int i )
{
	// Usually the entry will be at i
	assert( i < vec.size() );
	if ( start->sameEref( vec[i] ) )
		return vec[i];
	for ( unsigned int k = 0; k < vec.size(); ++k ) {
		if ( i == k )
			continue;
		if ( start->sameEref( vec[k] ) )
			return vec[k];
	}
	return 0;
}

/**
 * Marches through reduceQ executing pending operations and finally
 * freeing the Reduce entries and zeroing out the queue.
 */
//static func
void Qinfo::clearReduceQ( unsigned int numThreads )
{
	if ( reduceQ_.size() == 0 ) {
		reduceQ_.resize( numThreads );
		return;
	}
	assert( reduceQ_.size() == numThreads );

	for ( unsigned int i = 0; i < reduceQ_[0].size(); ++i ) {
		ReduceBase* start = reduceQ_[0][i];
		for ( unsigned int j = 1; j < numThreads ; ++j ) {
			const ReduceBase* r = findMatchingReduceEntry(
				start, reduceQ_[j], i );
			assert( r );
			start->secondaryReduce( r );
		}
		// At this point start has all the info from the current node.
		// The reduceNodes returns 0 if the assignment should happen only
		// on another node. Sometimes the assignment happens on all nodes.
		if ( start->reduceNodes() ) {
			start->assignResult();
		}
	}
	for ( unsigned int j = 0; j < reduceQ_.size(); ++j ) {
		for ( unsigned int k = 0; k < reduceQ_[j].size(); ++k ) {
			delete reduceQ_[j][k];
		}
		reduceQ_[j].resize( 0 );
	}
}

/*
void Qinfo::setProcInfo( const ProcInfo* p )
{
	procIndex_ = p->procIndex;
}

const ProcInfo* Qinfo::getProcInfo() const
{
	Eref sheller = Id().eref();
	const Shell* s = reinterpret_cast< const Shell* >( sheller.data() );
	return s->getProcInfo( procIndex_ );
}
*/
