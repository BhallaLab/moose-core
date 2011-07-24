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
vector< Qvec > Qinfo::q1_;
vector< Qvec > Qinfo::q2_;
vector< Qvec >* Qinfo::inQ_ = &Qinfo::q1_;
vector< Qvec >* Qinfo::outQ_ = &Qinfo::q2_;
vector< vector< ReduceBase* > > Qinfo::reduceQ_;

Qvec Qinfo::mpiQ1_;
Qvec Qinfo::mpiQ2_;
Qvec* Qinfo::mpiInQ_ = &mpiQ1_;
Qvec* Qinfo::mpiRecvQ_ = &mpiQ2_;

vector< SimGroup > Qinfo::g_;
vector< char > Qinfo::structuralQ_;

void hackForSendTo( const Qinfo* q, const char* buf );
static const unsigned int BLOCKSIZE = 20000;

Qinfo::Qinfo( FuncId f, DataId srcIndex, unsigned int size, bool useSendTo )
	:	
		useSendTo_( useSendTo ), 
		isForward_( 1 ), 
		// isDummy_( 0 ), 
		m_( 0 ), 
		f_( f ), 
		srcIndex_( srcIndex ),
		size_( size ),
		procIndex_( 0 )
{;}

Qinfo::Qinfo( DataId srcIndex, unsigned int size, bool useSendTo )
	:	
		useSendTo_( useSendTo ), 
		isForward_( 1 ), 
		// isDummy_( 0 ), 
		m_( 0 ), 
		f_( 0 ), 
		srcIndex_( srcIndex ),
		size_( size ),
		procIndex_( 0 )
{;}

Qinfo::Qinfo()
	:	
		useSendTo_( 0 ), 
		isForward_( 1 ), 
		// isDummy_( 0 ), 
		m_( 0 ), 
		f_( 0 ), 
		srcIndex_( 0 ),
		size_( 0 ),
		procIndex_( 0 )
{;}

/// Static function
// deprecated
/*
Qinfo Qinfo::makeDummy( unsigned int size )
{
	Qinfo ret( 0, size, 0 ) ;
	// ret.isDummy_ = 1;
	return ret;
}
*/

/**
 * Static func: Sets up a SimGroup to keep track of thread and node
 * grouping info. This is used by the Qinfo to manage assignment of
 * threads and queues.
 * numThreads is the number of threads present in this group on this node.
 * Returns the group number of the new group.
 * have that because it will require updates to messages.
 * Returns number of new sim group.
 */
unsigned int Qinfo::addSimGroup( unsigned short numThreads, 
	unsigned short numNodes )
{
	const unsigned int DefaultMpiBufSize = 1000; // bytes.
	/*
	unsigned short ng = g_.size();
	unsigned short si = 0;
	if ( ng > 0 )
		si = g_[ng - 1].startThread + g_[ng - 1].numThreads;
	*/
	// For starters, just put this group on all nodes.
	SimGroup sg( 0, numNodes, DefaultMpiBufSize );
	g_.push_back( sg );

	Qvec qv( numThreads );

	q1_.push_back( qv );
	q2_.push_back( qv );

	return g_.size() - 1;
}

// Static function
unsigned int Qinfo::numSimGroup()
{
	return g_.size();
}

// Static function
const SimGroup* Qinfo::simGroup( unsigned int index )
{
	assert( index < g_.size() );
	return &( g_[index] );
}

// Static function
void Qinfo::clearSimGroups()
{
	g_.clear();
}

/**
 * This is called within barrier1 of the ProcessLoop. It isn't
 * thread-safe, relies on the location of the call to achieve safety.
 */
void Qinfo::clearStructuralQ()
{
	isSafeForStructuralOps_ = 1;
	const char* begin = &( structuralQ_[0] );
	const char* end = begin + structuralQ_.size();
	const char* buf = begin;
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

/*

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
*/
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
	assert( qv.allocatedSize() >= Qvec::HeaderSize &&
		qv.mpiArrivedDataSize() >= Qvec::HeaderSize );
	const char* buf = qv.data();

	// unsigned int bufsize = qv.dataQsize();
	unsigned int bufsize = qv.mpiArrivedDataSize() - Qvec::HeaderSize;
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
 * Checks the data block size in case it is over the regular
 * blocksize allocated for the current node.
 */
void Qinfo::readMpiQ( const ProcInfo* proc, unsigned int node )
{
	assert( proc );
	// assert( proc->groupId < mpiQ_.size() );
	if ( mpiInQ_->mpiArrivedDataSize() > Qvec::HeaderSize )
		// cout << Shell::myNode() << ":" << proc->threadIndexInGroup << "	mpi data transf = " << mpiInQ_->mpiArrivedDataSize() << endl;

	readBuf( *mpiInQ_, proc );

	if ( proc->nodeIndexInGroup == 0 && mpiInQ_->isBigBlock() ) 
		// Clock::requestBigBlock( proc->groupId, node );
		;
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

/**
 * Need to allocate space for incoming block
 * Still pending: need a way to decide which SimGroup is
 * coming in.
 */
void Qinfo::swapMpiQ()
{
	if ( mpiInQ_ == &mpiQ2_ ) {
		mpiInQ_ = &mpiQ1_;
		mpiRecvQ_ = &mpiQ2_;
	} else {
		mpiInQ_ = &mpiQ2_;
		mpiRecvQ_ = &mpiQ1_;
	}
	// mpiRecvQ_->resizeLinearData( Qinfo::simGroup( 1 )->bufSize() );
	mpiRecvQ_->resizeLinearData( BLOCKSIZE );
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
	cout << endl;

	if ( inQ_->size() > 0 ) innerReportQ( (*inQ_)[0], "inQ[0]" );
	if ( inQ_->size() > 1 ) innerReportQ( (*inQ_)[1], "inQ[1]" );
	if ( outQ_->size() > 0 ) innerReportQ( (*outQ_)[0], "outQ[0]" );
	if ( outQ_->size() > 1 ) innerReportQ( (*outQ_)[1], "outQ[1]" );
	if ( mpiInQ_ ) innerReportQ( *mpiInQ_, "mpiInQ" );
	if ( mpiRecvQ_ ) innerReportQ( *mpiRecvQ_, "mpiRecvQ" );
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

void Qinfo::addToQforward( const ProcInfo* p, MsgFuncBinding b, 
	const char* arg )
{
	// static pthread_mutex_t *mutex = init_mutex(); // only called once.
	m_ = b.mid;
	f_ = b.fid;
	assert( b.mid != Msg::badMsg );
	isForward_ = 1;
	// pthread_mutex_lock( mutex );
		(*outQ_)[p->groupId].push_back( p->threadIndexInGroup, this, arg );
	// pthread_mutex_unlock( mutex );
}

void Qinfo::addToQbackward( const ProcInfo* p, MsgFuncBinding b, 
	const char* arg )
{
	// static pthread_mutex_t *mutex = init_mutex(); // only called once.

	m_ = b.mid;
	f_ = b.fid;
	assert( b.mid != Msg::badMsg );
	isForward_ = 0;
	// if ( p->threadIndexInGroup != 0 ) cout << "#" << flush;
	// if ( procIndex_ != p->threadIndexInGroup ) cout << "@" << flush;
	// cout << p->groupId << ":" << p->threadIndexInGroup << " " << flush;

	// pthread_mutex_lock( mutex );
		(*outQ_)[p->groupId].push_back( p->threadIndexInGroup, this, arg );
	// pthread_mutex_unlock( mutex );
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
bool Qinfo::addToStructuralQ() const
{
	if ( isSafeForStructuralOps_ )
		return 0;
	const char* begin = reinterpret_cast< const char* >( this );
	const char* end = begin + sizeof( Qinfo ) + size_;
	structuralQ_.insert( structuralQ_.end(), begin, end );
	// structuralQ_.push_back( reinterpret_cast< const char* >( this ) );
	return 1;
}

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

// static func. Typically only called during setup in Shell::setHardware.
void Qinfo::initMpiQs()
{
	mpiQ1_.resizeLinearData( BLOCKSIZE );
	mpiQ1_.setMpiDataSize( 0 );
	mpiQ2_.resizeLinearData( BLOCKSIZE );
	mpiQ2_.setMpiDataSize( 0 );
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
