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
const BindIndex Qinfo::DirectAdd = -1;
vector< double > Qinfo::q0_( 1000, 0.0 );
vector< double > Qinfo::q1_( 1000, 0.0 );

vector< vector< double > > Qinfo::dBuf_;
vector< vector< Qinfo > > Qinfo::qBuf_;

double* Qinfo::inQ_ = &Qinfo::q0_[0];
double* Qinfo::outQ_ = &Qinfo::q1_[0];
vector< vector< ReduceBase* > > Qinfo::reduceQ_;

vector< double > Qinfo::mpiQ1_( 1000, 0.0 );
vector< double > Qinfo::mpiQ2_( 1000, 0.0 );
double* Qinfo::mpiInQ_ = &mpiQ1_[0];
double* Qinfo::mpiRecvQ_ = &mpiQ2_[0];

vector< Qinfo > Qinfo::structuralQinfo_;
vector< double > Qinfo::structuralQdata_;

pthread_mutex_t* Qinfo::qMutex_;

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
	BindIndex bindIndex, ThreadId threadNum,
	unsigned int dataIndex )
	:	
		src_( src ),
		msgBindIndex_( bindIndex ),
		threadNum_( threadNum ),
		dataIndex_( dataIndex )
{;}

Qinfo::Qinfo( const Qinfo* orig, ThreadId threadNum )
	:	
		src_( orig->src_ ),
		msgBindIndex_( orig->msgBindIndex_ ),
		threadNum_( threadNum ),
		dataIndex_( orig->dataIndex_ )
{;}

bool Qinfo::execThread( Id id, unsigned int dataIndex ) const
{
	return threadNum_ == ( ( id.value() + dataIndex ) % Shell::numCores() );
}

/**
 * This is called within barrier1 of the ProcessLoop. It isn't
 * thread-safe, relies on the location of the call to achieve safety.
 */
void Qinfo::clearStructuralQ()
{
	enableStructuralOps();
	double* buf = &structuralQdata_[0];
	for ( unsigned int i = 0; i < structuralQinfo_.size(); ++i ) {
		const Qinfo* qi = &structuralQinfo_[i];
		if ( !qi->isDummy() ) {
			// Instead of ScriptThreadNum I use the first worker thread here
			Qinfo newQ( qi, 1 );
			const Element* e = qi->src_.element();
			e->exec( &newQ, buf + newQ.dataIndex_ );
		}
	}

	disableStructuralOps();
	structuralQinfo_.resize( 0 );
	structuralQdata_.resize( 0 );
}

void readBuf(const double* buf, ThreadId threadNum )
{
	unsigned int bufsize = static_cast< unsigned int >( buf[0] );
	unsigned int numQinfo = static_cast< unsigned int >( buf[1] );
	assert( bufsize > numQinfo * 3 );

	const double* qptr = buf + 2;

	for ( unsigned int i = 0; i < numQinfo; ++i ) {
		// const Qinfo* qi = reinterpret_cast< const Qinfo* >( qptr );
		Qinfo qi( reinterpret_cast< const Qinfo* >( qptr ), threadNum );
		qptr += 3;
		if ( !qi.isDummy() ) {
			const Element* e = qi.src().element();
			e->exec( &qi, buf + qi.dataIndex() );
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
 * Thread safety is fragile here. We require that all the exec functions
 * on the Element or on the Msg, should internally guarantee that they
 * will dispatch calls only to the ObjIds that are allowed for the
 * running thread.
 */ 
void Qinfo::readQ( ThreadId threadNum )
{
	readBuf( inQ_, threadNum );
}

/**
 * Static func. 
 * Deliver the contents of the mpiQ to target objects
 * Assumes that the Msgs invoked by readBuf handle the thread safety.
 * Checks the data block size in case it is over the regular
 * blocksize allocated for the current node.
 */
void Qinfo::readMpiQ( ThreadId threadNum, unsigned int node )
{
	readBuf( mpiInQ_, threadNum );

	/*
	if ( proc->nodeIndexInGroup == 0 && mpiInQ_->isBigBlock() ) 
		// Clock::requestBigBlock( proc->groupId, node );
		;
		*/
}

/**
 * Stitches together thread blocks on
 * the inQ so that the readBuf function will go through as a single 
 * unit.
 * Static func. Not thread safe. Must be done on single thread,
 * protected by barrier so that it happens at a defined point for all
 * threads.
 */
void Qinfo::swapQ()
{
	/**
	 * This whole function is protected by a mutex so that the master
	 * thread from the parser does not issue any more commands while we
	 * are shuffling data from the queues to the inQ. Note that queue0
	 * is for the master thread and therefore it must also be protected.
	 * clearStructuralQ happens here protected by the barrier, so that 
	 * operations that change the structure of the model can occur without 
	 * risk of affecting ongoing messaging.
	 */
	pthread_mutex_lock( qMutex_ );
		clearStructuralQ(); // static function.

		/**
	 	* Here we just deposit all the data from the data and Q vectors into
	 	* the inQ.
	 	*/
		unsigned int bufsize = 0;
		unsigned int datasize = 0;
		unsigned int numQinfo = 0;
		// Note that we have an extra queue for the parser
		for ( unsigned int i = 0; i <= Shell::numCores(); ++i ) {
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
		double* dptr = &q0_[0];
		unsigned int prevQueueDataIndex = numQinfo * 3 + 2;
		// Note that we have an extra queue for the parser
		for ( unsigned int i = 0; i <= Shell::numCores(); ++i ) {
			const double *dataOrig = &( dBuf_[i][0] );
			// vector< Qinfo >::iterator qEnd = qBuf_[i].end();
			vector< Qinfo >& qvec = qBuf_[i];
			unsigned int numQentries = qvec.size();

			for ( unsigned int j = 0; j < numQentries; ++j ) {
				Qinfo* qi = reinterpret_cast< Qinfo* >( qptr );
				*qi = qvec[j];
				qi->dataIndex_ += prevQueueDataIndex;

				unsigned int size = ( j + 1 == numQentries ) ? 
					dBuf_[i].size() : qvec[j+1].dataIndex();
				size -= qvec[j].dataIndex();

				memcpy( dptr + qi->dataIndex_, 
					dataOrig + qvec[j].dataIndex(),
					size * sizeof( double ) );

				qptr += 3;
			}
			prevQueueDataIndex += dBuf_[i].size();
		}
		assert( prevQueueDataIndex == bufsize );
		for ( unsigned int i = 0; i <= Shell::numCores(); ++i ) {
			qBuf_[i].resize( 0 );
			dBuf_[i].resize( 0 );
		}
	pthread_mutex_unlock( qMutex_ );
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
			", dataIndex=" << qi->dataIndex();
		if ( qi->isDirect() ) {
			const ObjFid *f;
			f = reinterpret_cast< const ObjFid* >( q + qi->dataIndex() );
			cout << ", dest= " << f->oi << ", fid= " << f->fid << 
				", size= " << f->entrySize << ", n= " << f->numEntries;
		}
		cout << endl;
		qptr += 3;
	}
}

void reportOutQ( const vector< vector< Qinfo > >& q, 
	const vector< vector< double > >& d, const string& name )
{
	cout << endl << Shell::myNode() << ": Reporting " << name << endl;
	for ( unsigned int i = 0; i < Shell::numCores(); ++i ) {
		for ( vector< Qinfo >::const_iterator qi = q[i].begin(); 
			qi != q[i].end(); ++qi ) {
			cout << i << ": src= " << qi->src() <<
				", msgBind=" << qi->bindIndex() <<
				", threadNum=" << qi->threadNum() <<
				", dataIndex=" << qi->dataIndex();
			if ( qi->isDirect() ) {
				const ObjFid *f;
				f = reinterpret_cast< const ObjFid* >( &d[qi->dataIndex()]);
				cout << ", dest= " << f->oi << ", fid= " << f->fid << 
					", size= " << f->entrySize << ", n= " << f->numEntries;
			}
			cout << endl;
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
	if ( numQinfo > 0 ) reportOutQ( qBuf_, dBuf_, "outQ" );
	if ( mpiInQ_[1] > 0 ) innerReportQ( mpiInQ_, "mpiInQ" );
	if ( mpiRecvQ_[1] > 0 ) innerReportQ( mpiRecvQ_, "mpiRecvQ" );
}

/**
 * Static function.
 * Adds a Queue entry.
 */
void Qinfo::addToQ( const ObjId& oi, 
	BindIndex bindIndex, ThreadId threadNum,
	const double* arg, unsigned int size )
{
	if ( threadNum == ScriptThreadNum ) pthread_mutex_lock( qMutex_ );
		vector< double >& vec = dBuf_[ threadNum ];
		qBuf_[ threadNum ].push_back( 
			Qinfo( oi, bindIndex, threadNum, vec.size() ) );
		if ( size > 0 ) {
			vec.insert( vec.end(), arg, arg + size );
		}
	if ( threadNum == ScriptThreadNum ) pthread_mutex_unlock( qMutex_ );
}

/// Utility variant that adds two args.
void Qinfo::addToQ( const ObjId& oi, 
	BindIndex bindIndex, ThreadId threadNum,
	const double* arg1, unsigned int size1, 
	const double* arg2, unsigned int size2 )
{
	if ( threadNum == ScriptThreadNum ) pthread_mutex_lock( qMutex_ );
		vector< double >& vec = dBuf_[ threadNum ];
		qBuf_[ threadNum ].push_back( 
			Qinfo( oi, bindIndex, threadNum, vec.size() ) );
		if ( size1 > 0 )
			vec.insert( vec.end(), arg1, arg1 + size1 );
		if ( size2 > 0 )
			vec.insert( vec.end(), arg2, arg2 + size2 );
	if ( threadNum == ScriptThreadNum ) pthread_mutex_unlock( qMutex_ );
}

/**
 * Static function.
 * Adds a Queue entry. This variant allocates the space and returns
 * the pointer into which the data can be copied. Useful when there are
 * multiple fields to be put in.
double* Qinfo::addToQ( const ObjId& oi, 
	BindIndex bindIndex, ThreadId threadNum,
	unsigned int size )
{
	if ( threadNum == ScriptThreadNum ) pthread_mutex_lock( qMutex_ );
		vector< double >& data = dBuf_[ threadNum ];
		unsigned int oldSize = data.size();
		qBuf_[ threadNum ].push_back( 
			Qinfo( oi, bindIndex, threadNum, oldSize ) );
		data.resize( oldSize + size );
	if ( threadNum == ScriptThreadNum ) pthread_mutex_unlock( qMutex_ );
		return &( data[oldSize] );
}
 */

// Static function
void Qinfo::addDirectToQ( const ObjId& src, const ObjId& dest, 
	ThreadId threadNum,
	FuncId fid, 
	const double* arg, unsigned int size )
{
	static const unsigned int ObjFidSizeInDoubles = 
		1 + ( sizeof( ObjFid ) - 1 ) / sizeof( double );

	if ( threadNum == ScriptThreadNum ) pthread_mutex_lock( qMutex_ );
		vector< double >& vec = dBuf_[ threadNum ];
		qBuf_[ threadNum ].push_back(
			Qinfo( src, DirectAdd, threadNum, vec.size() ) );
		ObjFid ofid = { dest, fid, size, 1 };
		const double* ptr = reinterpret_cast< const double* >( &ofid );
		vec.insert( vec.end(), ptr, ptr + ObjFidSizeInDoubles );
		if ( size > 0 ) {
			vec.insert( vec.end(), arg, arg + size );
		}
	if ( threadNum == ScriptThreadNum ) pthread_mutex_unlock( qMutex_ );
}

// Static function
void Qinfo::addDirectToQ( const ObjId& src, const ObjId& dest, 
	ThreadId threadNum,
	FuncId fid, 
	const double* arg1, unsigned int size1,
	const double* arg2, unsigned int size2 )
{
	static const unsigned int ObjFidSizeInDoubles = 
		1 + ( sizeof( ObjFid ) - 1 ) / sizeof( double );

	if ( threadNum == ScriptThreadNum ) pthread_mutex_lock( qMutex_ );
		vector< double >& vec = dBuf_[ threadNum ];
		qBuf_[ threadNum ].push_back(
			Qinfo( src, DirectAdd, threadNum, vec.size() ) );
		ObjFid ofid = { dest, fid, size1 + size2, 1 };
		const double* ptr = reinterpret_cast< const double* >( &ofid );
		vec.insert( vec.end(), ptr, ptr + ObjFidSizeInDoubles );
		if ( size1 > 0 )
			vec.insert( vec.end(), arg1, arg1 + size1 );
		if ( size2 > 0 )
			vec.insert( vec.end(), arg2, arg2 + size2 );
	if ( threadNum == ScriptThreadNum ) pthread_mutex_unlock( qMutex_ );
}

/*
// Static function
double* Qinfo::addDirectToQ( const ObjId& src, const ObjId& dest, 
	ThreadId threadNum,
	FuncId fid, unsigned int size )
{
	static const unsigned int ObjFidSizeInDoubles = 
		1 + ( sizeof( ObjFid ) - 1 ) / sizeof( double );

	if ( threadNum == ScriptThreadNum ) pthread_mutex_lock( qMutex_ );
		qBuf_[ threadNum ].push_back( 
			Qinfo( src, DirectAdd, threadNum, dBuf_[threadNum].size() ) );
	
		ObjFid ofid = { dest, fid, size, 1 };
		const double* ptr = reinterpret_cast< const double* >( &ofid );
		vector< double >& vec = dBuf_[ threadNum ];
		vec.insert( vec.end(), ptr, ptr + ObjFidSizeInDoubles );
	
		unsigned int oldSize = dBuf_[threadNum].size();
		dBuf_[ threadNum ].resize( oldSize + size );
	if ( threadNum == ScriptThreadNum ) pthread_mutex_unlock( qMutex_ );
		return &( dBuf_[threadNum][oldSize] );
}
*/

// Static function
void Qinfo::addVecDirectToQ( const ObjId& src, const ObjId& dest, 
	ThreadId threadNum,
	FuncId fid, 
	const double* arg, unsigned int entrySize, unsigned int numEntries )
{
	static const unsigned int ObjFidSizeInDoubles = 
		1 + ( sizeof( ObjFid ) - 1 ) / sizeof( double );

	if ( entrySize == 0 || numEntries == 0 )
		return;
	if ( threadNum == ScriptThreadNum ) pthread_mutex_lock( qMutex_ );
		qBuf_[ threadNum ].push_back( 
			Qinfo( src, DirectAdd, threadNum, dBuf_[threadNum].size() ) );
	
		ObjFid ofid = { dest, fid, entrySize, numEntries };
		const double* ptr = reinterpret_cast< const double* >( &ofid );
		vector< double >& vec = dBuf_[ threadNum ];
		vec.insert( vec.end(), ptr, ptr + ObjFidSizeInDoubles );
		vec.insert( vec.end(), arg, arg + entrySize * numEntries );
	if ( threadNum == ScriptThreadNum ) pthread_mutex_unlock( qMutex_ );
}

/// Static func.
void Qinfo::disableStructuralOps()
{
	// pthread_mutex_lock( qMutex_ );
		isSafeForStructuralOps_ = 0;
	// pthread_mutex_unlock( qMutex_ );
}

/// static func
void Qinfo::enableStructuralOps()
{
	// pthread_mutex_lock( qMutex_ );
		isSafeForStructuralOps_ = 1;
	// pthread_mutex_unlock( qMutex_ );
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
	bool ret = 0;
	// pthread_mutex_lock( qMutex_ );
		if ( !isSafeForStructuralOps_ ) {
			structuralQinfo_.push_back( *this );
			/// Here we assert that we are reading from the inQ_ 
			/// and that there are no gaps in the data or Qinfo sections,
			/// and that the last used Qinfo is terminated by a null Qinfo.
			const double* data = &( inQ_[ dataIndex_ ] );
			const Qinfo* nextQinfo = this + 1;
			unsigned int size = nextQinfo->dataIndex_ - dataIndex_;
			structuralQinfo_.back().dataIndex_ = structuralQdata_.size();
			structuralQdata_.insert( structuralQdata_.end(), data, data + size );
			ret = 1;
		}
	// pthread_mutex_unlock( qMutex_ );
	return ret;
}


/// Static func
void Qinfo::initMutex()
{
	qMutex_ = new pthread_mutex_t;
	pthread_mutex_init( qMutex_, NULL );
}

/// Static func
void Qinfo::freeMutex()
{
	pthread_mutex_destroy( qMutex_ );
	delete qMutex_;
}

/// Static func
void Qinfo::emptyAllQs()
{
	inQ_[0] = 0;
	inQ_[1] = 0;
	for ( unsigned int i = 0; i < Shell::numCores(); ++i ) {
		qBuf_[i].resize( 0 );
		dBuf_[i].resize( 0 );
	}
}

// Static function. Used only during single-thread tests, when the
// main thread-handling loop is inactive.
// Called after all the 'send' functions have been done. Typically just
// one is pending.
void Qinfo::clearQ( ThreadId threadNum )
{
	swapQ();
	readQ( threadNum );
}

// static func. Called during setup in Shell::setHardware.
void Qinfo::initQs( unsigned int numThreads, unsigned int reserve )
{
	qBuf_.resize( numThreads );
	dBuf_.resize( numThreads );
	for ( unsigned int i = 0; i < numThreads; ++i ) {
		// A reasonably large reserve size keeps different threads from
		// stepping on each others toes when writing to their individual
		// portions of the buffers.
		qBuf_.reserve( reserve );
		dBuf_.reserve( reserve );
	}
}

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
