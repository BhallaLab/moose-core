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
#include <time.h>
#ifdef WIN32	      // True for Visual C++ compilers
	#include <Windows.h>  // for Win32 Sleep function
#endif

/*
#ifdef USE_MPI
#include <mpi.h>
#endif
*/

// Declaration of static fields
const BindIndex Qinfo::DirectAdd = -1;
vector< double > Qinfo::q0_( 1000, 0.0 );

vector< vector< double > > Qinfo::dBuf_;
vector< vector< Qinfo > > Qinfo::qBuf_;

double* Qinfo::inQ_ = &Qinfo::q0_[0];
vector< vector< ReduceBase* > > Qinfo::reduceQ_;

vector< double > Qinfo::mpiQ0_( 1000, 0.0 );
vector< double > Qinfo::mpiQ1_( 1000, 0.0 );
double* Qinfo::mpiRecvQ_ = &mpiQ0_[0];

vector< Qinfo > Qinfo::structuralQinfo_( 0 );
vector< double > Qinfo::structuralQdata_( 0 );

bool Qinfo::waiting_ = 0;
int Qinfo::numCyclesToWait_ = 0;
unsigned long Qinfo::numProcessCycles_ = 0;
const double Qinfo::blockMargin_ = 1.1;
const unsigned int Qinfo::historySize_ = 4;
unsigned int Qinfo::sourceNode_ = 0;
vector< vector< unsigned int > > Qinfo::history_;
vector< unsigned int > Qinfo::blockSize_;

static const unsigned int QinfoSizeInDoubles = 
			1 + ( sizeof( Qinfo ) - 1 ) / sizeof( double );


/// Default dummy Qinfo creation.
Qinfo::Qinfo()
	:	
		src_(),
		msgBindIndex_(0),
		threadNum_( 0 ),
		dataIndex_( 0 ),
		dataSize_( 0 )
{;}

Qinfo::Qinfo( const ObjId& src, 
	BindIndex bindIndex, ThreadId threadNum,
	unsigned int dataIndex, unsigned int dataSize )
	:	
		src_( src ),
		msgBindIndex_( bindIndex ),
		threadNum_( threadNum ),
		dataIndex_( dataIndex ),
		dataSize_( dataSize )
{;}

Qinfo::Qinfo( const Qinfo* orig, ThreadId threadNum )
	:	
		src_( orig->src_ ),
		msgBindIndex_( orig->msgBindIndex_ ),
		threadNum_( threadNum ),
		dataIndex_( orig->dataIndex_ ),
		dataSize_( orig->dataSize_ )
{;}

/*
bool Qinfo::execThread( Id id, unsigned int dataIndex ) const
{
	// Note that nothing will ever be executed on thread# 0 unless it is
	// in single thread mode.
	return ( Shell::isSingleThreaded() || 
		( threadNum_ == ( 1 + ( ( id.value() + dataIndex ) % Shell::numProcessThreads() ) ) ) );
}
*/

/**
 * This is called within barrier1 of the ProcessLoop. It isn't
 * thread-safe, relies on the location of the call to achieve safety.
 */
void Qinfo::clearStructuralQ()
{
 	structuralQdata_.resize( 0 );
}

void readBuf(const double* buf, ThreadId threadNum )
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
 * System-independent timeout for busy loops when nothing is happening.
 */
void quickNap()
{
#ifdef WIN32 // If this is an MS VC++ compiler..
	unsigned int milliseconds = 1;
	Sleep( milliseconds );
#else           // else assume POSIX compliant..
	struct timespec req = { 0, 1000000 };
	nanosleep( &req, 0 );
#endif // _MSC_VER
}

/**
 * Stitches together thread blocks on
 * the inQ so that the readBuf function will go through as a single unit.
 * Runs only in barrier 1.
 * Static func. Not thread safe. Must be done on single thread,
 * protected by barrier so that it happens at a defined point for all
 * threads.
 */
void Qinfo::swapQ()
{
	/**
	 * clearStructuralQ is not protected by the mutex, as it may issue
	 * calls that put further entries into the Q, and this would cause a
	 * race condition. 
	 * While this means that the parser may request operations at the same
	 * time as we have operations that change the structure of the model,
	 * note that the parser requests just go into the queue. So things
	 * should not conflict.
	 */
	clearStructuralQ(); // static function.

	/**
	 * This whole function is protected by a mutex so that the master
	 * thread from the parser does not issue any more commands while we
	 * are shuffling data from the queues to the inQ. Note that queue0
	 * is for the master thread and therefore it must also be protected.
	 */

		/**
	 	* Here we just deposit all the data from the data and Q vectors into
	 	* the inQ.
	 	*/
		unsigned int bufsize = 0;
		unsigned int datasize = 0;
		unsigned int numQinfo = 0;
		// Note that we have an extra queue for the parser
		for ( unsigned int i = 0; i <= Shell::numProcessThreads(); ++i ) {
			numQinfo += qBuf_[i].size();
			datasize += dBuf_[i].size();
		}
	
		bufsize = numQinfo * QinfoSizeInDoubles + datasize + 2;
		if ( bufsize > q0_.capacity() ) {
			// cout << "Qinfo::swapQ: Raising bufsize from " << q0_.capacity() << " to " << bufsize << ", numQinfo = " << numQinfo << ", datasize = " << datasize << endl;
			q0_.reserve( bufsize + bufsize / 2 );
		}
		q0_.resize( bufsize, 0 );
		inQ_ = &q0_[0];
		q0_[0] = bufsize;
		q0_[1] = numQinfo;
		double* qptr = &q0_[2];
		double* dptr = &q0_[0];
		unsigned int prevQueueDataIndex = numQinfo * QinfoSizeInDoubles + 2;
		// Note that we have an extra queue for the parser
		for ( unsigned int i = 0; i <= Shell::numProcessThreads(); ++i ) {
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

				qptr += QinfoSizeInDoubles;
			}
			prevQueueDataIndex += dBuf_[i].size();
		}
		assert( prevQueueDataIndex == bufsize );
		assert( qBuf_.size() > Shell::numProcessThreads() );
		assert( dBuf_.size() > Shell::numProcessThreads() );
		for ( unsigned int i = 0; i <= Shell::numProcessThreads(); ++i ) {
			qBuf_[i].resize( 0 );
			dBuf_[i].resize( 0 );
		}

	++numProcessCycles_;


	// Used to avoid pounding on the CPU when nothing is happening.
	if ( Shell::isParserIdle() && numQinfo == 0 ) {
		quickNap();
	}
}

/**
 * Waits the specified number of Process cycles before returning.
 * Should only be called on the master thread (thread 0).
 * Must not be called recursively.
 * Static func.
 * Deprecated, here as dummy till I get round to cleaning out.
 */
void Qinfo::waitProcCycles( unsigned int numCycles )
{
		for ( unsigned int i = 0; i < numCycles; ++i )
			clearQ( ScriptThreadNum );
}

//////////////////////////////////////////////////////////////////////
// MPI data stuff
//////////////////////////////////////////////////////////////////////
/**
 * updateQhistory() keeps track of recent bufsizes and has a simple 
 * heuristic to judge how big the next mpi data transfer should be:
 * It takes the second-biggest of the last historySize_ entries, 
 * adds 10% and adds 10 to that.
 * static func.
 */
void Qinfo::updateQhistory()
{
	assert( history_.size() == Shell::numNodes() );
	assert( blockSize_.size() == Shell::numNodes() );
	assert( sourceNode_ < Shell::numNodes() );
	vector< unsigned int >& h = history_[sourceNode_];
	assert( h.size() == historySize_ );
	assert( inQ_ );
	h[ numProcessCycles_ % h.size() ] = inQ_[0];
	unsigned int max = 0;
	unsigned int nextMax = 0;
	for ( unsigned int i = 0; i < h.size(); ++i ) {
		unsigned int j = h[i];
		if ( max < j ) {
			nextMax = max;
			max = j;
		} else if ( nextMax < j ) {
			nextMax = j;
		}
	}
	blockSize_[ sourceNode_ ] = 
		static_cast< double >( nextMax ) * blockMargin_ + 10;
	// cout << Shell::myNode() << ": blockSize[ " << sourceNode_ << " ] = " << blockSize_[ sourceNode_ ] << endl;
}


/**
 * If an MPI packet comes in asking for more space than the blockSize, this 
 * function resizes the buffer to fit.
 * Static func.
 */
void Qinfo::expandMpiRecvQ( unsigned int size )
{
	if ( mpiRecvQ_ == &mpiQ0_[0] ) {
		if ( mpiQ0_.size() < size ) {
			// cout << Shell::myNode() << ": expanding mpiQ0 from " << mpiQ0_.size() << " to " << size << endl;
			mpiQ0_.resize( size );
			mpiRecvQ_ = &mpiQ0_[0];
		}
	} else {
		if ( mpiQ1_.size() < size ) {
			// cout << Shell::myNode() << ": expanding mpiQ1 from " << mpiQ1_.size() << " to " << size << endl;
			mpiQ1_.resize( size );
			mpiRecvQ_ = &mpiQ1_[0];
		}
	}
}

// static func
void Qinfo::setSourceNode( unsigned int n )
{
	sourceNode_ = n;
}

// static func
unsigned int Qinfo::blockSize( unsigned int node )
{
	assert ( node < blockSize_.size() );
	return blockSize_[ node ];
}

/**
 * Puts the data ready to be processed into InQ, and sets up a suitable
 * buffer for the RecvQ.
 * Also updates history of MPI data transfer size.
 * I have 3 pointers and 3 buffers.
 * InQ always points to the buffer that has to be processed.
 * recvQ always points to the buffer ready to receive data.
 * sendQ simply points to the buffer holding the outgoing data, &q0_[0],
 * and does not change.
 * Static func.
 */
void Qinfo::swapMpiQ()
{
	assert( mpiQ0_.size() > 0 );
	assert( mpiQ1_.size() > 0 );
	assert( sourceNode_ < Shell::numNodes() );
	unsigned int nextNode = ( sourceNode_ + 1 ) % Shell::numNodes();
	// cout << Shell::myNode() << ": Qinfo::swapMpiQ: mpiRecvQ_=" << mpiRecvQ_ << ", &mpiQ0= " << &mpiQ0_[0] << " (" << mpiQ0_.size() << "), &mpiQ1= " << &mpiQ1_[0] << " (" << mpiQ1_.size() << ")\n"; 
	if ( mpiRecvQ_ == &mpiQ0_[0] ) {
		if ( mpiQ1_.size() < blockSize_[ nextNode ] ) {
			// cout << Shell::myNode() << ": resizing mpiQ1 from " << mpiQ1_.size() << " to " << blockSize_[ nextNode ] << endl;
			mpiQ1_.resize( blockSize_[ nextNode ] );
		}
		mpiRecvQ_ = &mpiQ1_[0];
		inQ_ = &mpiQ0_[0];
	} else {
		if ( mpiQ0_.size() < blockSize_[ nextNode ] ) {
			// cout << Shell::myNode() << ": resizing mpiQ0 from " << mpiQ0_.size() << " to " << blockSize_[ nextNode ] << endl;
			mpiQ0_.resize( blockSize_[ nextNode ] );
		}
		mpiRecvQ_ = &mpiQ0_[0];
		inQ_ = &mpiQ1_[0];
	}
	if ( sourceNode_ == Shell::myNode() ) {
		inQ_ = &q0_[0];
	} else {
		if ( nextNode == Shell::myNode() ) {
			if ( q0_.capacity() < blockSize_[ nextNode ] ) {
				// cout << "Reserving q0 capacity from " << q0_.capacity() << " to " << blockSize_[nextNode] << endl;
				q0_.reserve( blockMargin_ * blockSize_[ nextNode ] );
			}
		}
	}
	updateQhistory();
	// cout << Shell::myNode() << ": Qinfo::swapMpiQ: bufsize=" << inQ_[0] << ", numQinfo= " << inQ_[1] << endl;

	// Qinfo::reportQ();
}


//////////////////////////////////////////////////////////////////////
void reportQentry( unsigned int i, const Qinfo* qi, const double *q )
{
	cout << i << ": src= " << qi->src() << 
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
}

void innerReportQ( const double* q, const string& name )
{
	cout << endl << Shell::myNode() << ": Reporting " << name << endl;
	unsigned int numQinfo = q[1];
	const double* qptr = q + 2;
	for ( unsigned int i = 0; i < numQinfo; ++i ) {
		const Qinfo* qi = reinterpret_cast< const Qinfo* >( qptr );
		reportQentry( i, qi, q );
		qptr += QinfoSizeInDoubles;
	}
}

void reportOutQ( const vector< vector< Qinfo > >& q, 
	const vector< vector< double > >& d, const string& name )
{
	cout << endl << Shell::myNode() << ": Reporting " << name << endl;
	for ( unsigned int i = 0; i <= Shell::numProcessThreads(); ++i ) {
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

void reportStructQ( const vector< Qinfo >& sq, const double* q )
{
	cout << endl << Shell::myNode() << ": Reporting structuralQinfo\n";
	for ( unsigned int i = 0; i < sq.size(); ++i ) {
		reportQentry( i, &sq[i], q + 2 );
	}
}

/**
 * Static func. readonly, so it is thread safe
 */
void Qinfo::reportQ()
{
	cout << Shell::myNode() << ":	inQ: size =  " << inQ_[0] << 
		", numQinfo = " << inQ_[1] << endl;
	cout << Shell::myNode() << ":	mpiRecvQ: size =  " << mpiRecvQ_[0] << 
		", numQinfo = " << mpiRecvQ_[1] << endl;

	unsigned int datasize = 0;
	unsigned int numQinfo = 0;
	for ( unsigned int i = 0; i <= Shell::numProcessThreads(); ++i ) {
		numQinfo += qBuf_[i].size();
		datasize += dBuf_[i].size();
	}
	cout << Shell::myNode() << ":	outQ: numProcessThreads = " << 
		Shell::numProcessThreads() << ", size = " << datasize + QinfoSizeInDoubles * numQinfo << 
		", numQinfo = " << numQinfo << endl;

	cout << Shell::myNode() << ":	structuralQ: size = " << 
		structuralQdata_.size() + QinfoSizeInDoubles * structuralQinfo_.size() << ", numQinfo =  " << structuralQinfo_.size() << endl;

	if ( inQ_[1] > 0 ) innerReportQ(inQ_, "inQ" );
	if ( numQinfo > 0 ) reportOutQ( qBuf_, dBuf_, "outQ" );
	if ( mpiRecvQ_[1] > 0 ) innerReportQ( mpiRecvQ_, "mpiRecvQ" );
	if ( structuralQinfo_.size() > 0 ) reportStructQ( structuralQinfo_, inQ_ );
}

/**
 * Static function.
 * Adds a Queue entry.
 */
void Qinfo::addToQ( const ObjId& oi, 
	BindIndex bindIndex, ThreadId threadNum,
	const double* arg, unsigned int size )
{
	if ( oi.element()->hasMsgs( bindIndex ) ) {
			assert( dBuf_.size() > threadNum );
			assert( qBuf_.size() > threadNum );
			vector< double >& vec = dBuf_[ threadNum ];
			qBuf_[ threadNum ].push_back( 
				Qinfo( oi, bindIndex, threadNum, vec.size(), size ) );
			if ( size > 0 ) {
				vec.insert( vec.end(), arg, arg + size );
			}
	}
}

/// Utility variant that adds two args.
void Qinfo::addToQ( const ObjId& oi, 
	BindIndex bindIndex, ThreadId threadNum,
	const double* arg1, unsigned int size1, 
	const double* arg2, unsigned int size2 )
{
	if ( oi.element()->hasMsgs( bindIndex ) ) {
			assert( dBuf_.size() > threadNum );
			assert( qBuf_.size() > threadNum );
			vector< double >& vec = dBuf_[ threadNum ];
			qBuf_[ threadNum ].push_back( 
				Qinfo( oi, bindIndex, threadNum, vec.size(), size1 + size2 ) );
			if ( size1 > 0 )
				vec.insert( vec.end(), arg1, arg1 + size1 );
			if ( size2 > 0 )
				vec.insert( vec.end(), arg2, arg2 + size2 );
	}
}

// Static function
void Qinfo::addDirectToQ( const ObjId& src, const ObjId& dest, 
	ThreadId threadNum,
	FuncId fid, 
	const double* arg, unsigned int size )
{
	static const unsigned int ObjFidSizeInDoubles = 
		1 + ( sizeof( ObjFid ) - 1 ) / sizeof( double );
		assert( dBuf_.size() > threadNum );
		assert( qBuf_.size() > threadNum );

		vector< double >& vec = dBuf_[ threadNum ];
		qBuf_[ threadNum ].push_back(
			Qinfo( src, DirectAdd, threadNum, vec.size(), size ) );
		ObjFid ofid = { dest, fid, size, 1 };
		const double* ptr = reinterpret_cast< const double* >( &ofid );
		vec.insert( vec.end(), ptr, ptr + ObjFidSizeInDoubles );
		if ( size > 0 ) {
			vec.insert( vec.end(), arg, arg + size );
		}
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

		vector< double >& vec = dBuf_[ threadNum ];
		qBuf_[ threadNum ].push_back(
			Qinfo( src, DirectAdd, threadNum, vec.size(), size1 + size2 ) );
		ObjFid ofid = { dest, fid, size1 + size2, 1 };
		const double* ptr = reinterpret_cast< const double* >( &ofid );
		vec.insert( vec.end(), ptr, ptr + ObjFidSizeInDoubles );
		if ( size1 > 0 )
			vec.insert( vec.end(), arg1, arg1 + size1 );
		if ( size2 > 0 )
			vec.insert( vec.end(), arg2, arg2 + size2 );
}

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
		qBuf_[ threadNum ].push_back( 
			Qinfo( src, DirectAdd, threadNum, dBuf_[threadNum].size(),
				entrySize * numEntries ) );
	
		ObjFid ofid = { dest, fid, entrySize, numEntries };
		const double* ptr = reinterpret_cast< const double* >( &ofid );
		vector< double >& vec = dBuf_[ threadNum ];
		vec.insert( vec.end(), ptr, ptr + ObjFidSizeInDoubles );
		vec.insert( vec.end(), arg, arg + entrySize * numEntries );
}

/**
 * This is called by the 'handle<op>' functions in Shell. So it is always
 * either in phase2/3, or within clearStructuralQ() which is on a single 
 * thread inside Barrier1.
 */
bool Qinfo::addToStructuralQ() const
{
	static const unsigned int ObjFidSizeInDoubles =
		1 + ( sizeof( ObjFid ) - 1 ) / sizeof( double );

	bool ret = 0;
		if ( !( threadNum_ == ScriptThreadNum )){
			if ( isDummy() )
				cout << "d" << flush;
			structuralQinfo_.push_back( *this );
			/*
			const Qinfo* qstart = 
				reinterpret_cast< const Qinfo* >( inQ_ + 2);
			unsigned int qnum = this - qstart;
			unsigned int nq = inQ_[0];
			unsigned int nextData = 0;
			if ( qnum < nq - 1 )
				nextData = ( this + 1 )->dataIndex_;
			else
				nextData = inQ_[0];
			unsigned int dataSize = nextData - dataIndex_;
			assert( dataSize < ( 1 << ( sizeof( unsigned int ) - 1 ) ) );

			structuralQinfo_.back().dataIndex_ = structuralQdata_.size();
			*/
			structuralQinfo_.back().dataIndex_ = structuralQdata_.size();
			structuralQinfo_.back().threadNum_ = ScriptThreadNum;
			if ( isDirect() ) {
				structuralQdata_.insert( structuralQdata_.end(), 
					inQ_ + dataIndex_, inQ_ + dataIndex_ + 
					ObjFidSizeInDoubles + dataSize_ );
			} else {
				structuralQdata_.insert( structuralQdata_.end(), 
					inQ_ + dataIndex_, inQ_ + dataIndex_ + dataSize_ );
			}
			ret = 1;
		}
	return ret;
}

/// Static func
void Qinfo::emptyAllQs()
{
	inQ_[0] = 0;
	inQ_[1] = 0;
	for ( unsigned int i = 0; i <= Shell::numProcessThreads(); ++i ) {
		qBuf_[i].resize( 0 );
		dBuf_[i].resize( 0 );
	}
}

// Static function. Used only during single-thread tests, when the
// main thread-handling loop is inactive.
// Called after all the 'send' functions have been done. Typically just
// one is pending.
// Static func.
void Qinfo::clearQ( ThreadId threadNum )
{
	swapQ();
	readQ( threadNum );
}

// static function
double* Qinfo::sendQ()
{
	return &q0_[0];
}

// static function
double* Qinfo::mpiRecvQ()
{
	return mpiRecvQ_;
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
	
	if ( Shell::numNodes() > 1 ) {
		history_.resize( Shell::numNodes() );
		// blockSize_.resize( Shell::numNodes(), reserve * blockMargin_ );
		blockSize_.resize( Shell::numNodes(), 20 );
		for ( unsigned int i = 0; i < Shell::numNodes(); ++i ) {
			// history_[i].resize( historySize_, reserve );
			history_[i].resize( historySize_, 20 );
		}
		
		/*
		mpiQ0_.resize( reserve * 100 );
		mpiQ1_.resize( reserve * 100 );
		*/
		mpiQ0_.resize( 20 );
		mpiQ1_.resize( 20 );
		mpiRecvQ_ = &mpiQ0_[0];
	}
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
	assert( threadIndex >= 1 );
	threadIndex -= 1;
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
		if ( numThreads <= 1 )
			start->setInited();
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
