/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "PostMaster.h"
#include "../shell/Shell.h"

const unsigned int TgtInfo::headerSize = 
		1 + ( sizeof( TgtInfo ) - 1 )/sizeof( double );

const unsigned int PostMaster::reserveBufSize = 4096;
const int PostMaster::MSGTAG = 1;
const int PostMaster::SETTAG = 2;
const int PostMaster::CONTROLTAG = 3;
const int PostMaster::DIETAG = 4;
PostMaster::PostMaster()
		: 
				recvBufSize_( reserveBufSize ),
				setSendBuf_( reserveBufSize, 0 ),
				setRecvBuf_( reserveBufSize, 0 ),
				sendBuf_( Shell::numNodes() ),
				recvBuf_( Shell::numNodes() ),
				sendSize_( Shell::numNodes(), 0 ),
				doneIndices_( Shell::numNodes(), 0 ),
				isSetSent_( 1 ), // Flag. Have any pending 'set' gone?
				isSetRecv_( 0 ), // Flag. Has some data come in?
				setSendSize_( 0 ),
				numSendDone_( 0 )
{
	for ( unsigned int i = 0; i < Shell::numNodes(); ++i ) {
		sendBuf_[i].resize( reserveBufSize, 0 );
	}
#ifdef USE_MPI
	recvReq_.resize( Shell::numNodes() );
	sendReq_.resize( Shell::numNodes() );
	for ( unsigned int i = 0; i < Shell::numNodes(); ++i ) {
		// Set up the Recv already for later sends. This might be a problem
		// for some polling-based implementations, but let's try for now.
		MPI_Status temp;
		temp.MPI_SOURCE = temp.MPI_TAG = temp.MPI_ERROR = 0;
		doneStatus_.resize( Shell::numNodes(), temp );
		if ( i != Shell::myNode() ) {
			recvBuf_[i].resize( recvBufSize_, 0 );
			MPI_Irecv( &recvBuf_[i][0], recvBufSize_, MPI_DOUBLE,
				i, MSGTAG, MPI_COMM_WORLD,
				&recvReq_[i]
			);
		}
	}
#endif
}

///////////////////////////////////////////////////////////////
// Moose class stuff.
///////////////////////////////////////////////////////////////
const Cinfo* PostMaster::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ReadOnlyValueFinfo< PostMaster, unsigned int > numNodes(
			"numNodes",
			"Returns number of nodes that simulation runs on.",
			&PostMaster::getNumNodes
		);
		static ReadOnlyValueFinfo< PostMaster, unsigned int > myNode(
			"myNode",
			"Returns index of current node.",
			&PostMaster::getMyNode
		);
		static ValueFinfo< PostMaster, unsigned int > bufferSize(
			"bufferSize",
			"Size of the send a receive buffers for each node.",
			&PostMaster::setBufferSize,
			&PostMaster::getBufferSize
		);
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new EpFunc1< PostMaster, ProcPtr >( &PostMaster::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new EpFunc1< PostMaster, ProcPtr >( &PostMaster::reinit ) );

		//////////////////////////////////////////////////////////////
		// SharedFinfo Definitions
		//////////////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* postMasterFinfos[] = {
		&numNodes,	// ReadOnlyValue
		&myNode,	// ReadOnlyValue
		&bufferSize,	// ReadOnlyValue
		&proc		// SharedFinfo
	};

	static Dinfo< PostMaster > dinfo;
	static Cinfo postMasterCinfo (
		"PostMaster",
		Neutral::initCinfo(),
		postMasterFinfos,
		sizeof( postMasterFinfos ) / sizeof ( Finfo* ),
		&dinfo
	);

	return &postMasterCinfo;
}

//
/**
 * PostMaster class: handles cross-node messaging using MPI.
 * Identical to the Process call: sends out what needs to go, and then
 * waits for any incoming messages and passes them on.
 */
void PostMaster::reinit( const Eref& e, ProcPtr p )
{
#ifdef USE_MPI
	for ( unsigned int i = 0; i < Shell::numNodes(); ++i )
	{
		if ( i == Shell::myNode() ) continue;
		// MPI_scatter would have been better but it doesn't allow
		// one to post larger recvs than the actual data sent.
		MPI_Isend( 
			&sendBuf_[i][0], sendSize_[i], MPI_DOUBLE,
			i, 		// Where to send to.
			MSGTAG, MPI_COMM_WORLD,
			&sendReq_[i]
		);
		clearPending(); // Try to interleave communications.
	}
	while ( numSendDone_ < Shell::numNodes() )
		clearPending();
	numSendDone_ = 0;
#endif
}

void PostMaster::process( const Eref& e, ProcPtr p )
{
#ifdef USE_MPI
	for ( unsigned int i = 0; i < Shell::numNodes(); ++i )
	{
		if ( i == Shell::myNode() ) continue;
		// MPI_scatter would have been better but it doesn't allow
		// one to post larger recvs than the actual data sent.
		MPI_Isend( 
			&sendBuf_[i][0], sendSize_[i], MPI_DOUBLE,
			i, 		// Where to send to.
			MSGTAG, MPI_COMM_WORLD,
			&setSendReq_
		);
		clearPending(); // Try to interleave communications.
	}
	while ( numSendDone_ < Shell::numNodes() )
		clearPending();
	numSendDone_ = 0;
#endif
}

void PostMaster::clearPending()
{
	if ( Shell::numNodes() == 1 )
		return;
	clearPendingSet();
	clearPendingSend();
}

void PostMaster::clearPendingSet()
{
#ifdef USE_MPI
	// isSetSent_ is checked before doing another x-node set operation
	// in dispatchSetBuf.
	if ( !isSetSent_ ) {
		MPI_Test( &setSendReq_, &isSetSent_, &setSendStatus_ );
		assert ( isSetSent_ != MPI_UNDEFINED );
	}

	MPI_Test( &setRecvReq_, &isSetRecv_, &setRecvStatus_ );
	if ( isSetRecv_ && isSetRecv_ != MPI_UNDEFINED )
	{
		// Handle arrived Set call
		const TgtInfo* tgt = 
				reinterpret_cast< const TgtInfo * >( &setRecvBuf_[0] );
		const Eref& e = tgt->eref();
		const OpFunc *op = OpFunc::lookop( tgt->bindIndex() );
		assert( op );
		op->opBuffer( e, &setRecvBuf_[ TgtInfo::headerSize ] );
		// Now the operation is done. Re-post recv.
		MPI_Irecv( &setRecvBuf_[0], recvBufSize_, MPI_DOUBLE,
						MPI_ANY_SOURCE,
						SETTAG, MPI_COMM_WORLD,
						&setRecvReq_
		);
		isSetRecv_ = 0;
	}
#endif // USE_MPI
}

void PostMaster::clearPendingSend()
{
#ifdef USE_MPI
	int done = 0;
	MPI_Testsome( Shell::numNodes() -1, &recvReq_[0], &done, 
					&doneIndices_[0], &doneStatus_[0] );
	if ( done == MPI_UNDEFINED )
		return;
	for ( int i = 0; i < done; ++i ) {
		unsigned int recvNode = doneIndices_[i];
		if ( recvNode >= Shell::myNode() )
			recvNode += 1; // Skip myNode
		int recvSize = 0;
		MPI_Get_count( &doneStatus_[i], MPI_DOUBLE, &recvSize );
		int j = 0;
		assert( recvSize <= static_cast< int >( recvBufSize_ ) );
		double* buf = &recvBuf_[ recvNode ][0];
		while ( j < recvSize ) {
			const TgtInfo* tgt = reinterpret_cast< const TgtInfo * >( buf );
			const Eref& e = tgt->eref();
			const Finfo *f = 
				e.element()->cinfo()->getSrcFinfo( tgt->bindIndex() );
			buf += TgtInfo::headerSize;
			const SrcFinfo* sf = dynamic_cast< const SrcFinfo* >( f );
			assert( sf );
			sf->sendBuffer( e, buf );
			buf += tgt->dataSize();
			j += TgtInfo::headerSize + tgt->dataSize();
			assert( buf - &recvBuf_[recvNode][0] == j );
		}
		// Post the next Irecv.
		MPI_Irecv( &recvBuf_[recvNode][0],
						recvBufSize_, MPI_DOUBLE, 
						recvNode,
						MSGTAG, MPI_COMM_WORLD,
						&recvReq_[ recvNode ] 
				 );
	}
	numSendDone_ += done;
#endif
}

///////////////////////////////////////////////////////////////
// Data transfer and fillup operations.
///////////////////////////////////////////////////////////////

double* PostMaster::addToSendBuf( const Eref& e, unsigned int bindIndex,
		unsigned int size )
{
	unsigned int node = e.fieldIndex(); // nasty evil wicked hack
	unsigned int end = sendSize_[node];
	if ( end + TgtInfo::headerSize + size > recvBufSize_ ) {
		// Here we need to activate the fallback second send which will
		// deal with the big block. Also various routines for tracking
		// send size so we don't get too big or small.
		cerr << "Error: PostMaster::addToSendBuf on node " << 
				Shell::myNode() << 
				": Data size (" << size << ") goes past end of buffer\n";
		assert( 0 );
	}
	TgtInfo* tgt = reinterpret_cast< TgtInfo* >( &sendBuf_[node][end] );
	tgt->set( e.id(), e.dataIndex(), bindIndex, size );
	end += TgtInfo::headerSize;
	sendSize_[node] = end + size;
	return &sendBuf_[node][end];
}

double* PostMaster::addToSetBuf( const Eref& e, unsigned int opIndex, 
						unsigned int size )
{
	unsigned int node = e.getNode(); // The Eref is an offnode one.
	assert( node != Shell::myNode() );
	if ( TgtInfo::headerSize + size > recvBufSize_ ) {
		// Here we need to activate the fallback second send which will
		// deal with the big block. Also various routines for tracking
		// send size so we don't get too big or small.
		cerr << "Error: PostMaster::addToSetBuf on node " << 
				Shell::myNode() << 
				": Data size (" << size << ") goes past end of buffer\n";
		assert( 0 );
	}
	TgtInfo* tgt = reinterpret_cast< TgtInfo* >( &setSendBuf_[0] );
	tgt->set( e.id(), e.dataIndex(), opIndex, size );
	unsigned int end = TgtInfo::headerSize;
	setSendSize_ = end + size;
	return &setSendBuf_[end];
}

void PostMaster::dispatchSetBuf( const Eref& e )
{
	assert ( e.element()->isGlobal() || e.getNode() != Shell::myNode() );
	while ( isSetSent_ == 0 ) { // Can't add a set while old set is pending
		clearPending();
	}
	isSetSent_ = 0;
#ifdef USE_MPI
	if ( e.element()->isGlobal() ) {
		for ( unsigned int i = 0; i < Shell::numNodes(); ++i ) {
			if ( i != Shell::myNode() ) {
			// A bcast would be marginally more efficient, but would need
			// us to inform all target nodes to expect one. So just do
			// multiple sends.
				MPI_Isend( 
					&setSendBuf_[0], setSendSize_, MPI_DOUBLE,
					i, 		// Where to send to.
					SETTAG, MPI_COMM_WORLD,
					&setSendReq_
		// Need to monitor all the sends to make sure they all complete 
		// before permitting another 'set'
				);
			}
		}
	} else {
		MPI_Isend( 
			&setSendBuf_[0], setSendSize_, MPI_DOUBLE,
				e.getNode(), 		// Where to send to.
				SETTAG, MPI_COMM_WORLD,
				&setSendReq_
		);
	}
#endif
}

///////////////////////////////////////////////////////////////
// Fields
///////////////////////////////////////////////////////////////

unsigned int PostMaster::getNumNodes() const
{
	return Shell::numNodes();
}

unsigned int PostMaster::getMyNode() const
{
	return Shell::myNode();
}

unsigned int PostMaster::getBufferSize() const
{
	if ( sendBuf_.size() == 0 )
		return 0;

	return sendBuf_[0].size();
}

void PostMaster::setBufferSize( unsigned int size )
{
	for ( unsigned int i =0; i < sendBuf_.size(); ++i )
		sendBuf_[i].resize( size );
}
