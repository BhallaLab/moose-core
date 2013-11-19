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
#ifdef USE_MPI
#include <mpi.h>
#endif

const unsigned int TgtInfo::headerSize = 
		1 + ( sizeof( TgtInfo ) - 1 )/sizeof( double );

PostMaster::PostMaster()
		: 
				recvBufSize_( 1 ),
				sendBuf_( Shell::numNodes() ),
				recvBuf_( Shell::numNodes() ),
				sendSize_( Shell::numNodes(), 0 )
{;}

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
 */
void PostMaster::reinit( const Eref& e, ProcPtr p )
{
#ifdef USE_MPI
	for ( unsigned int i = 0; i < Shell::numNodes(); ++i )
	{
		if ( i == Shell::myNode() ) continue;
		MPI_IRecv( recvBuf_[i], recvBufSize_, MPI_DOUBLE,
			i, WORKTAG, MPI_COMM_WORLD,
			recvReq[i]
	}
		// MPI_scatter would have been better but it doesn't allow
		// one to post larger recvs than the actual data sent.

		while ( numDone < Shell::numNodes() )
			numDone += clearPending();
	}
#endif
}

void PostMaster::process( const Eref& e, ProcPtr p )
{
#ifdef USE_MPI
	for ( unsigned int i = 0; i < Shell::numNodes(); ++i )
	{
		if ( i == Shell::myNode() ) continue;
		MPI_Isend( sendBuf_[i], sendSize_[i], MPI_DOUBLE,
			i, WORKTAG, MPI_COMM_WORLD,
			&sendReq[i]
		);
		// MPI_scatter would have been better but it doesn't allow
		// one to post larger recvs than the actual data sent.

		while ( numDone < Shell::numNodes() )
			numDone += clearPending();
	}
#endif
}

int PostMaster::clearPending()
{
	int done = 0;
#ifdef USE_MPI
	MPI_Testsome( Shell::numNodes() -1, recvReq_, &done, 
					doneIndices, doneStatus );
	if ( done = MPI_UNDEFINED )
		return 0;
	for ( int i = 0; i < done; ++i ) {
		int recvNode = doneIndices[i];
		if ( recvNode >= myrank )
			recvNode += 1; // Skip myrank

		// Here we go through the recvBuf to deliver the received msgs.
		int recvSize = foo;
		double* buf = &recvBuf_[ recvNode ][0];
		while ( ) {
			const TgtInfo* tgt = reinterpret_cast< const TgtInfo * >( buf );
			const Eref& e = tgt->eref();
			const Finfo *f = 
				e.element()->cinfo()->getSrcFinfo( tgt->srcFid );
			buf += TgtInfo::headerSize;
			f->sendBuffer( e, buf );
			buf += tgt->dataSize();
		}
	}
#endif
	return done;
}

double* PostMaster::addToSendBuf( const Eref& e, unsigned int bindIndex,
		unsigned int size )
{
	unsigned int node = e.fieldIndex(); // nasty evil wicked hack
	unsigned int end = sendSize_[node];
	TgtInfo* tgt = reinterpret_cast< TgtInfo* >( &sendBuf_[node][end] );
	tgt->set( e.id(), e.dataIndex(), bindIndex, size );
	end += TgtInfo::headerSize;
	sendSize_[node] = end + size;
	// Need to do stuff here when sendSize gets bigger than the buffer.
	return &sendBuf_[node][end];
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
