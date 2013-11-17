/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

const unsigned int TgtInfo::headerSize = 
		1 + ( sizeof( TgtInfo ) - 1 )/sizeof( double );

/**
 * PostMaster class: handles cross-node messaging using MPI.
 */
void PostMaster::reinit( const Eref& e, ProcPtr p )
{
	for ( unsigned int i = 0; i < numNodes_; ++i )
	{
		if ( i == myNode() ) continue;
		MPI_IRecv( recvbuf_[i], recvBufSize, MPI_DOUBLE,
			i, WORKTAG, MPI_COMM_WORLD,
			recvReq[i]
	}
		// MPI_scatter would have been better but it doesn't allow
		// one to post larger recvs than the actual data sent.

		while ( numDone < numNodes_ )
			numDone += clearPending();
	}
}

void PostMaster::process( const Eref& e, ProcPtr p )
{
	for ( unsigned int i = 0; i < numNodes_; ++i )
	{
		if ( i == myNode() ) continue;
		MPI_Isend( sendbuf_[i], bufsize_[i], MPI_DOUBLE,
			i, WORKTAG, MPI_COMM_WORLD,
			&sendReq[i]
		);
		// MPI_scatter would have been better but it doesn't allow
		// one to post larger recvs than the actual data sent.

		while ( numDone < numNodes_ )
			numDone += clearPending();
	}
}

int PostMaster::clearPending()
{
	int ret = MPI_Testsome( numNodes_ -1, recvReq_, &done, 
					doneIndices, doneStatus );
	if ( done = MPI_UNDEFINED )
		return 0;
	for ( int i = 0; i < done; ++i ) {
		int recvNode = doneIndices[i];
		if ( recvNode >= myrank )
			recvNode += 1; // Skip myrank

		// Here we go through the recvBuf to deliver the received msgs.
		int recvSize = foo;
		const double* buf = &recvBuf[ recvNode ][0];
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
}

double* PostMaster::addtoSendBuf( const Eref& e, unsigned int bindIndex,
		unsigned int size )
{
	unsigned int node = e.fieldIndex(); // nasty evil wicked hack
	unsigned int end = sendSize_[node];
	TgtInfo* tgt = reinterpret_cast< TgtInfo* >( &sendbuf_[node][end] );
	tgt->set( e.id(), e.dataIndex(), bindIndex, size );
	end += TgtInfo::headerSize;
	sendSize_[node] = end + size;
	// Need to do stuff here when sendSize gets bigger than the buffer.
	return &sendbuf_[node][end];
}

void PostMaster::send()
{
	
}
