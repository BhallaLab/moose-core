/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// The # of queues is around 2x the # of threads (including offnode ones)
// in the hardware. There do exist machines where a short will not suffice,
// but not too many of them at this time!
typedef unsigned short Qid;


/**
 * SimGroup keeps track of queues and threads. Each SimGroup operates a set
 * of objects that are closely coupled, so that they should share an inQ.
 * This means that all threads within the SimGroup will examine the same
 * inQ, with a high probability that the events on it will be relevant to
 * all the threads. Each thread within the SimGroup will dump its output
 * events into a private outQ.
 *
 * Each SimGroup has 
 *  startThread: the identifier (index) of the first thread in the group.
 *  numGroupThreads: the number of threads in this group.
 * 	inQ: Index of input queue. Equal to groupIndex at this point.
 * 	outQ, Starting index of output queues. There are numGroupThreads of them
 * 	mpiQ: index of queue used for arriving stuff from other nodes. Each
 * 		SimGroup has one per remote node.
 *
 * If I put the inQs separately, I could index the outQs directly from
 * the threadId. That would be handy. The inQ would also be easier, it would
 * just be the groupId. But I still need to look up the groupId from the
 * threadId. I could put it in the ProcInfo.
 *
 */
class SimGroup {
	public: 
		SimGroup( unsigned short startNode, unsigned short endNode_,
			unsigned int bufSize )
			: startNode_( startNode ), endNode_( endNode_ ), 
			bufSize_( bufSize )
			{;}
		unsigned short startNode() const {
			return startNode_;
		}
		unsigned short endNode() const {
			return endNode_;
		}

		unsigned int bufSize() const {
			return bufSize_;
		}

		void setBufSize( unsigned int val ) {
			bufSize_ = val;
		}

		/**
		 * returns Qid for the absolute threadId
		Qid outQ( unsigned int threadId, unsigned int groupIndex ) const {
			Qid ret = threadId + groupIndex + 1;
			assert( ret - startThread_ < numThreads );
			return( ret );
		}
		 */
	private:
		/**
		 * Starting node index for this group
		 */
		unsigned short startNode_;
		/**
		 * End node index for this group. 1+last node in group.
		 */
		unsigned short endNode_;

		/**
		 * Current data buffer size for MPI data transfers in this group
		 * We assume that all nodes within group use equal data sizes.
		 * This number goes up and down based on recent traffic.
		 */
		unsigned int bufSize_;

};

