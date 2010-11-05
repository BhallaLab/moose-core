/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
class ProcInfo {
	public:
		ProcInfo()
			: 
				threadIndexInGroup( 0 ),
				numThreadsInGroup( 1 ),
				groupId( 0 ),
				myNode( 0 ),
				numNodes( 1 )
		{;}

		unsigned int threadIndexInGroup;
		unsigned int numThreadsInGroup;
		unsigned int groupId;
		unsigned int myNode;
		unsigned int numNodes;

		// pthread_barrier_t* barrier1;
		// pthread_barrier_t* barrier2;
		FuncBarrier* barrier1;
		FuncBarrier* barrier2;
		pthread_barrier_t* barrier3;
		pthread_mutex_t* shellSendMutex;
		pthread_cond_t* parserBlockCond;
};

