/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class ProcInfo
{
	public:
		ProcInfo() 
			: dt( 1.0 ), currTime( 0.0 ), numThreads( 1 ), threadId( 0 ),
//				outQid( 0 ), 
				threadIndexInGroup( 0 ), 
				numThreadsInGroup( 1 ), 
				nodeIndexInGroup( 0 ),
				numNodesInGroup( 1 ), 
				groupId( 0 ),
				isMpiThread( 0 ),
				barrier1( 0 ),
				barrier2( 0 )
			{;}
		double dt;
		double currTime;
		unsigned int numThreads; // Includes the mpiThread, if any
		unsigned int threadId;
//		Qid outQid;	// Index of outQ to use.
		unsigned int threadIndexInGroup;
		unsigned int numThreadsInGroup; // compute threads
		unsigned int nodeIndexInGroup;
		unsigned int numNodesInGroup;
		unsigned int groupId;
		bool isMpiThread;
		void* barrier1;
		void* barrier2;
};

typedef ProcInfo* ProcPtr;
