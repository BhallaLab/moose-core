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
				qId( 0 ), threadIndexInGroup( 0 ),
				barrier( 0 )
			{;}
		double dt;
		double currTime;
		unsigned int numThreads;
		unsigned int threadId;
		Qid qId;
		unsigned int threadIndexInGroup;
		unsigned int node;
		void* barrier;
};

typedef ProcInfo* ProcPtr;
