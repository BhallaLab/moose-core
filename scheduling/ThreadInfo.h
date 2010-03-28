/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _THREAD_INFO_H
#define _THREAD_INFO_H

class ThreadInfo
{
	public:
		Element* clocke;
		Qinfo* qinfo;
		double runtime;
		unsigned int threadId;
		unsigned int threadIndexInGroup;
		unsigned int groupId;
		unsigned int nodeIndexInGroup;
		Qid outQid;
		pthread_mutex_t* sortMutex; // Protects sorting of TickPtrs
		pthread_mutex_t* timeMutex; // Protects time increment in TickPtrs
};

#endif // _THREAD_INFO_H
