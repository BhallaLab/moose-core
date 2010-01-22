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

class SimGroup {
	public: 
		SimGroup( unsigned short nt, unsigned short si )
			: numThreads( nt ), startIndex( si )
			{;}
		unsigned short numThreads; // Number of threads in this group.
		unsigned short startIndex; // Index of first thread, used for inQ.

		/**
		 * returns Qid for the thread specified within this group
		Qid outQ( unsiged int relativeThreadId ) const {
			assert( relativeThreadId < numThreads );
			return startIndex + threadId + 1;
		}
		 */

		/**
		 * returns Qid for the absolute threadId
		 */
		Qid outQ( unsigned int threadId, unsigned int groupIndex ) const {
			Qid ret = threadId + groupIndex + 1;
			assert( ret - startIndex < numThreads );
			return( ret );
		}
		// Stuff here for off-node queues.
};

