/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This class manages data for queues. This is tricky because it has
 * to present a single contiguous data block for the purposes of 
 * MPI transfer, but also has to handle expandable internal blocks, one for
 * each thread. In order to do this safely, it has to ensure that there
 * is a reasonable amount of safe space between blocks. It also then
 * has to guarantee that no threads are accessing the data blocks when
 * it has to resize.
 */
class Qvec
{
	public:
		Qvec( unsigned int numThreads ); /// Create and allocate Qvec

		/**
		 * Adds data to queue on specified thread. Ensures that there
		 * is space (threadOverlapProtection) between thread data blocks.
		 * Also does the right thing with threads to prevent problems
		 * when resizing.
		 */
		void push_back( unsigned int thread, const Qinfo* q, const char* arg );

		/**
		 * Clear out all the thread data. It does so without changing
		 * the spacing between blocks, so that the historical buffer size
		 * is preserved.
		 */
		void clear();

		/**
		 * Stitch together the threads with intervening Qinfo entries
		 * so that ReadBuf will go through the whole buffer seamlessly
		 * This is also a good place to accumulate diagnostics on 
		 * message traffic, so as to adjust the size of the Qs.
		 */
		void stitch();

		/**
		 * Returns start of data block. Must NOT be used concurrently
		 * with push_back, as the data will be messed up.
		 */
		const char* data() const;

		/**
		 * Returns start of data block. Used for MPI data transfers
		 * which need the data block address to write into
		 */
		char* writableData();

		/**
		 * Returns size in bytes of used data block. Note that the
		 * internal allocation may be bigger. Also note that this size
		 * is from the start of the data on thread0 to the end of data on
		 * the last thread. There will usually be gaps in between, 
		 * paddedout using dummy Qinfos to skip over the gaps.
		 */
		unsigned int dataQsize() const;

		/**
		 * Returns allocated size of entire data vector in bytes.
		 */
		unsigned int allocatedSize() const;

		/**
		 * Returns number of threads handled by Qvec.
		 */
		unsigned int numThreads() const;

		/**
		 * Returns number of message Q entries on given thread
		 */
		unsigned int numEntries( unsigned int threadNum ) const;

		/**
		 * Returns total number of message Q entries on all threads
		 */
		unsigned int totalNumEntries() const;

		/**
		 * test function for Qvec
		 */
		static void testQvec();
	private:
		vector< vector< char > > data_;

		vector< char > linearData_;
		static const unsigned int threadQreserve;
};
