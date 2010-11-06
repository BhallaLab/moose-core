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
		unsigned int usedSize() const;

		/**
		 * Returns allocated size of entire data vector in bytes.
		 */
		unsigned int allocatedSize() const;

		/**
		 * test function for Qvec
		 */
		static void testQvec();
	private:
		unsigned int numThreads_;


		vector < char > data_;

		/**
		 * Index of start of each chunk of data for each thread
		 * Used only when filling up data, and when stitching
		 * together the thread sections for reading.
		 */
		vector < unsigned int > threadBlockStart_;

		/**
		 * Index of end of each chunk of data for each thread
		 */
		vector < unsigned int > threadBlockEnd_;

		/**
		 * Number of bytes between thread blocks.
		 * Threads may step on each other's toes when writing to 
		 * adjacent memory. This buffer protects against this.
		 */
		static const unsigned int threadOverlapProtection;

		/**
		 * Assigns preliminary size allocation. Each thread gets the same
		 * initial block.
		 */
		static const unsigned int threadQreserve;
};
