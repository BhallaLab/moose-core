/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _PSPARSE_MSG_H
#define _PSPARSE_MSG_H



/**
 * This is a parallelized sparse message.
 * It is a general message type optimized for sparse matrix like
 * projection patterns. For each source object[DataId] there can be a
 * target object[DataId].
 * For parallel/multithreaded use, we need to guarantee that all requests
 * to the same target object (and all its synapses) are on the same queue.
 * So it builds up a separate SparseMatrix for each thread.
 *
 * It has a specialized version of exec, to select the appropriate
 * SparseMatrix. It goes through the entire set of incoming events.
 *
 * It has a function to do the node/thread decomposition to generate an
 * equivalent of the original sparse matrix, but using only the appropriate
 * RNG seed.
 *
 * A typical case is from an array of IntFire objects to an array of 
 * Synapses, which are array fields of IntFire objects.
 * The sparse connectivity maps between the source IntFire and target
 * Synapses.
 * The location of the entry in the sparse matrix provides the index of
 * the target IntFire.
 * The data value in the sparse matrix provides the index of the Synapse
 * at that specific connection.
 * This assumes that only one Synapse mediates a given connection between
 * any two IntFire objects.
 *
 * It is optimized for input coming on Element e1, and going to Element e2.
 * If you expect any significant backward data flow, please use 
 * BiSparseMsg.
 * It can be modified after creation to add or remove message entries.
 */
class PsparseMsg: public SparseMsg
{
	friend void initMsgManagers(); // for initializing Id.
	public:
		PsparseMsg( Element* e1, Element* e2 );
		~PsparseMsg();

		void exec( const char* arg, const ProcInfo* p ) const;


		/**
		 * Creates a message between e1 and e2, with connections
		 * ocdurring at the specified probability
		 */
		static bool add( Element* e1, const string& srcField, 
			Element* e2, const string& destField, double probability,
			unsigned int numThreadsInGroup );
		
		void loadBalance( unsigned int numThreads );
		void loadUnbalance();

		Id id() const;
	private:
		unsigned int numThreads_; // Number of threads to partition
		unsigned int nrows_; // The original size of the matrix.
		static Id id_; // The Element that manages Psparse Msgs.
};

extern void sparseMatrixBalance( 
	unsigned int numThreads, SparseMatrix< unsigned int >& matrix );

#endif // _PSPARSE_MSG_H
