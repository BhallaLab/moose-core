/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SPARSE_MSG_H
#define _SPARSE_MSG_H

/**
 * This is a general message type optimized for sparse matrix like
 * projection patterns. For each source object[DataId] there can be a
 * target object[DataId].
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
class SparseMsg: public Msg
{
	public:
		SparseMsg( Element* e1, Element* e2 );
		~SparseMsg() {;}

		void exec( Element* target, const char* arg) const;

		/**
		 * Set up connections randomly. Probability should be low to keep
		 * it sparse.
		 * Returns # of connections.
		 */
		unsigned int randomConnect( double probability );

		/**
		 * Creates a message between e1 and e2, with connections
		 * ocdurring at the specified probability
		 */
		static bool add( Element* e1, const string& srcField, 
			Element* e2, const string& destField, double probability );
	private:
		SparseMatrix< unsigned int > matrix_;
};

#endif // _SPARSE_MSG_H
