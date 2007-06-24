/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ID_MANAGER_H
#define _ID_MANAGER_H

class Element;

/**
 * First pass at a way to keep track of what each node is doing.
 */
class NodeLoad
{
	public:
		unsigned int memory;
		unsigned int cpu;
		unsigned int messages;
		unsigned int pollWaits;
};

/**
 * This class coordinates closely with the PostMaster to set up 
 * policies for distributing ids across nodes. On the master node this
 * actually assigns ids to requested objects. On the slave nodes, if
 * any, this helps to keep ids updated.
 */
class IdManager
{
	public:
		IdManager();

		void setNodes( unsigned int myNode, unsigned int numNodes );

		//////////////////////////////////////////////////////////////////
		// Id creation
		//////////////////////////////////////////////////////////////////
		/**
		 * Allocates a new scratch id on the current node and returns it.
		 */
		unsigned int scratchId();


		/**
		 * Generates an id for a child object, using parent id
		 * and name as heuristics. This is the key interface point
		 * for assigning nodes. 
		 * Main use cases:
		 * - Global object: Parent is global object.
		 * - Local node object: Parent is in local node set.
		 * - Remote node object: Parent is on a remote node.
		 * - New remote node object: Parent is local, but name indicates
		 *   a new group, such as kinetics.
		 */
		unsigned int childId( unsigned int parent );
		
		//////////////////////////////////////////////////////////////////
		// Id info
		//////////////////////////////////////////////////////////////////

		/**
		 * Returns the Element pointed to by the index.
		 * If it is off-node, returns an allocated wrapper element with 
		 * postmaster and id information. Calling func has to free it.
		 * This wrapper element may also point to UNKNOWN NODE, in which
		 * case the master node IdManager has to figure out where it
		 * belongs.
		 * Returns 0 on failure.
		 */
		Element* getElement( unsigned int index ) const;

		/**
		 * Assigns element to specified index. On master node, the
		 * index should be consecutive. On slave nodes this assignment
		 * means that intermediate blank ids need to be set to UNKNOWN_NODE.
		 * Returns true on success.
		 * If Element e is zero, it assumes we are clearing out the 
		 * element. 
		 * \todo This is where to put in code for storing a list of reusable
		 * ids.
		 */
		bool setElement( unsigned int index, Element* e );

		/**
		 * Checks which node specified element is on.
		 */
		unsigned int findNode( unsigned int index ) const;
		
		/**
		 * Returns the most recently created id on current node
		 */
		unsigned int lastId();

		/**
		 * Tells us if specified index is out of range of the allocated
		 * ids on the element list.
		 */
		bool outOfRange( unsigned int index ) const;

		/**
		 * True if specified index is for a scratch id
		 */
		bool isScratch( unsigned int index ) const;


		/**
		 * Returns local node
		 */
		// unsigned int myNode() const;

	private:

		/**
		 * This function moves all the scratch ids on this node onto
		 * regular Ids, after consulting with the master node.
		 */
		void regularizeScratch();

		/**
		 * These keep track of size of cluster
		 */
		unsigned int myNode_;
		unsigned int numNodes_;
		
		/**
		 * Keep track of assigned load on each node
		 * This is the job only of the master node.
		 */
		vector< NodeLoad > nodeLoad;
		vector< Element* > elementList_;

		/**
		 * This keeps track of the # of scratch ids. Only on slave nodes.
		 */
		unsigned int scratchIndex_;

		/**
		 * This keeps tracks of the # of main ids. On all nodes.
		 */
		unsigned int mainIndex_;

		/**
		 * This keeps track of the last created object
		 */
		unsigned int lastId_;

		static const unsigned int numScratch;
		static const unsigned int blockSize;
};

#endif // _ID_MANAGER_H
