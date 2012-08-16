/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _NEURO_NODE_H
#define _NEURO_NODE_H

/**
 * Helper class for the NeuroMesh. Defines the geometry of the branching
 * neuron.
 */

class NeuroNode: public CylBase
{
	public:
		NeuroNode( const CylBase& cb, 
			unsigned int parent, const vector< unsigned int >& children,
			unsigned int startFid_, Id elecCompt,
			bool isDummyNode, bool isSphere, bool isStartNode );
		NeuroNode();

		unsigned int parent() const;
		unsigned int startFid() const;
		Id elecCompt() const;
		bool isDummyNode() const;
		bool isSphere() const;
		bool isStartNode() const;
		const vector< unsigned int >& children() const;

	private:
		/**
		 * Index of parent NeuroNode, typically a diffusive compartment. 
		 * In the special case where the junction to the parent electrical 
		 * compartment is NOT at the end of the parent compartment, this
		 * refers instead to a dummy NeuroNode which has the coordinates.
		 *
		 * One of the nodes, typically the soma, will have no parent.
		 */
		unsigned int parent_; 

		/**
		 * Index of children of this NeuroNode.
		 */
		vector< unsigned int >children_;

		/**
		 * Index of starting MeshEntry handled by this NeuroNode. Assumes
		 * a block of contiguous fids are handled by each NeuroNode.
		 */
		unsigned int startFid_;


		/// Id of electrical compartment in which this diffusive compt lives
		Id elecCompt_; 

		/**
		 * True when this is a dummy node to represent the coordinates
		 * of the start end of a compartment. For example, the start coords
		 * of a compartment 
		 * sitting on a spherical soma, or the start coords of a spine neck 
		 * along a longer dendritic compartment.
		 * In all other cases the start coordinates are just those of the
		 * end of the parent compartment.
		 *
		 * When the isDummyNode is true, the elecCompt represents the Id of
		 * the compartment whose start it is.
		 */
		bool isDummyNode_; 

		/**
		 * Special case for soma, perhaps for spine heads.
		 * When true, xyz are centre, and dia is dia.
		 */
		bool isSphere_;

		/**
		 * Special case for starting node of tree, typically soma. 
		 * There should
		 * be just one StartNode. Its parent is undefined.
		 */
		bool isStartNode_;

	// For spines we will also need a double to indicate position along
	// parent dendrite. 
	//
};

#endif	// _NEURO_NODE_H
