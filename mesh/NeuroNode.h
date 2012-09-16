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
		/**
		 * This function explicitly fills in all fields of the NeuroNode
		 */
		NeuroNode( const CylBase& cb, 
			unsigned int parent, const vector< unsigned int >& children,
			unsigned int startFid_, Id elecCompt,
			bool isSphere );
		/**
		 * This builds the node using info from the compartment. But the
		 * parent and children have to be filled in later
		 */
		NeuroNode( Id elecCompt );
		/**
		 * Empty constructor for vectors
		 */
		NeuroNode();


		unsigned int parent() const;
		unsigned int startFid() const;
		Id elecCompt() const;

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
		bool isDummyNode() const; // True if CylBase::numDivs is zero.
		bool isSphere() const;
		bool isStartNode() const; // True if startFid_ == 0
		const vector< unsigned int >& children() const;


		/**
		 * Fills in child vector
		 */
		void addChild( unsigned int child );

		/**
		 * Assigns parent node info
		 */
		void setParent( unsigned int parent );

		/**
		 * Assignes startFid
		 */
		void setStartFid( unsigned int f );

		/**
		 * Calculates and returns compartment length, from parent xyz to
		 * self xyz. Assigns own length as a side-effect.
		 */
		double calculateLength( const CylBase& parent );

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
		 * Special case for soma, perhaps for spine heads.
		 * When true, xyz are centre, and dia is dia.
		 */
		bool isSphere_;

	// For spines we will also need a double to indicate position along
	// parent dendrite. 
	//
};

#endif	// _NEURO_NODE_H
