/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "CylBase.h"
#include "NeuroNode.h"

/**
 * Helper class for the NeuroMesh. Defines the geometry of the branching
 * neuron.
 */

NeuroNode::NeuroNode( const CylBase& cb, 
		unsigned int parent, const vector< unsigned int >& children,
		unsigned int startFid, Id elecCompt,
		bool isDummyNode, bool isSphere, bool isStartNode
   	)
		:
				CylBase( cb ), 
				parent_( parent ),
				children_( children ),
				startFid_( startFid ),
				elecCompt_( elecCompt ),
				isDummyNode_( isDummyNode ),
				isSphere_( isSphere ),
				isStartNode_( isStartNode )
{;}

NeuroNode::NeuroNode()
		:
				parent_( 0 ),
				startFid_( 0 ),
				elecCompt_( Id() ),
				isDummyNode_( false ),
				isSphere_( false ),
				isStartNode_( false )
{;}


unsigned int NeuroNode::parent() const
{
		return parent_;
}

unsigned int NeuroNode::startFid() const
{
		return startFid_;
}

Id NeuroNode::elecCompt() const
{
		return elecCompt_;
}
bool NeuroNode::isDummyNode() const
{
		return isDummyNode_;
}
bool NeuroNode::isSphere() const
{
		return isSphere_;
}
bool NeuroNode::isStartNode() const
{
		return isStartNode_;
}

const vector< unsigned int >& NeuroNode::children() const
{
		return children_;
}
