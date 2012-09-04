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
		unsigned int startFid, Id elecCompt, bool isSphere
   	)
		:
				CylBase( cb ), 
				parent_( parent ),
				children_( children ),
				startFid_( startFid ),
				elecCompt_( elecCompt ),
				isSphere_( isSphere )
{;}

NeuroNode::NeuroNode()
		:
				parent_( 0 ),
				startFid_( 0 ),
				elecCompt_( Id() ),
				isSphere_( false )
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
		return ( getNumDivs() == 0 );
}
bool NeuroNode::isSphere() const
{
		return isSphere_;
}
bool NeuroNode::isStartNode() const
{
		return ( startFid_ == 0 );
}

const vector< unsigned int >& NeuroNode::children() const
{
		return children_;
}
void NeuroNode::addChild( unsigned int child )
{
	children_.push_back( child );
}

double NeuroNode::calculateLength( const CylBase& parent )
{
	if ( &parent == this ) // Do nothing
			return getLength();
	double dx = parent.getX() - getX();
	double dy = parent.getY() - getY();
	double dz = parent.getZ() - getZ();
	double ret = sqrt( dx * dx + dy * dy + dz * dz );
	setLength( ret );
	return ret;
}
