/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "CylBase.h"
#include "NeuroNode.h"
#include "Stencil.h"
#include "NeuroStencil.h"

NeuroStencil::NeuroStencil()
{;}

NeuroStencil::~NeuroStencil()
{;}

void NeuroStencil::setNodes( const vector< NeuroNode >* nodes )
{
	nodes_ = nodes;
}

void NeuroStencil::setNodeIndex( const vector< unsigned int >* nodeIndex )
{
	nodeIndex_ = nodeIndex;
}

		/**
		 * computes the Flux f in the voxel on meshIndex. Takes the
		 * matrix of molNumber[meshIndex][pool] and 
		 * the vector of diffusionConst[pool] as arguments.
		 */
void NeuroStencil::addFlux( unsigned int meshIndex, 
			vector< double >& f, const vector< vector< double > >& S, 
			const vector< double >& diffConst ) const
{

	;
}

