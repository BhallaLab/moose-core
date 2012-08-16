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

NeuroStencil::NeuroStencil( const vector< NeuroNode >& nodes, 
					const vector< unsigned int >& nodeIndex,
					const vector< double >& vs,
					const vector< double >& area)
		:
				nodes_( nodes ),
				nodeIndex_( nodeIndex ),
				vs_( vs ),
				area_( area )
{;}

static vector< NeuroNode > nn;
static vector< unsigned int > ni;
static vector< double > dv;

NeuroStencil::NeuroStencil()
		:
				nodes_( nn ),
				nodeIndex_( ni ),
				vs_( dv ),
				area_( dv )
{;}

NeuroStencil::~NeuroStencil()
{;}

		/**
		 * computes the Flux f in the voxel on meshIndex. Takes the
		 * matrix of molNumber[meshIndex][pool] and 
		 * the vector of diffusionConst[pool] as arguments.
		 */
/**
 * This is a little nasty and isn't yet done for the CylMesh, though
 * perhaps it should be called a ConeFrustrumMesh.
 * Since we have different areas at either end of the voxel, but each
 * voxel has the same length, we get the term for the flux as:
 *
 * flux = ( Cminus*Aminus + Cplus*Aplus - C0*(Aminus + Aplus) ) / lensq
 *
 * where Cminus is the concentration in the compartment to the left,
 * Cplus is conc in compt to right
 * C0 is the conc in current compartment
 * Aminus and Aplus are the areas of the left and right faces of current
 * compartment.
 * To add to the fun, S is in # units, not conc. So in terms of 
 * tminus, t0 and tplus, the # unit versions, we have:
 *
 * numFlux = VS0 * ( tminus*Aminus/VSminus + tplus*Aplus/VSplus - t0*(Aminus+Aplus)/VS0 ) / lensq
 *
 * where VS is VolScale, to go from conc units to # units, taking local
 * volume into account.
 *
 */
void NeuroStencil::addFlux( unsigned int index, 
			vector< double >& f, const vector< vector< double > >& S, 
			const vector< double >& diffConst ) const
{
	assert( index < nodeIndex_.size() );
	assert( nodes_.size() > nodeIndex_[index] );
	const vector< double >& t0 = S[ index ];
	const NeuroNode& node = nodes_[ nodeIndex_[ index ] ];
	const NeuroNode& parent = nodes_[ nodeIndex_[ node.parent() ] ];
	double vs0 = vs_[index];
	double len = node.getLength() / node.getNumDivs();
	double invSq = 1.0 / ( len * len );
	unsigned int minusIndex = index - 1;

	if ( index - node.startFid() < node.getNumDivs() -1 ) { //Not last voxel
		if ( index == node.startFid() ) { // Is first voxel
			if ( node.isStartNode() ) { // root of the tree.
				addHalfFlux( index, f, t0, S[ index+1 ],
					area_[ index+1 ], vs0, vs_[ index+1 ],
					invSq, diffConst );
				return;
			} else { // Not root, can make linear diffusion with parent.
				minusIndex = parent.startFid() + parent.getNumDivs() - 1;
			}
		}
		addLinearFlux( index, f, S[ minusIndex ], t0, S[ index+1 ],
			area_[ index ], area_[ index+1 ], 
			vs_[ minusIndex ], vs0, vs_[ index+1],
			invSq, diffConst );
		return;
	} else { // Last voxel
		if ( node.children().size() == 1 ) { // OK, another linear diffusion
			unsigned int plusIndex = 
				nodes_[ node.children()[0] ].startFid();
			if ( index == node.startFid() ) { // First voxel
				if ( node.isStartNode() ) { // root of the tree
					// do stuff here for half diffusion.
					addHalfFlux( index, f, t0, S[ plusIndex ],
						area_[ plusIndex ], vs0, vs_[ plusIndex ],
						invSq, diffConst );
					return;
				} else {
					minusIndex = parent.startFid() + parent.getNumDivs() -1;
				}
			}
			addLinearFlux( index, f, S[ minusIndex ], t0, S[ plusIndex ],
				area_[ index ], area_[ plusIndex ], 
				vs_[ minusIndex ], vs0, vs_[ plusIndex ],
				invSq, diffConst );
			return;
		} else { // Do halfFluxes for all combinations.
			for ( unsigned int i = 0; i < node.children().size(); ++i ) {
				unsigned int plusIndex = 
					nodes_[ node.children()[i] ].startFid();
				addHalfFlux( index, f, t0, S[ plusIndex ],
					area_[ plusIndex ], vs0, vs_[ plusIndex ],
					invSq, diffConst );
			}
			if ( index == node.startFid() ) {
				if ( node.isStartNode() ) 
					return; // No diffusion at all here.
				minusIndex = parent.startFid() + parent.getNumDivs() - 1;
			}
			addHalfFlux( index, f, t0, S[ minusIndex ],
				area_[ minusIndex ], vs0, vs_[ minusIndex ],
				invSq, diffConst );
		}
	}
}

void NeuroStencil::addLinearFlux( unsigned int index, 
			vector< double >& f, 
			const vector< double >& tminus,
			const vector< double >& t0,
			const vector< double >& tplus,
			double aminus,
			double aplus,
			double vsminus,
			double vs0,
			double vsplus,
			double invSq,
			const vector< double >& diffConst ) const
{
		for ( unsigned int i = 0; i < f.size(); ++i )
			f[i] += vs0 * diffConst[i] * invSq * 
				( tminus[i] * aminus / vsminus + 
				  tplus[i] * aplus / vsplus - 
				  t0[i] * (aminus + aplus )/vs0 );
}

void NeuroStencil::addHalfFlux( unsigned int index, 
			vector< double >& f, 
			const vector< double >& t0,
			const vector< double >& tprime,
			double area,
			double vs0,
			double vsprime,
			double invSq,
			const vector< double >& diffConst ) const
{
		for ( unsigned int i = 0; i < f.size(); ++i )
			f[i] += vs0 * diffConst[i] * invSq * area *
				( tprime[i] / vsprime - t0[i] / vs0 );
}
