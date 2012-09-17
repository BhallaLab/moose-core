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
 * matrix S of molNumber[meshIndex][pool] and 
 * the vector of diffusionConst[pool] as arguments.
 * The vectors Flux and DiffConst are indexed by Pool number.
 *
 * This is a little nasty and isn't yet done for the CylMesh, though
 * perhaps it should be called a ConeFrustrumMesh.
 * Since we have different areas at either end of the voxel, but each
 * voxel has the same length, we get the term for the flux as:
 *
 * flux = D*( Cminus*Aminus + Cplus*Aplus - C0*(Aminus + Aplus) ) / lensq
 * <Unit error.>
 *
 * where Cminus is the concentration in the compartment to the left,
 * Cplus is conc in compt to right
 * C0 is the conc in current compartment
 * Aminus and Aplus are the areas of the left and right faces of current
 * compartment.
 * To add to the fun, S is in # units, not conc. So in terms of 
 * tminus, t0 and tplus, the # unit versions, we have:
 *
 * numFlux = D*VS0 * ( tminus*Aminus/VSminus + tplus*Aplus/VSplus - t0*(Aminus+Aplus)/VS0 ) / lensq
 * <Unit error.>
 *
 * where VS is VolScale, to go from conc units to # units, taking local
 * volume into account.
 *
 * I can only use a 3-point stencil if I have a uniform subdivision of a
 * long compartment, otherwise I have to do individual fluxes on either 
 * side.
 *
 * numFlux|left = D * Aminus * (tminus/VSminus - t0/VS0)*2/(lenMinus + len0)
 * numFlux|right = D * Aplus * (tplus/VSplus - t0/VS0)*2/(lenPlus + len0)
 *
 * To confirm, if we have a uniform length subdivision we can merge these:
 * 2/(lenMinus+len0) = 2/lenPlus+len0) = 1/len
 * numFlux = numFluxLeft+numFluxright =
 * 	D * (tminus*Aminus/VSminus + tplus*Aplus/VSplus - 
 * 		t0*(Aminus+Aplus)/VS0) / len
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
	const NeuroNode* pa = &nodes_[ node.parent() ];
	if ( pa->isDummyNode() )
			pa = &nodes_[ pa->parent() ];
	assert( !pa->isDummyNode() );
	const NeuroNode& parent = *pa;
	double vs0 = vs_[index];
	double len = node.getLength() / node.getNumDivs();
	unsigned int minusIndex = index - 1;

	if ( index - node.startFid() < node.getNumDivs() -1 ) { //Not last voxel
		if ( index == node.startFid() ) { // Is first voxel
			addHalfFlux( index, f, t0, S[ index+1 ],
				area_[ index+1 ], vs0, vs_[ index+1 ],
				len, diffConst );
			if ( !node.isStartNode() ) { // Not root, diff with parent node.
				minusIndex = parent.startFid() + parent.getNumDivs() - 1;
				double paLen = parent.getLength() / parent.getNumDivs();
				addHalfFlux( index, f, t0, S[ minusIndex ],
					area_[ minusIndex ], vs0, vs_[ minusIndex],
					(len + paLen)/2.0, diffConst );
			}
		} else { // this is the only time we have a nice linear case, but
				// it is likely very frequent especially for long compts.
			addLinearFlux( index, f, S[ minusIndex ], t0, S[ index+1 ],
				area_[ index ], area_[ index+1 ], 
				vs_[ minusIndex ], vs0, vs_[ index+1],
				len, diffConst );
		}
	} else { // Last voxel. Diffuse into all children. Check for dummies
		for ( unsigned int i = 0; i < node.children().size(); ++i ) {
			const NeuroNode* pchild = &nodes_[ node.children()[i] ];
			if ( pchild->isDummyNode() ) {
				assert( pchild->children().size() == 1 );
				pchild = &nodes_[ pchild->children()[0] ];
				assert( !pchild->isDummyNode() );
			}
			const NeuroNode& child = *pchild;
			unsigned int plusIndex = child.startFid();
			double childLen = child.getLength() / child.getNumDivs();
			addHalfFlux( index, f, t0, S[ plusIndex ],
				area_[ plusIndex ], vs0, vs_[ plusIndex ],
				( len + childLen) / 2.0, diffConst );
		}
		if ( index == node.startFid() ) { // Is first voxel
			if ( node.isStartNode() ) {
				return; // No diffusion at all here.
			} else { // Diffuse into parent.
				minusIndex = parent.startFid() + parent.getNumDivs() - 1;
				double paLen = parent.getLength() / parent.getNumDivs();
				addHalfFlux( index, f, t0, S[ minusIndex ],
					area_[ minusIndex ], vs0, vs_[ minusIndex ],
					( len + paLen ) / 2.0 , diffConst );
			}
		} else { // Not first voxel. Connect to previous.
			addHalfFlux( index, f, t0, S[ minusIndex ],
				area_[ minusIndex ], vs0, vs_[ minusIndex ],
				len, diffConst );
		}
	}
}

/**
 * This only works if we have uniform subdivisions of length len.
 * The XA can vary, but again has to do so linearly
 */
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
			double len,
			const vector< double >& diffConst ) const
{
		for ( unsigned int i = 0; i < f.size(); ++i )
			f[i] += diffConst[i] * NA *
				( tminus[i] * aminus / vsminus + 
				  tplus[i] * aplus / vsplus - 
				  t0[i] * (aminus + aplus ) / vs0 ) / len;
}

void NeuroStencil::addHalfFlux( unsigned int index, 
			vector< double >& f, 
			const vector< double >& t0,
			const vector< double >& tprime,
			double area,
			double vs0,
			double vsprime,
			double aveLen,
			const vector< double >& diffConst ) const
{
		for ( unsigned int i = 0; i < f.size(); ++i )
			f[i] += diffConst[i] * NA * area *
				( tprime[i] / vsprime - t0[i] / vs0 ) / aveLen;
}
