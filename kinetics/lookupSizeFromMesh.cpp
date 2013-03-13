/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
/*
#include "../mesh/VoxelJunction.h"
#include "../mesh/MeshEntry.h"
#include "../mesh/Boundary.h"
#include "../mesh/ChemCompt.h"
*/
#include "lookupSizeFromMesh.h"

// Utility function: return the compartment in which the specified
// object is located.
// Simply traverses the tree toward the root till it finds a
// compartment. Pools use a special msg, but this works for reacs too.
Id getCompt( Id id )
{
	const Element* e = id.element();
	if ( e->cinfo()->isA( "PoolBase" ) ) {
		vector< Id > neighbours;
		if ( e->getNeighbours( neighbours, e->cinfo()->findFinfo( "requestSize" ) ) == 1 ) {
			Id pa = Neutral::parent( neighbours[0].eref() ).id;
			if ( pa.element()->cinfo()->isA( "ChemCompt" ) )
				return pa;
		}
	}
	Id pa = Neutral::parent( id.eref() ).id;
	if ( pa == Id() )
		return pa;
	else if ( pa.element()->cinfo()->isA( "ChemCompt" ) )
		return pa;
	return getCompt( pa );
}


/// Utility function to find the size of a pool.
double lookupSizeFromMesh( const Eref& e, const SrcFinfo* sf )
{
	const vector< MsgFuncBinding >* mfb = 
		e.element()->getMsgAndFunc( sf->getBindIndex() );
	if ( !mfb ) return 1.0;
	if ( mfb->size() == 0 ) return 1.0;

	// Can replace with regular get if it is local.
	double size = 
		Field< double >::fastGet( e, (*mfb)[0].mid, (*mfb)[0].fid );

	if ( size <= 0 ) size = 1.0;

	return size;
}

/**
 * Figures out all the volumes of the substrates or products on the
 * specified reaction 'reac'. The SrcFinfo is for the sub or prd msg.
 * Returns the index of the smallest vol. Passes back a vector of volumes.
 * The meshIndex is zero. Reasoning is as follows: both in the case of 
 * well-stirred (single mesh entry) models, and in the case of spatial
 * models with consistent mesh sizes and alignments, the mesh entry 
 * volumes are in the same ratio.
 * Cases with more complex arrangements may also use the current vols as
 * a starting point, but will need to add index-specific scaling factors
 * to their reaction system.
 */

unsigned int getReactantVols( const Eref& reac, const SrcFinfo* pools, 
	vector< double >& vols )
{
	static const unsigned int meshIndex = 0;
	static const Cinfo* poolCinfo = Cinfo::find( "PoolBase" );
	static const Finfo* f1 = poolCinfo->findFinfo( "requestSize" );
	static const SrcFinfo* poolRequestSize = 
		dynamic_cast< const SrcFinfo* >( f1 );

	const vector< MsgFuncBinding >* mfb = 
		reac.element()->getMsgAndFunc( pools->getBindIndex() );
	unsigned int smallIndex = 0;

	vols.resize( 0 );
	if ( mfb ) {
		for ( unsigned int i = 0; i < mfb->size(); ++i ) {
			double v = 1;
			Element* pool = Msg::getMsg( (*mfb)[i].mid )->e2();
			if ( pool == reac.element() )
				pool = Msg::getMsg( (*mfb)[i].mid )->e1();
			assert( pool != reac.element() );
			Eref pooler( pool, meshIndex );
			if ( pool->cinfo()->isA( "PoolBase" ) ) {
				v = lookupSizeFromMesh( pooler, poolRequestSize );
			} else {
				cout << "Error: getReactantVols: pool is of unknown type\n";
				assert( 0 );
			}
			vols.push_back( v );
			if ( v < vols[0] )
				smallIndex = i;
		}
	}
	return smallIndex;
}

/**
 * Returns conversion factor to convert rates from concentration to 
 * mol# units.
 * Handles arbitrary combinations of volumes.
 * Assumes that the reference volume for computing rates is the 
 * smallest volume.
 * 26 Feb 2013: This is now changed to use the volume of the first entry.
 * Should only be used for substrates. For products need to find the
 * first substrate, separately, and use that to scale down the conv factor.
 * Assumes all calculations are in SI: cubic metres and millimolar.
 * 27 Feb 2013: This is changed to use the volume of a voxel of the 
 * the home compartment of the reac.
 */

double convertConcToNumRateUsingMesh( const Eref& e, const SrcFinfo* pools, 
	bool doPartialConversion )
{
	vector< double > vols;
	// unsigned int smallest = getReactantVols( e, pools, vols );
	double conv = 1.0;
	getReactantVols( e, pools, vols );
	for ( unsigned int i = 0; i < vols.size(); ++i ) {
		conv *= vols[i] * NA;
	}
	if ( !doPartialConversion ) {
		Id compt = getCompt( e.id() );
		if ( compt != Id() ) {
			Id mesh( compt.value() + 1 );
			double meshVol = Field< double >::get( mesh, "size" );
			/*
			ChemCompt* cc = 
					reinterpret_cast< ChemCompt* >( compt.eref().data() ):
			meshVol = cc->getMeshEntrySize( 0 );
			*/
			conv /= meshVol * NA;
		}
	}
	/*
	for ( unsigned int i = 0; i < vols.size(); ++i ) {
		if ( doPartialConversion || i != 0 ) {
			conv *= vols[i] * NA;
		}
	}
	*/
	return conv;
}


/**
 * Generates conversion factor for rates from concentration to mol# units.
 * Assumes that all reactant pools (substrate and product) are within the
 * same mesh entry and therefore have the same volume.
 */
/*
double convertConcToNumRateUsingMesh( const Eref& e, const SrcFinfo* pools, 
	unsigned int meshIndex, double scale, bool doPartialConversion )
{
	static const Cinfo* poolCinfo = Cinfo::find( "Pool" );
	static const Cinfo* zombiePoolCinfo = Cinfo::find( "ZombiePool" );

	static const Finfo* f1 = poolCinfo->findFinfo( "requestSize" );
	static const SrcFinfo* poolRequestSize = 
		dynamic_cast< const SrcFinfo* >( f1 );

	static const Finfo* f2 = zombiePoolCinfo->findFinfo( "requestSize" );
	static const SrcFinfo* zombiePoolRequestSize = 
		dynamic_cast< const SrcFinfo* >( f2 );

	const vector< MsgFuncBinding >* mfb = 
		e.element()->getMsgAndFunc( pools->getBindIndex() );
	double conversion = 1.0;
	if ( mfb && mfb->size() > 0 ) {
		if ( doPartialConversion || mfb->size() > 1 ) {
			Element* pool = Msg::getMsg( (*mfb)[0].mid )->e2();
			if ( pool == e.element() )
				pool = Msg::getMsg( (*mfb)[0].mid )->e1();
			assert( pool != e.element() );
			Eref pooler( pool, meshIndex );
			if ( pool->cinfo() == poolCinfo ) {
				conversion = lookupSizeFromMesh( pooler, poolRequestSize );
			} else if ( pool->cinfo()->isA( "ZombiePool" ) ) {
				conversion = lookupSizeFromMesh( pooler, zombiePoolRequestSize );
			}
			conversion *= scale * NA;
			double power = doPartialConversion + mfb->size() - 1;
			if ( power > 1.0 ) {
				conversion = pow( conversion, power );
			}
		}
		if ( conversion <= 0 ) 
			conversion = 1.0;
	}

	return conversion;
}
*/

/**
 * Generates conversion factor for rates from concentration to mol# units.
 * This variant already knows the volume, but has to figure out # of
 * reactants.
 */
double convertConcToNumRateUsingVol( const Eref& e, const SrcFinfo* pools, 
	double volume, double scale, bool doPartialConversion )
{
	const vector< MsgFuncBinding >* mfb = 
		e.element()->getMsgAndFunc( pools->getBindIndex() );
	double conversion = 1.0;
	if ( mfb && mfb->size() > 0 ) {
		if ( doPartialConversion || mfb->size() > 1 ) {
			conversion = scale * NA * volume;
			double power = doPartialConversion + mfb->size() - 1;
			if ( power > 1.0 ) {
				conversion = pow( conversion, power );
			}
		}
		if ( conversion <= 0 ) 
			conversion = 1.0;
	}

	return conversion;
}

/**
 * Generates conversion factor for rates from concentration to mol# units.
 * This variant is used when the reactants are in different compartments
 * or mesh entries, and may therefore have different volumes.
 * We already know the reactants and their affiliations.
 */
double convertConcToNumRateInTwoCompts( double v1, unsigned int n1, 
	double v2, unsigned int n2, double scale )
{
	double conversion = 1.0;

	for ( unsigned int i = 1; i < n1; ++i )
		conversion *= scale * NA * v1;
	for ( unsigned int i = 0; i < n2; ++i )
		conversion *= scale * NA * v2;

	if ( conversion <= 0 ) 
			conversion = 1.0;

	return conversion;
}
