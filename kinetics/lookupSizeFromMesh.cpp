/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "lookupSizeFromMesh.h"

const double CONC_UNIT_CONV = 1.0e-3; // convert micromolar to SI units.

/// Utility function to find the size of a pool.
double lookupSizeFromMesh( const Eref& e, const SrcFinfo* sf )
{
	const vector< MsgFuncBinding >* mfb = 
		e.element()->getMsgAndFunc( sf->getBindIndex() );
	if ( !mfb ) return 1.0;
	if ( mfb->size() == 0 ) return 1.0;

	double size = 
		Field< double >::fastGet( e, (*mfb)[0].mid, (*mfb)[0].fid );

	if ( size <= 0 ) size = 1.0;

	return size;
}

/**
 * Generates conversion factor for rates from concentration to mol# units.
 * Assumes that all reactant pools (substrate and product) are within the
 * same mesh entry and therefore have the same volume.
 */

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
