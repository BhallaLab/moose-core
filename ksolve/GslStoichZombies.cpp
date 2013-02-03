/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include "SolverBase.h"
#include "VoxelPools.h"
#include "OdeSystem.h"
#include "../shell/Shell.h"
#include "../mesh/MeshEntry.h"
#include "../mesh/Boundary.h"
#include "../mesh/ChemMesh.h"
#include "GslStoich.h"

///////////////////////////////////////////////////
// Field access functions
///////////////////////////////////////////////////

void GslStoich::setN( const Eref& e, double v )
{
	unsigned int i = e.index().value(); // Later: Handle node location.
	unsigned int j = coreStoich()->convertIdToPoolIndex( e.id() );
	assert( i < pools_.size() );
	assert( j < pools_[i].size() );
	pools_[i].varS()[j] = v;
	assert( i < y_.size() );
	assert( j < y_[i].size() );
	y_[i][j] = v;
}

double GslStoich::getN( const Eref& e ) const
{
	unsigned int i = e.index().value();
	unsigned int j = coreStoich()->convertIdToPoolIndex( e.id() );
	assert( i < pools_.size() );
	assert( j < pools_[i].size() );
	return pools_[i].S()[j];
}

void GslStoich::setNinit( const Eref& e, double v )
{
	unsigned int i = e.index().value();
	unsigned int j = coreStoich()->convertIdToPoolIndex( e.id() );
	assert( i < pools_.size() );
	assert( j < pools_[i].size() );
	pools_[i].varSinit()[j] = v;
}

double GslStoich::getNinit( const Eref& e ) const
{
	unsigned int i = e.index().value();
	unsigned int j = coreStoich()->convertIdToPoolIndex( e.id() );
	assert( i < pools_.size() );
	assert( j < pools_[i].size() );
	return pools_[i].Sinit()[j];
}

void GslStoich::setSpecies( const Eref& e, unsigned int v )
{
	unsigned int j = coreStoich()->convertIdToPoolIndex( e.id() );
	assert( j < pools_[ e.index().value() ].size() );
	coreStoich_.setSpecies( j, v );
}

unsigned int GslStoich::getSpecies( const Eref& e )
{
	unsigned int j = coreStoich()->convertIdToPoolIndex( e.id() );
	assert( j < pools_[ e.index().value() ].size() );
	return coreStoich()->getSpecies( j );
}

void GslStoich::setDiffConst( const Eref& e, double v )
{
	unsigned int j = coreStoich()->convertIdToPoolIndex( e.id() );
	assert( j < pools_[ e.index().value() ].size() );
	coreStoich_.setDiffConst( j, v );
}

double GslStoich::getDiffConst( const Eref& e ) const
{
	unsigned int j = coreStoich()->convertIdToPoolIndex( e.id() );
	assert( j < pools_[ e.index().value() ].size() );
	return coreStoich()->getDiffConst( j );
}
/////////////////////////////////////////////////////////////////////////
// Reac stuff.
/////////////////////////////////////////////////////////////////////////
void GslStoich::setReacKf( const Eref& e, double v ) const
{
	coreStoich()->setReacKf( e, v );
}
void GslStoich::setReacKb( const Eref& e, double v ) const
{
	coreStoich()->setReacKb( e, v );
}

double GslStoich::getReacNumKf( const Eref& e ) const
{
	return coreStoich()->getR1( e );
}
double GslStoich::getReacNumKb( const Eref& e ) const
{
	if ( coreStoich()->getOneWay() )
		return coreStoich()->getR1offset1( e );
	else
		return coreStoich()->getR2( e );
}
/////////////////////////////////////////////////////////////////////////
// Michaels-Menten Enz stuff.
/////////////////////////////////////////////////////////////////////////
// Assignment of Km is in conc units.
void GslStoich::setMMenzKm( const Eref& e, double v ) const
{
	coreStoich()->setMMenzKm( e, v );
}

void GslStoich::setMMenzKcat( const Eref& e, double v ) const
{
	coreStoich()->setMMenzKcat( e, v );
}

double GslStoich::getMMenzNumKm( const Eref& e ) const
{
	return coreStoich()->getR1( e );
}

double GslStoich::getMMenzKcat( const Eref& e ) const
{
	return coreStoich()->getR2( e );
}

/////////////////////////////////////////////////////////////////////////
// Regular (explicit enz complex form) Enz stuff.
/////////////////////////////////////////////////////////////////////////

/**
 * Sets the rate v (given in millimolar concentration units)
 * for the forward enzyme reaction of binding substrate to enzyme.
 */
void GslStoich::setEnzK1( const Eref& e, double v ) const
{
	coreStoich()->setEnzK1( e, v );
}


/// Set rate k2 (1/sec) for enzyme
void GslStoich::setEnzK2( const Eref& e, double v ) const
{
	coreStoich()->setEnzK2( e, v );
}

/// Set rate k3 (1/sec) for enzyme
void GslStoich::setEnzK3( const Eref& e, double v ) const
{
	coreStoich()->setEnzK3( e, v );
}

// Returns K1 in # units
double GslStoich::getEnzNumK1( const Eref& e ) const
{
	return coreStoich()->getR1( e );
}
/// get rate k2 (1/sec) for enzyme
double GslStoich::getEnzK2( const Eref& e ) const
{
	if ( coreStoich()->getOneWay() )
		return coreStoich()->getR1offset1( e );
	else
		return coreStoich()->getR2( e );
}

/// get rate k3 (1/sec) for enzyme
double GslStoich::getEnzK3( const Eref& e ) const
{
	if ( coreStoich()->getOneWay() )
		return coreStoich()->getR1offset2( e );
	else
		return coreStoich()->getR1offset1( e );
}

//////////////////////////////////////////////////////////////////////

ZeroOrder* makeHalfReaction(
	double rate, const StoichCore* sc, const vector< Id >& reactants )
{
	ZeroOrder* rateTerm = 0;
	if ( reactants.size() == 1 ) {
		rateTerm =  new FirstOrder( 
			rate, sc->convertIdToPoolIndex( reactants[0] ) );
	} else if ( reactants.size() == 2 ) {
		rateTerm = new SecondOrder( rate,
			sc->convertIdToPoolIndex( reactants[0] ),
			sc->convertIdToPoolIndex( reactants[1] ) );
	} else if ( reactants.size() > 2 ) {
		vector< unsigned int > temp;
		for ( unsigned int i = 0; i < reactants.size(); ++i )
			temp.push_back( sc->convertIdToPoolIndex( reactants[i] ) );
		rateTerm = new NOrder( rate, temp );
	} else {
		cout << "Error: GslStoichZombies::makeHalfReaction: no reactants\n";
	}
	return rateTerm;
}

/**
 * This takes the specified forward and reverse half-reacs belonging
 * to the specified Reac, and builds them into the Stoich.
 */
void GslStoich::installReaction( Id reacId,
		const vector< Id >& subs, 
		const vector< Id >& prds )
{
	ZeroOrder* forward = makeHalfReaction( 0, coreStoich(), subs );
	ZeroOrder* reverse = makeHalfReaction( 0, coreStoich(), prds );
	coreStoich_.installReaction( forward, reverse, reacId );
}

/**
 * This takes the forward, backward and product formation half-reacs
 * belonging to the specified Enzyme, and builds them into the
 * Stoich
 */
void GslStoich::installEnzyme( Id enzId, Id enzMolId, Id cplxId,
	const vector< Id >& subs, const vector< Id >& prds )
{
	vector< Id > temp( subs );
	temp.insert( temp.begin(), enzMolId );
	ZeroOrder* r1 = makeHalfReaction( 0, coreStoich(), temp );
	temp.clear();
	temp.resize( 1, cplxId );
	ZeroOrder* r2 = makeHalfReaction( 0, coreStoich(), temp );
	ZeroOrder* r3 = makeHalfReaction( 0, coreStoich(), temp );

	coreStoich_.installEnzyme( r1, r2, r3, enzId, enzMolId, prds );
}

/**
 * This takes the baseclass for an MMEnzyme and builds the
 * MMenz into the Stoich.
 */
void GslStoich::installMMenz( Id enzId, Id enzMolId,
	const vector< Id >& subs, const vector< Id >& prds )
{
	MMEnzymeBase* meb;
	unsigned int enzIndex = coreStoich()->convertIdToPoolIndex( enzMolId );
	unsigned int enzSiteIndex = coreStoich()->convertIdToReacIndex( enzId );
	if ( subs.size() == 1 ) {
		unsigned int subIndex = coreStoich()->convertIdToPoolIndex( subs[0] );
		meb = new MMEnzyme1( 1, 1, enzIndex, subIndex );
	} else if ( subs.size() > 1 ) {
		vector< unsigned int > v;
		for ( unsigned int i = 0; i < subs.size(); ++i )
			v.push_back( coreStoich()->convertIdToPoolIndex( subs[i] ) );
		ZeroOrder* rateTerm = new NOrder( 1.0, v );
		meb = new MMEnzyme( 1, 1, enzIndex, rateTerm );
	} else {
		cout << "Error: GslStoich::installEnzyme: No substrates for "  <<
			enzId.path() << endl;
		return;
	}
	coreStoich_.installMMenz( meb, enzSiteIndex, subs, prds );
}


unsigned int GslStoich::convertIdToPoolIndex( Id id ) const
{
	return coreStoich()->convertIdToPoolIndex( id );
}
