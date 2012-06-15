/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "../shell/Wildcard.h"

//////////////////////////////////////////////////////////////////
//
// Here we have a set of utility functions to estimate a suitable
// dt for the simulation. Needed for fixed-timestep methods, but
// also useful to set a starting timestep for variable timestep methods.
//
//////////////////////////////////////////////////////////////////

/**
 * Returns largest propensity for a reaction Element e. Direction of
 * reaction calculation is set by the isPrd flag.
 */
double findReacPropensity( Id id, bool isPrd )
{
	static const Cinfo* rCinfo = Cinfo::find( "Reac" );
	static const Finfo* substrateFinfo = rCinfo->findFinfo( "sub" );
	static const Finfo* productFinfo = rCinfo->findFinfo( "prd" );

	assert( id.element()->cinfo()->isA( "Reac" ) );

	double prop;
	if ( isPrd )
		prop = Field< double >::get( id, "kb" );
	else 
		prop = Field< double >::get( id, "kf" );
	double min = 1.0e10;
	double mval;

	vector< Id > reactants;
	if ( isPrd )
		id.element()->getNeighbours( reactants, productFinfo );
	else
		id.element()->getNeighbours( reactants, substrateFinfo );

	for ( vector< Id >::iterator i = reactants.begin(); 
		i != reactants.end(); ++i ) {
		assert( i->element()->cinfo()->isA( "Pool" ) );
		mval = Field< double >::get( *i, "n" );
		prop *= mval;
		if ( min > mval )
			min = mval;
	}

	if ( min > 0.0 )
		return prop / min;
	else
		return 0.0;
}

/**
 * Returns largest propensity for a reaction Element e. Direction of
 * reaction calculation is set by the isPrd flag.
 */
double findEnzSubPropensity( Id id )
{
	static const Cinfo* eCinfo = Cinfo::find( "Enz" );
	static const Finfo* substrateFinfo = eCinfo->findFinfo( "sub" );
	static const Finfo* enzymeFinfo = eCinfo->findFinfo( "enz" );
	assert( id.element()->cinfo()->isA("Enz" ) );
	double prop;

	if ( id.element()->cinfo()->isA( "MMenz" ) ) { 
		// An MM enzyme, implicit form.
		// Here we compute it as rate / nmin.
		double Km;
		double kcat;
		Km = Field< double >::get( id, "Km" );
		kcat = Field< double >::get( id, "kcat" );
		assert( Km > 0 );
		prop = kcat / Km;
	} else {
		prop = Field< double >::get( id, "k1" ); // k1 is in num units
	}
	double min = 1.0e10;
	double mval;
	vector< Id > substrates;
	vector< Id > enzymes;
	id.element()->getNeighbours( substrates, substrateFinfo );
	id.element()->getNeighbours( enzymes, enzymeFinfo );
	if ( substrates.size() == 0 || enzymes.size() != 1 ) {
		cout << "Warning: findEnzSumPropensity: no substrates or "
			"enzymes != 1 (" << enzymes.size() << ")\n";
		return 0.0;
	}

	assert( enzymes[0].element()->cinfo()->isA( "Pool" ) );
	mval = Field< double >::get( enzymes[0], "n" );
	prop *= mval;
	min = mval;
	for ( vector< Id >::iterator i = substrates.begin(); 
		i != substrates.end(); ++i ) {
		assert( i->element()->cinfo()->isA( "Pool" ) );
		mval = Field< double >::get( *i, "n" );
		prop *= mval;
		if ( min > mval ) 
			min = mval;
	}
	
	if ( min > 0.0 )
		return prop / min;
	else
		return 0.0;
}

double findEnzPrdPropensity( Id id )
{
	if ( !id.element()->cinfo()->isA( "Enz" ) )
		return 0.0;

	double k2;
	double k3;
	k2 = Field< double >::get( id, "k2" );
	k3 = Field< double >::get( id, "k3" );
	return k2 + k3;
}

/**
 * This function figures out an appropriate dt for a fixed timestep 
 * method. It does so by estimating the permissible ( default 1%) error 
 * assuming a forward Euler advance.
 * Also returns the element and field that have the highest propensity,
 * that is, the shortest dt.
 */
// #ifndef NDEBUG
// #include <limits>
	// Needed for the isInfinity call?
// #endif



double estimateDt( Id parent, 
	Id& elm, string& field, double error, string method ) 
{
	assert( error > 0.0 );
	vector< Id > elist;
	vector< Id >::iterator i;

	// int allChildren( start, insideBrace, index, ret)
	if ( allChildren( parent, "", elist ) == 0 ) {
		elm = Id();
		field = "";
		return 1.0;
	}

	double prop = 0.0;
	double maxProp = 0.0;
	double sumProp = 0.0;
	double recommendedDt = 1.0;
	unsigned int numProp = 0;
	unsigned int memEstimate = 0;

	for ( i = elist.begin(); i != elist.end(); i++ )
	{
		const Cinfo* ci = i->element()->cinfo();
		memEstimate += ci->dinfo()->size();
		if ( ci->isA( "Reac" ) ) {
			prop = findReacPropensity( *i, 0 );
			if ( prop > 0 ) {
				sumProp += prop;
				++numProp;
			}
			if ( maxProp < prop ) {
				maxProp = prop;
				elm = *i;
				field = "kf";
			}
			prop = findReacPropensity( *i, 1 );
			if ( prop > 0 ) {
				sumProp += prop;
				++numProp;
			}
			if ( maxProp < prop ) {
				maxProp = prop;
				elm = *i;
				field = "kb";
			}
		} else if ( i->element()->cinfo()->isA( "Enz" ) ) {
			prop = findEnzSubPropensity( *i );
			if ( prop > 0 ) {
				sumProp += prop;
				++numProp;
			}
			if ( maxProp < prop ) {
				maxProp = prop;
				elm = *i;
				field = "k1";
			}
			prop = findEnzPrdPropensity( *i );
			if ( prop > 0 ) {
				sumProp += prop;
				++numProp;
			}
			if ( maxProp < prop ) {
				maxProp = prop;
				elm = *i;
				field = "k3";
			}
		}
	}
	assert ( error < 1.0 );
	if ( maxProp <= 0 ) 
		maxProp = 10.0;

	recommendedDt = sqrt( error ) / maxProp;
	// isinf is in 'utility.h'
	// assert ( !isInfinity< double >( recommendedDt ) );

	return recommendedDt;
}
