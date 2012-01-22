/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "StoichHeaders.h"
#include "ElementValueFinfo.h"
#include "Pool.h"
#include "BufPool.h"
#include "FuncPool.h"
#include "Reac.h"
#include "Enz.h"
#include "MMenz.h"
#include "SumFunc.h"
#include "MathFunc.h"
#include "Boundary.h"
#include "MeshEntry.h"
#include "ChemMesh.h"
#include "ZombiePool.h"
#include "ZombieBufPool.h"
#include "ZombieFuncPool.h"
#include "ZombieReac.h"
#include "ZombieEnz.h"
#include "ZombieMMenz.h"
#include "ZombieSumFunc.h"

#include "../shell/Shell.h"

#include "GssaStoich.h"
#include "../randnum/randnum.h"

/**
 * The SAFETY_FACTOR Protects against the total propensity exceeding
 * the cumulative
 * sum of propensities, atot. We do a lot of adding and subtracting of
 * dependency terms from atot. Roundoff error will eventually cause
 * this to drift from the true sum. To guarantee that we never lose
 * the propensity of the last reaction, this safety factor scales the
 * first calculation of atot to be slightly larger. Periodically this
 * will cause the reaction picking step to exceed the last reaction 
 * index. This is safe, we just pick another random number.  
 * This will happen rather infrequently.
 * That is also a good time to update the cumulative sum.
 * A double should have >15 digits, so cumulative error will be much
 * smaller than this.
 */
const double SAFETY_FACTOR = 1.0 + 1.0e-9;

const Cinfo* GssaStoich::initCinfo()
{
	//////////////////////////////////////////////////////////////
	// Process func Definitions
	//////////////////////////////////////////////////////////////
	static DestFinfo process( "process",
		"Handles process call",
		new ProcOpFunc< GssaStoich >( &GssaStoich::process ) );
	static DestFinfo reinit( "reinit",
		"Handles reinint call",
		new ProcOpFunc< GssaStoich >( &GssaStoich::reinit ) );
	static Finfo* procShared[] = {
		&process, &reinit
	};

	static SharedFinfo proc( "proc",
		"Shared message for process and reinit",
		procShared, sizeof( procShared ) / sizeof( const Finfo* )
	);

	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
	static ElementValueFinfo< GssaStoich, string > path(
		"path",
		"Path of reaction system to take over and solve",
		&GssaStoich::setPath,
		&GssaStoich::getPath
	);

	static ValueFinfo< GssaStoich, string > method( 
		"method", 
		"Numerical method to use for the GssaStoich. The default"
		"and currently the only method is Gillespie1.",
		&GssaStoich::setMethod,
		&GssaStoich::getMethod
	);

	///////////////////////////////////////////////////////
	//These are the fields of the stoich class
	static Finfo* gssaStoichFinfos[] =
	{
		&path,
		&method,
		&proc,
	};

	static string doc[] =
	{
		"Name", "GssaStoich",
		"Author", "Upinder S. Bhalla, 2008, 2011, NCBS",
		"Description", "GssaStoich: Gillespie Stochastic Simulation Algorithm object.Closely based on the "
		"Stoich object and inherits its handling functions for constructing the matrix. Sets up "
		"stoichiometry matrix based calculations from a\nwildcard path for the reaction system.Knows how to "
		"compute derivatives for most common things, also knows how to handle special cases where the object "
		"will have to do its own computation.Generates a stoichiometry matrix, which is useful for lots of "
		"other operations as well.",
	};

	static Cinfo gssaStoichCinfo(
		"GssaStoich",
		Stoich::initCinfo(),
		gssaStoichFinfos,
		sizeof( gssaStoichFinfos )/sizeof(Finfo *),
		new Dinfo< GssaStoich >()
	);

	return &gssaStoichCinfo;
}

static const Cinfo* gssaStoichCinfo = GssaStoich::initCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

GssaStoich::GssaStoich()
	: Stoich(), atot_( 0.0 ), t_( 0.0 )
{
	useOneWay_ = 1;
}
		
///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

string GssaStoich::getMethod() const
{
	return method_;
}

void GssaStoich::setMethod( string method )
{
	method_ = method;
}

void GssaStoich::setPath( const Eref& e, const Qinfo* q, string path )
{
	Stoich::setPath( e, q, path );
	rebuildMatrix();
}

///////////////////////////////////////////////////
// Setup funcs
///////////////////////////////////////////////////

/**
 * Virtual function to make the data structures from the 
 * object oriented specification of the signaling network.
 * Assumes that the Stoich::setPath has already been invoked.
 */
void GssaStoich::rebuildMatrix()
{
	// Stuff here to set up the dependencies.
	unsigned int numRates = N_.nColumns();
	assert ( numRates == rates_.size() );

	v_.resize( numReac_, 0 );

	// Here we fix the issue of having a single substrate at
	// more than first order. Its rate must be computed differently
	// for stoch calculations, since one molecule is consumed for
	// each order.
	for ( unsigned int i = 0; i < numRates; ++i ) {
		vector< unsigned int > molIndex;
		if ( rates_[i]->getReactants( molIndex ) > 1 ) {
			if ( molIndex.size() == 2 && molIndex[0] == molIndex[1] ) {
				RateTerm* oldRate = rates_[i];
				rates_[ i ] = new StochSecondOrderSingleSubstrate(
					oldRate->getR1(), molIndex[ 0 ]
				);
				delete oldRate;
			} else if ( molIndex.size() > 2 ) {
				RateTerm* oldRate = rates_[ i ];
				rates_[ i ] = new StochNOrder( oldRate->getR1(), molIndex);
				delete oldRate;
			}
		}
	}

	// Here we set up dependency stuff. First the basic reac deps.
	// transN_.setSize( numRates, N_.nRows() );
	assert( N_.nRows() == S_[0].size() );
	transN_ = N_;
	transN_.transpose();
	transN_.truncateRow( numVarPools_ );
	dependency_.resize( numRates );
	for ( unsigned int i = 0; i < numRates; ++i ) {
		transN_.getGillespieDependence( i, dependency_[ i ] );
	}

	// Fill in dependency list for MMEnzs: they depend on their enzymes.
	fillMmEnzDep();

	// Fill in dependency list for SumTots on reactions
	fillMathDep();

	makeReacDepsUnique();
}

/**
 * Fill in dependency list for all MMEnzs on reactions.
 * The dependencies of MMenz products are already in the system,
 * so here we just need to add cases where any reaction product
 * is the Enz of an MMEnz.
 */
void GssaStoich::fillMmEnzDep()
{
	unsigned int numRates = N_.nColumns();
	vector< unsigned int > indices;

	// Make a map to look up enzyme RateTerm using 
	// the key of the enzyme molecule.
	map< unsigned int, unsigned int > enzMolMap;
	for ( unsigned int i = 0; i < numRates; ++i ) {
		const MMEnzymeBase* mme = dynamic_cast< const MMEnzymeBase* >(
			rates_[ i ] );
		if ( mme ) {
			vector< unsigned int > reactants;
			mme->getReactants( reactants );
			if ( reactants.size() > 1 )
				enzMolMap[ reactants.front() ] = i; // front is enzyme.
		}
	}

	// Use the map to fill in deps.
	for ( unsigned int i = 0; i < numRates; ++i ) {
		// Extract the row of all molecules that depend on the reac.
		const int* entry;
		const unsigned int* colIndex;

		unsigned int numInRow = transN_.getRow( i, &entry, &colIndex );
		for( unsigned int j = 0; j < numInRow; ++j ) {
			map< unsigned int, unsigned int >::iterator pos = 
				enzMolMap.find( colIndex[j] );
			if ( pos != enzMolMap.end() )
				dependency_[i].push_back( pos->second );
		}

		/*
		transN_.getRowIndices( i, indices );
		for ( vector< unsigned int >::iterator 
			j = indices.begin(); j != indices.end(); ++j )
		{
			map< unsigned int, unsigned int >::iterator pos = 
				enzMolMap.find( *j );
			if ( pos != enzMolMap.end() )
				dependency_[i].push_back( pos->second );
		}
		*/
	}
}

/**
 * Fill in dependency list for all MathExpns on reactions.
 * Note that when a MathExpn updates, it alters a further
 * molecule, that may be a substrate for another reaction.
 * So we need to also add further dependent reactions.
 * In principle we might also cascade to deeper MathExpns. Later.
 */
void GssaStoich::fillMathDep()
{
	unsigned int numRates = N_.nColumns();
	dependentMathExpn_.resize( numRates );
	vector< unsigned int > indices;
	for ( unsigned int i = 0; i < numRates; ++i ) {
		vector< unsigned int >& dep = dependentMathExpn_[ i ];
		dep.resize( 0 );
		// Extract the row of all molecules that depend on the reac.
		// However, the sumTotals and other math dep stuff needs to be
		// redefined to use the math object.
		/*
		const int* entry;
		const unsigned int* colIndex;
		unsigned int numInRow = transN_.getRow( i, &entry, &colIndex );
		for ( unsigned int j = 0; j < sumTotals_.size(); ++j ) {
			if ( sumTotals_[ j ].hasInput( colIndex, S_ ) ) {
				insertMathDepReacs( j, i );
				dep.push_back( j );
			}
		}
		*/



		/*
		transN_.getRowIndices( i, indices );
		// This looks like N^2, but usually there will be very few
		// SumTots, so a simple linear scan should do.
		for ( unsigned int j = 0; j < sumTotals_.size(); ++j ) {
			if ( sumTotals_[ j ].hasInput( indices, S_ ) ) {
				insertMathDepReacs( j, i );
				dep.push_back( j );
			}
		}
		*/
	}
}

/**
 * Inserts reactions that depend on molecules modified by the
 * specified MathExpn, into the dependency list.
 * Later.
 */
void GssaStoich::insertMathDepReacs( unsigned int mathDepIndex,
	unsigned int firedReac )
{
	/*
	unsigned int molIndex = sumTotals_[ mathDepIndex ].target( S_ );
	vector< unsigned int > reacIndices;

	// Extract the row of all reacs that depend on the target molecule
	if ( N_.getRowIndices( molIndex, reacIndices ) > 0 ) {
		vector< unsigned int >& dep = dependency_[ firedReac ];
		dep.insert( dep.end(), reacIndices.begin(), reacIndices.end() );
	}
	*/
}

/*
void makeVecUnique( vector< unsigned int >& v )
{
	vector< unsigned int >::iterator pos = unique( v.begin(), v.end() );
	v.resize( pos - v.begin() );
}
*/


// Clean up dependency lists: Ensure only unique entries.
void GssaStoich::makeReacDepsUnique()
{
	unsigned int numRates = N_.nColumns();
	for ( unsigned int i = 0; i < numRates; ++i ) {
//		makeVecUnique( dependency_[ i ] );
		vector< unsigned int >& dep = dependency_[ i ];
		/// STL stuff follows, with the usual weirdness.
		vector< unsigned int >::iterator pos = 
			unique( dep.begin(), dep.end() );
		dep.resize( pos - dep.begin() );
	}
}

///////////////////////////////////////////////////
// Runtime funcs
///////////////////////////////////////////////////

void GssaStoich::reinit( const Eref& e, ProcPtr p )
{
	Stoich::innerReinit();
	// Here we round off up or down with prob depending on fractional
	// part of the init value.
	for ( vector< unsigned int >::iterator j = localMeshEntries_.begin(); 
		j != localMeshEntries_.end(); ++j ) {
		for ( vector< double >::iterator i = S_[*j].begin(); i != S_[*j].end(); ++i ) {
			double base = floor( *i );
			double frac = *i - base;
			if ( mtrand() > frac )
				*i = base;
			else
				*i = base + 1.0;
		}
	}
	t_ = 0.0;
	updateAllRates();
}


unsigned int GssaStoich::pickReac()
{
	double r = mtrand() * atot_;
	double sum = 0.0;
	// This is an inefficient way to do it. Can easily get to 
	// log time or thereabouts by doing one or two levels of 
	// subsidiary tables. Too many levels causes slow-down because
	// of overhead in managing the tree. 
	// Slepoy, Thompson and Plimpton 2008
	// report a linear time version.
	for ( vector< double >::iterator i = v_.begin(); i != v_.end(); ++i )
		if ( r < ( sum += *i ) )
			return static_cast< unsigned int >( i - v_.begin() );
	return v_.size();
}

void GssaStoich::process( const Eref& e, ProcPtr info )
{
	double nextt = info->currTime + info->dt;
	while ( t_ < nextt ) {
		// Figure out when the reaction will occur. The atot_
		// calculation actually estimates time for which reaction will
		// NOT occur, as atot_ sums all propensities.
		if ( atot_ <= 0.0 ) { // Nothing is going to happen.
			// We have to advance t_ because we may resume calculations
			// with a different atot at a later time.
			t_ = nextt;
			break;
		}
		unsigned int rindex = pickReac(); // Does a randnum call
		if ( rindex >= rates_.size() ) {
			// Probably cumulative roundoff error here. Simply
			// recalculate atot to avoid, and redo.
			updateAllRates();
			continue;
		}
		transN_.fireReac( rindex, S_[0] );

		// Math expns must be first, because they may alter 
		// substrate mol #.
		updateDependentMathExpn( dependentMathExpn_[ rindex ] );
		// The rates list includes rates dependent on mols changed
		// by the MathExpns.
		updateDependentRates( dependency_[ rindex ] );

		double r = mtrand();
		while ( r <= 0.0 )
			r = mtrand();
		t_ -= ( 1.0 / atot_ ) * log( r );
		// double dt = ( 1.0 / atot_ ) * log( 1.0 / mtrand() );
	}
}

void GssaStoich::updateDependentRates( const vector< unsigned int >& deps )
{
	for( vector< unsigned int >::const_iterator i = deps.begin(); 
		i != deps.end(); ++i ) {
		atot_ -= v_[ *i ];
		atot_ += ( v_[ *i ] = ( *rates_[ *i ] )( &S_[0][0] ) );
	}
}

/**
 * For now this just handles SumTots, but I think the formalism
 * will extend to general math expressions.
 * Will need to cascade to dependent rates
 */
void GssaStoich::updateDependentMathExpn( const vector< unsigned int >& deps )
{
	/*
	for( vector< unsigned int >::const_iterator i = deps.begin(); 
		i != deps.end(); ++i ) {
		sumTotals_[ *i ].sum();
	}
	*/
}

void GssaStoich::updateAllRates()
{
	// SumTots must go first because rates depend on them.
	/*
	vector< SumTotal >::const_iterator k;
	for ( k = sumTotals_.begin(); k != sumTotals_.end(); k++ )
		k->sum();
		*/

	atot_ = 0.0;
	for( unsigned int i = 0; i < rates_.size(); ++i ) {
		atot_ += ( v_[ i ] = ( *rates_[ i ] )( &S_[0][0] ) );
	}
	// Here we put in a safety factor into atot to ensure that
	// cumulative roundoff errors from the dependency 
	// addition/subtraction
	// steps do not make it smaller than the actual total of 
	// all the reactions. If that were to occur, we would begin
	// to lose calls to the last reaction.
	atot_ *= SAFETY_FACTOR;
}
