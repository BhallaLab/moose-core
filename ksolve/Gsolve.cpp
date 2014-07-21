/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"

#include "VoxelPoolsBase.h"
#include "ZombiePoolInterface.h"

#include "RateTerm.h"
#include "FuncTerm.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "GssaSystem.h"
#include "Stoich.h"
#include "GssaVoxelPools.h"

#include "Gsolve.h"

const unsigned int OFFNODE = ~0;

const Cinfo* Gsolve::initCinfo()
{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		
		static ValueFinfo< Gsolve, Id > stoich (
			"stoich",
			"Stoichiometry object for handling this reaction system.",
			&Gsolve::setStoich,
			&Gsolve::getStoich
		);

		static ValueFinfo< Gsolve, Id > compartment (
			"compartment",
			"Compartment that contains this reaction system.",
			&Gsolve::setCompartment,
			&Gsolve::getCompartment
		);

		static ReadOnlyValueFinfo< Gsolve, unsigned int > numLocalVoxels(
			"numLocalVoxels",
			"Number of voxels in the core reac-diff system, on the "
			"current solver. ",
			&Gsolve::getNumLocalVoxels
		);
		static LookupValueFinfo< 
				Gsolve, unsigned int, vector< double > > nVec(
			"nVec",
			"vector of pool counts",
			&Gsolve::setNvec,
			&Gsolve::getNvec
		);
		static ValueFinfo< Gsolve, unsigned int > numAllVoxels(
			"numAllVoxels",
			"Number of voxels in the entire reac-diff system, "
			"including proxy voxels to represent abutting compartments.",
			&Gsolve::setNumAllVoxels,
			&Gsolve::getNumAllVoxels
		);

		static ValueFinfo< Gsolve, unsigned int > numPools(
			"numPools",
			"Number of molecular pools in the entire reac-diff system, "
			"including variable, function and buffered.",
			&Gsolve::setNumPools,
			&Gsolve::getNumPools
		);

		static ValueFinfo< Gsolve, bool > useRandInit(
			"useRandInit",
			"Flag: True when using probabilistic (random) rounding. "
			"When initializing the mol# from floating-point Sinit values, "
			"we have two options. One is to look at each Sinit, and round "
			"to the nearest integer. The other is to look at each Sinit, "
			"and probabilistically round up or down depending on the  "
			"value. For example, if we had a Sinit value of 1.49,  "
			"this would always be rounded to 1.0 if the flag is false, "
			"and would be rounded to 1.0 and 2.0 in the ratio 51:49 if "
			"the flag is true. ",
			&Gsolve::setRandInit,
			&Gsolve::getRandInit
		);

		///////////////////////////////////////////////////////
		// DestFinfo definitions
		///////////////////////////////////////////////////////

		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< Gsolve >( &Gsolve::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< Gsolve >( &Gsolve::reinit ) );
		
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* gsolveFinfos[] =
	{
		&stoich,			// Value
		&numLocalVoxels,	// ReadOnlyValue
		&nVec,				// LookupValue
		&numAllVoxels,		// ReadOnlyValue
		&numPools,			// Value
		&proc,				// SharedFinfo
		// Here we put new fields that were not there in the Ksolve. 
		&useRandInit,		// Value
	};
	
	static Dinfo< Gsolve > dinfo;
	static  Cinfo gsolveCinfo(
		"Gsolve",
		Neutral::initCinfo(),
		gsolveFinfos,
		sizeof(gsolveFinfos)/sizeof(Finfo *),
		&dinfo
	);

	return &gsolveCinfo;
}

static const Cinfo* gsolveCinfo = Gsolve::initCinfo();

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////

Gsolve::Gsolve()
	: 
		pools_( 1 ),
		startVoxel_( 0 ),
		stoich_(),
		stoichPtr_( 0 ),
		compartment_()
{;}

Gsolve::~Gsolve()
{;}

//////////////////////////////////////////////////////////////
// Field Access functions
//////////////////////////////////////////////////////////////

Id Gsolve::getStoich() const
{
	return stoich_;
}

void Gsolve::setCompartment( Id compt )
{
	if ( ( compt.element()->cinfo()->isA( "ChemCompt" ) ) ) {
		compartment_ = compt;
		vector< double > vols = 
			Field< vector< double > >::get( compt, "voxelVolume" );
		if ( vols.size() > 0 ) {
			pools_.resize( vols.size() );
			for ( unsigned int i = 0; i < vols.size(); ++i ) {
				pools_[i].setVolume( vols[i] );
			}
		}
	}
}

Id Gsolve::getCompartment() const
{
	return compartment_;
}

void Gsolve::setStoich( Id stoich )
{
	// This call is done _before_ setting the path on stoich
	assert( stoich.element()->cinfo()->isA( "Stoich" ) );
	stoich_ = stoich;
	stoichPtr_ = reinterpret_cast< Stoich* >( stoich.eref().data() );
	sys_.stoich = stoichPtr_;
	sys_.isReady = false;
}

unsigned int Gsolve::getNumLocalVoxels() const
{
	return pools_.size();
}

unsigned int Gsolve::getNumAllVoxels() const
{
	return pools_.size(); // Need to redo.
}

// If we're going to do this, should be done before the zombification.
void Gsolve::setNumAllVoxels( unsigned int numVoxels )
{
	if ( numVoxels == 0 ) {
		return;
	}
	pools_.resize( numVoxels );
	sys_.isReady = false;
}

vector< double > Gsolve::getNvec( unsigned int voxel) const
{
	static vector< double > dummy;
	if ( voxel < pools_.size() ) {
		return const_cast< GssaVoxelPools* >( &( pools_[ voxel ]) )->Svec();
	}
	return dummy;
}

void Gsolve::setNvec( unsigned int voxel, vector< double > nVec )
{
	if ( voxel < pools_.size() ) {
		if ( nVec.size() != pools_[voxel].size() ) {
			cout << "Warning: Gsolve::setNvec: size mismatch ( " <<
				nVec.size() << ", " << pools_[voxel].size() << ")\n";
			return;
		}
		double* s = pools_[voxel].varS();
		for ( unsigned int i = 0; i < nVec.size(); ++i )
			s[i] = nVec[i];
	}
}

bool Gsolve::getRandInit() const
{
	return sys_.useRandInit;
}

void Gsolve::setRandInit( bool val )
{
	sys_.useRandInit = val;
}

//////////////////////////////////////////////////////////////
// Process operations.
//////////////////////////////////////////////////////////////
void Gsolve::process( const Eref& e, ProcPtr p )
{
	if ( !stoichPtr_ )
		return;
	for ( vector< GssaVoxelPools >::iterator 
					i = pools_.begin(); i != pools_.end(); ++i ) {
		i->advance( p, &sys_ );
	}
}

void Gsolve::reinit( const Eref& e, ProcPtr p )
{
	if ( !stoichPtr_ )
		return;
	if ( !sys_.isReady )
		rebuildGssaSystem();
	
	for ( vector< GssaVoxelPools >::iterator 
					i = pools_.begin(); i != pools_.end(); ++i ) {
		i->reinit( &sys_ );
	}
}
//////////////////////////////////////////////////////////////
// Solver setup
//////////////////////////////////////////////////////////////

void Gsolve::rebuildGssaSystem()
{
	stoichPtr_->convertRatesToStochasticForm();
	sys_.transposeN = stoichPtr_->getStoichiometryMatrix();
	sys_.transposeN.transpose();
	sys_.transposeN.truncateRow( stoichPtr_->getNumVarPools() );
	vector< vector< unsigned int > > & dep = sys_.dependency;
	dep.resize( stoichPtr_->getNumRates() );
	for ( unsigned int i = 0; i < stoichPtr_->getNumRates(); ++i ) {
		sys_.transposeN.getGillespieDependence( i, dep[i] );
	}
	fillMmEnzDep();
	fillMathDep();
	makeReacDepsUnique();
	for ( vector< GssaVoxelPools >::iterator 
					i = pools_.begin(); i != pools_.end(); ++i ) {
		i->setNumReac( stoichPtr_->getNumRates() );
	}
	sys_.isReady = true;
}

/**
 * Fill in dependency list for all MMEnzs on reactions.
 * The dependencies of MMenz products are already in the system,
 * so here we just need to add cases where any reaction product
 * is the Enz of an MMEnz.
 */
void Gsolve::fillMmEnzDep()
{
	unsigned int numRates = stoichPtr_->getNumRates();
	vector< unsigned int > indices;

	// Make a map to look up enzyme RateTerm using 
	// the key of the enzyme molecule.
	map< unsigned int, unsigned int > enzMolMap;
	for ( unsigned int i = 0; i < numRates; ++i ) {
		const MMEnzymeBase* mme = dynamic_cast< const MMEnzymeBase* >(
			stoichPtr_->rates( i ) );
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

		unsigned int numInRow = 
				sys_.transposeN.getRow( i, &entry, &colIndex );
		for( unsigned int j = 0; j < numInRow; ++j ) {
			map< unsigned int, unsigned int >::iterator pos = 
				enzMolMap.find( colIndex[j] );
			if ( pos != enzMolMap.end() )
				sys_.dependency[i].push_back( pos->second );
		}
	}
}

/**
 * Fill in dependency list for all MathExpns on reactions.
 * Note that when a MathExpn updates, it alters a further
 * molecule, that may be a substrate for another reaction.
 * So we need to also add further dependent reactions.
 * In principle we might also cascade to deeper MathExpns. Later.
 */
void Gsolve::fillMathDep()
{
	// create map of funcs that depend on specified molecule.
	vector< vector< unsigned int > > funcMap( stoichPtr_->getNumAllPools());
	unsigned int numFuncs = stoichPtr_->getNumFuncs();
	for ( unsigned int i = 0; i < numFuncs; ++i ) {
		const FuncTerm *f = stoichPtr_->funcs( i );
		vector< unsigned int > molIndex;
		unsigned int numMols = f->getReactants( molIndex );
		for ( unsigned int j = 0; j < numMols; ++j )
			funcMap[ molIndex[j] ].push_back( i );
	}
	// The output of each func is a mol indexed as 
	// numVarMols + numBufMols + i
	unsigned int funcOffset = 
			stoichPtr_->getNumVarPools() + stoichPtr_->getNumBufPools();
	unsigned int numRates = stoichPtr_->getNumRates();
	sys_.dependentMathExpn.resize( numRates );
	vector< unsigned int > indices;
	for ( unsigned int i = 0; i < numRates; ++i ) {
		vector< unsigned int >& dep = sys_.dependentMathExpn[ i ];
		dep.resize( 0 );
		// Extract the row of all molecules that depend on the reac.
		const int* entry;
		const unsigned int* colIndex;
		unsigned int numInRow = 
				sys_.transposeN.getRow( i, &entry, &colIndex );
		for ( unsigned int j = 0; j < numInRow; ++j ) {
			unsigned int molIndex = colIndex[j];
			vector< unsigned int >& funcs = funcMap[ molIndex ];
			dep.insert( dep.end(), funcs.begin(), funcs.end() );
			for ( unsigned int k = 0; k < funcs.size(); ++k ) {
				unsigned int outputMol = funcs[k] + funcOffset;
				// Insert reac deps here. Columns are reactions.
				vector< int > e; // Entries: we don't need.
				vector< unsigned int > c; // Column index: the reactions.
				stoichPtr_->getStoichiometryMatrix().
						getRow( outputMol, e, c );
				// Each of the reacs (col entries) depend on this func.
				vector< unsigned int > rdep = sys_.dependency[i];
				rdep.insert( rdep.end(), c.begin(), c.end() );
			}
		}
	}
}

/**
 * Inserts reactions that depend on molecules modified by the
 * specified MathExpn, into the dependency list.
 * Later.
 */
void Gsolve::insertMathDepReacs( unsigned int mathDepIndex,
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

// Clean up dependency lists: Ensure only unique entries.
void Gsolve::makeReacDepsUnique()
{
	unsigned int numRates = stoichPtr_->getNumRates();
	for ( unsigned int i = 0; i < numRates; ++i ) {
		vector< unsigned int >& dep = sys_.dependency[ i ];
		/// STL stuff follows, with the usual weirdness.
		vector< unsigned int >::iterator pos = 
			unique( dep.begin(), dep.end() );
		dep.resize( pos - dep.begin() );
	}
}

//////////////////////////////////////////////////////////////
// Solver ops
//////////////////////////////////////////////////////////////
unsigned int Gsolve::getPoolIndex( const Eref& e ) const
{
	return stoichPtr_->convertIdToPoolIndex( e.id() );
}

unsigned int Gsolve::getVoxelIndex( const Eref& e ) const
{
	unsigned int ret = e.dataIndex();
	if ( ret < startVoxel_  || ret >= startVoxel_ + pools_.size() ) 
		return OFFNODE;
	return ret - startVoxel_;
}

void Gsolve::setDsolve( Id dsolve )
{
		;
}


//////////////////////////////////////////////////////////////
// Zombie Pool Access functions
//////////////////////////////////////////////////////////////

void Gsolve::setN( const Eref& e, double v )
{
	unsigned int vox = getVoxelIndex( e );
	if ( vox != OFFNODE )
		pools_[vox].setN( getPoolIndex( e ), v );
}

double Gsolve::getN( const Eref& e ) const
{
	unsigned int vox = getVoxelIndex( e );
	if ( vox != OFFNODE )
		return pools_[vox].getN( getPoolIndex( e ) );
	return 0.0;
}

void Gsolve::setNinit( const Eref& e, double v )
{
	unsigned int vox = getVoxelIndex( e );
	if ( vox != OFFNODE )
		pools_[vox].setNinit( getPoolIndex( e ), v );
}

double Gsolve::getNinit( const Eref& e ) const
{
	unsigned int vox = getVoxelIndex( e );
	if ( vox != OFFNODE )
		return pools_[vox].getNinit( getPoolIndex( e ) );
	return 0.0;
}

void Gsolve::setDiffConst( const Eref& e, double v )
{
		; // Do nothing.
}

double Gsolve::getDiffConst( const Eref& e ) const
{
		return 0;
}

void Gsolve::setNumPools( unsigned int numPoolSpecies )
{
	sys_.isReady = false;
	unsigned int numVoxels = pools_.size();
	for ( unsigned int i = 0 ; i < numVoxels; ++i ) {
		pools_[i].resizeArrays( numPoolSpecies );
	}
}

unsigned int Gsolve::getNumPools() const
{
	if ( pools_.size() > 0 )
		return pools_[0].size();
	return 0;
}

void Gsolve::getBlock( vector< double >& values ) const
{
	unsigned int startVoxel = values[0];
	unsigned int numVoxels = values[1];
	unsigned int startPool = values[2];
	unsigned int numPools = values[3];

	assert( startVoxel >= startVoxel_ );
	assert( numVoxels <= pools_.size() );
	assert( pools_.size() > 0 );
	assert( numPools + startPool <= pools_[0].size() );
	values.resize( 4 + numVoxels * numPools );

	for ( unsigned int i = 0; i < numVoxels; ++i ) {
		const double* v = pools_[ startVoxel + i ].S();
		for ( unsigned int j = 0; j < numPools; ++j ) {
			values[ 4 + j * numVoxels + i]  = v[ j + startPool ];
		}
	}
}

void Gsolve::setBlock( const vector< double >& values )
{
	unsigned int startVoxel = values[0];
	unsigned int numVoxels = values[1];
	unsigned int startPool = values[2];
	unsigned int numPools = values[3];

	assert( startVoxel >= startVoxel_ );
	assert( numVoxels <= pools_.size() );
	assert( pools_.size() > 0 );
	assert( numPools + startPool <= pools_[0].size() );

	for ( unsigned int i = 0; i < numVoxels; ++i ) {
		double* v = pools_[ startVoxel + i ].varS();
		for ( unsigned int j = 0; j < numPools; ++j ) {
			v[ j + startPool ] = values[ 4 + j * numVoxels + i ];
		}
	}
}

// Inherited virtual
void Gsolve::setupCrossSolverReacs( const map< Id, vector< Id > >& xr,
	   Id otherStoich )
{
}
