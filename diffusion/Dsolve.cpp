/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "ZombiePoolInterface.h"
#include "DiffPoolVec.h"
#include "Dsolve.h"

Dsolve::Dsolve()
{;}

Dsolve::~Dsolve()
{;}

unsigned int Dsolve::getNumVarPools() const
{
	return 0;
}

void Dsolve::setPath( const Eref& e, string v )
{;}

string Dsolve::getPath( const Eref& e ) const
{
	return "foo";
}

void zombifyModel( const Eref& e, const vector< Id >& elist )
{
	;
}


void unZombifyModel( const Eref& e )
{
		;
}
//////////////////////////////////////////////////////////////
// Zombie Pool Access functions
//////////////////////////////////////////////////////////////

unsigned int Dsolve::convertIdToPoolIndex( const Eref& e ) const
{
	unsigned int i  = e.id().value() - poolMapStart_;
	if ( i < poolMap_.size() ) {
		return poolMap_[i];
	}
	cout << "Warning: Dsolve::convertIdToPoollndex: Id out of range, (" <<
		poolMapStart_ << ", " << e.id() << ", " <<
		poolMap_.size() + poolMapStart_ << "\n";
	return 0;
}

void Dsolve::setN( const Eref& e, double v )
{
	unsigned int vox = e.dataIndex();
	if ( vox < numVoxels_ )
		pools_[ convertIdToPoolIndex( e ) ].setN( vox, v );
	else 
		cout << "Warning: Dsolve::setN: Eref out of range\n";
}

double Dsolve::getN( const Eref& e ) const
{
	unsigned int vox = e.dataIndex();
	if ( vox <  numVoxels_ )
		return pools_[ convertIdToPoolIndex( e ) ].getN( vox );
	cout << "Warning: Dsolve::getN: Eref out of range\n";
	return 0.0;
}

void Dsolve::setNinit( const Eref& e, double v )
{
	unsigned int vox = e.dataIndex();
	if ( vox < numVoxels_ )
		pools_[ convertIdToPoolIndex( e ) ].setNinit( vox, v );
	else 
		cout << "Warning: Dsolve::setNinit: Eref out of range\n";
}

double Dsolve::getNinit( const Eref& e ) const
{
	unsigned int vox = e.dataIndex();
	if ( vox < numVoxels_ )
		return pools_[ convertIdToPoolIndex( e ) ].getNinit( vox );
	cout << "Warning: Dsolve::getNinit: Eref out of range\n";
	return 0.0;
}

void Dsolve::setDiffConst( const Eref& e, double v )
{
	pools_[ convertIdToPoolIndex( e ) ].setDiffConst( v );
}

double Dsolve::getDiffConst( const Eref& e ) const
{
	return pools_[ convertIdToPoolIndex( e ) ].getDiffConst();
}

void Dsolve::setNumPools( unsigned int numPoolSpecies )
{
	// Decompose numPoolSpecies here, assigning some to each node.
	numTotPools_ = numPoolSpecies;
	numLocalPools_ = numPoolSpecies;
	poolStartIndex_ = 0;

	pools_.resize( numLocalPools_ );
	for ( unsigned int i = 0 ; i < numLocalPools_; ++i ) {
		pools_[i].setNumVoxels( numVoxels_ );
		// pools_[i].setId( reversePoolMap_[i] );
		// pools_[i].setParent( me );
	}
}

unsigned int Dsolve::getNumPools() const
{
	return numTotPools_;
}
