/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "RateTerm.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "Stoich.h"
#include "Mol.h"
#include "Reac.h"
#include "Enz.h"
#include "MMenz.h"
#include "ZombieMol.h"
#include "ZombieReac.h"
#include "ZombieEnz.h"
#include "ZombieMMenz.h"

#define EPSILON 1e-15

/*
static Finfo* reacShared[] = {
	&reacDest, &nOut
};
*/

const Cinfo* Stoich::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Stoich, bool > useOneWay(
			"useOneWayReacs",
			"Flag: use bidirectional or one-way reacs. One-way is needed"
			"for Gillespie type stochastic calculations. Two-way is"
			"likely to be margninally more efficient in ODE calculations",
			&Stoich::setOneWay,
			&Stoich::getOneWay
		);

		static ElementValueFinfo< Stoich, string > path(
			"path",
			"Path of reaction system to take over",
			&Stoich::setPath,
			&Stoich::getPath
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new EpFunc1< Stoich, ProcPtr >( &Stoich::eprocess ) );
		static DestFinfo reinit( "reinit",
			"Handles reinint call",
			new EpFunc1< Stoich, ProcPtr >( &Stoich::reinit ) );

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////
		/*
		static SharedFinfo reac( "reac",
			"Connects to reaction",
			reacShared, sizeof( reacShared ) / sizeof( const Finfo* )
		);
		*/

	static Finfo* stoichFinfos[] = {
		&useOneWay,		// Value
		&path,			// Value
		&process,			// DestFinfo
		&reinit,			// DestFinfo
	};

	static Cinfo stoichCinfo (
		"Stoich",
		Neutral::initCinfo(),
		stoichFinfos,
		sizeof( stoichFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Stoich >()
	);

	return &stoichCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* stoichCinfo = Stoich::initCinfo();

Stoich::Stoich()
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Stoich::eprocess( Eref e, const Qinfo* q, ProcInfo* p )
{
	process( p, e );
}

void Stoich::process( const ProcInfo* p, const Eref& e )
{
	;
}

void Stoich::reinit( Eref e, const Qinfo*q, ProcInfo* p )
{
	;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Stoich::setOneWay( bool v )
{
	useOneWay_ = v;
}

bool Stoich::getOneWay() const
{
	return useOneWay_;
}

void Stoich::allocateObjMap( const vector< Id >& elist )
{
	objMapStart_ = ~0;
	unsigned int maxId = 0;
	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		if ( objMapStart_ > i->value() )
			objMapStart_ = i->value();
		if ( maxId < i->value() )
			maxId = i->value();
	}
	objMap_.resize(0);
	objMap_.resize( 1 + maxId - objMapStart_, 0 );
	assert( objMap_.size() >= elist.size() );

	/*
	for ( unsigned int i = 0; i < elist.size(); ++i ) {
		unsigned int index = elist[i].value() - objMapStart_;
		objMap_[ index ] = i;
	}
	*/
}

void Stoich::allocateModel( const vector< Id >& elist )
{
	static const Cinfo* molCinfo = Mol::initCinfo();
	static const Cinfo* reacCinfo = Reac::initCinfo();
	static const Cinfo* enzCinfo = Enz::initCinfo();
	static const Cinfo* mmEnzCinfo = MMenz::initCinfo();
	numVarMols_ = 0;
	numReac_ = 0;
	unsigned int numBufMols = 0;
	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		Element* ei = (*i)();
		if ( ei->cinfo() == molCinfo ) {
			objMap_[ i->value() - objMapStart_ ] = numVarMols_;
			++numVarMols_;
		}
		/*
		if ( ei->cinfo() == bufMolCinfo ) {
			++numBufMols;
		}
		*/
		if ( ei->cinfo() == reacCinfo || ei->cinfo() == mmEnzCinfo ) {
			objMap_[ i->value() - objMapStart_ ] = numReac_;
			++numReac_;
		}
		if ( ei->cinfo() == enzCinfo ) {
			objMap_[ i->value() - objMapStart_ ] = numReac_;
			numReac_ += 2;
		}
	}

	S_.resize( numVarMols_ + numBufMols, 0.0 );
	Sinit_.resize( numVarMols_ + numBufMols, 0.0 );
	rates_.resize( numReac_ );
}

void Stoich::zombifyModel( Eref& e, const vector< Id >& elist )
{
	static const Cinfo* molCinfo = Mol::initCinfo();
	static const Cinfo* reacCinfo = Reac::initCinfo();
	static const Cinfo* enzCinfo = Enz::initCinfo();
	static const Cinfo* mmEnzCinfo = MMenz::initCinfo();

	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		Element* ei = (*i)();
		if ( ei->cinfo() == molCinfo ) {
			ZombieMol::zombify( e.element(), (*i)() );
		}
		/*
		else if ( ei->cinfo() == bufMolCinfo ) {
			ZombieBufMol::zombify( e.element(), (*i)() );
		}
		*/
		else if ( ei->cinfo() == reacCinfo ) {
			ZombieReac::zombify( e.element(), (*i)() );
		}
		else if ( ei->cinfo() == mmEnzCinfo ) {
			ZombieMMenz::zombify( e.element(), (*i)() );
		}
		else if ( ei->cinfo() == enzCinfo ) {
			ZombieEnz::zombify( e.element(), (*i)() );
		}
	}
}

void Stoich::buildStoichFromModel( const vector< Id >& elist )
{
}

void Stoich::setPath( Eref e, const Qinfo* q, string v )
{
	if ( path_ != "" && path_ != v ) {
		// unzombify( path_ );
		cout << "Stoich::setPath: need to clear old path.\n";
		return;
	}
	path_ = v;
	vector< Id > elist;
	Shell::wildcard( path_, elist );

	allocateObjMap( elist );
	allocateModel( elist );
	zombifyModel( e, elist );
	buildStoichFromModel( elist );

	cout << "Zombified " << numVarMols_ << " Molecules, " <<
		numReac_ << " reactions\n";
}

string Stoich::getPath( Eref e, const Qinfo* q ) const
{
	return path_;
}

unsigned int Stoich::convertIdToMolIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < S_.size() );
	return i;
}

unsigned int Stoich::convertIdToReacIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < rates_.size() );
	return i;
}
