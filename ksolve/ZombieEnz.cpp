/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"
#include "ElementValueFinfo.h"
#include "EnzBase.h"
#include "CplxEnzBase.h"
#include "ZombieEnz.h"
#include "Enz.h"
#include "DataHandlerWrapper.h"

const Cinfo* ZombieEnz::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////

	static Cinfo zombieEnzCinfo (
		"ZombieEnz",
		CplxEnzBase::initCinfo(),
		0,
		0,
		new Dinfo< ZombieEnz >()
	);

	return &zombieEnzCinfo;
}
//////////////////////////////////////////////////////////////

static const Cinfo* zombieEnzCinfo = ZombieEnz::initCinfo();

static const SrcFinfo2< double, double >* toSub =
	dynamic_cast< const SrcFinfo2< double, double >* >(
	zombieEnzCinfo->findFinfo( "toSub" ) );

//////////////////////////////////////////////////////////////
// ZombieEnz internal functions
//////////////////////////////////////////////////////////////

ZombieEnz::ZombieEnz( )
{ ; }
ZombieEnz::~ZombieEnz( )
{ ; }

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void ZombieEnz::vRemesh( const Eref& e, const Qinfo* q )
{   
	stoich_->setEnzK1( e, concK1_ );
}   


//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZombieEnz::vSetK1( const Eref& e, const Qinfo* q, double v )
{
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub, 1 );

	concK1_ = v / volScale;
	stoich_->setEnzK1( e, concK1_ );
}

double ZombieEnz::vGetK1( const Eref& e, const Qinfo* q ) const
{
	return stoich_->getR1( stoich_->convertIdToReacIndex( e.id() ), 0 );
}

void ZombieEnz::vSetK2( const Eref& e, const Qinfo* q, double v )
{
	stoich_->setEnzK2( e, v );
}

double ZombieEnz::vGetK2( const Eref& e, const Qinfo* q ) const
{
	if ( stoich_->getOneWay() )
		return stoich_->getR1( 
			stoich_->convertIdToReacIndex( e.id() ) + 1, 0 );
	else
		return stoich_->getR2( stoich_->convertIdToReacIndex( e.id() ), 0 );
}

void ZombieEnz::vSetKcat( const Eref& e, const Qinfo* q, double v )
{
	stoich_->setEnzK3( e, v );
}

double ZombieEnz::vGetKcat( const Eref& e, const Qinfo* q ) const
{
	if ( stoich_->getOneWay() )
		return stoich_->getR1(
			stoich_->convertIdToReacIndex( e.id() ) + 2, 0 );
	else
		return stoich_->getR1(
			stoich_->convertIdToReacIndex( e.id() ) + 1, 0 );
}


void ZombieEnz::vSetKm( const Eref& e, const Qinfo* q, double v )
{
	double k2 = getK2( e, q );
	double k3 = getKcat( e, q );
	concK1_ = ( k2 + k3 ) / v;
	stoich_->setEnzK1( e, concK1_ );
}

double ZombieEnz::vGetKm( const Eref& e, const Qinfo* q ) const
{
	double k2 = getK2( e, q );
	double k3 = getKcat( e, q );

	return ( k2 + k3 ) / concK1_;
}

void ZombieEnz::vSetNumKm( const Eref& e, const Qinfo* q, double v )
{
	double k2 = getK2( e, q );
	double k3 = getKcat( e, q );
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub, 1 );
	concK1_ = volScale * ( k2 + k3 ) / v;

	stoich_->setEnzK1( e, concK1_ );
}

double ZombieEnz::vGetNumKm( const Eref& e, const Qinfo* q ) const
{
	double k2 = vGetK2( e, q );
	double k3 = vGetKcat( e, q );
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub, 1 );

	return ( k2 + k3 ) / ( concK1_ * volScale );
}

void ZombieEnz::vSetRatio( const Eref& e, const Qinfo* q, double v )
{
	double Km = getKm( e, q );
	double k2 = getK2( e, q );
	double k3 = getKcat( e, q );

	k2 = v * k3;

	stoich_->setEnzK2( e, k2 );
	double k1 = ( k2 + k3 ) / Km;

	setConcK1( e, q, k1 );
}

double ZombieEnz::vGetRatio( const Eref& e, const Qinfo* q ) const
{
	double k2 = getK2( e, q );
	double k3 = getKcat( e, q );
	return k2 / k3;
}

void ZombieEnz::vSetConcK1( const Eref& e, const Qinfo* q, double v )
{
	concK1_ = v;
	stoich_->setEnzK1( e, v );
}

double ZombieEnz::vGetConcK1( const Eref& e, const Qinfo* q ) const
{
	return concK1_;
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

ZeroOrder* ZombieEnz::makeHalfReaction( 
	Element* orig, double rate, const SrcFinfo* finfo, Id enz ) const
{
	vector< Id > pools;
	unsigned int numReactants = orig->getNeighbours( pools, finfo ); 
	if ( enz != Id() ) // Used to add the enz to the reactants.
		pools.push_back( enz );
	numReactants = pools.size();

	ZeroOrder* rateTerm = 0;
	if ( numReactants == 1 ) {
		rateTerm = 
			new FirstOrder( rate, stoich_->convertIdToPoolIndex( pools[0] ) );
	} else if ( numReactants == 2 ) {
		rateTerm = new SecondOrder( rate,
				stoich_->convertIdToPoolIndex( pools[0] ), 
				stoich_->convertIdToPoolIndex( pools[1] ) );
	} else if ( numReactants > 2 ) {
		vector< unsigned int > v;
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			v.push_back( stoich_->convertIdToPoolIndex( pools[i] ) );
		}
		rateTerm = new NOrder( rate, v );
	} else {
		cout << "Error: ZombieEnz::makeHalfReaction: zero reactants\n";
	}
	return rateTerm;
}

// static func
void ZombieEnz::setSolver( Id solver, Id orig )
{
	static const SrcFinfo* sub = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toSub" ) );
	static const SrcFinfo* prd = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toPrd" ) );
	static const SrcFinfo* enzFinfo = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toEnz" ) );
	static const SrcFinfo* cplx = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toCplx" ) );

	assert( sub );
	assert( prd );
	assert( enzFinfo );
	assert( cplx );

	stoich_ = reinterpret_cast< Stoich* >( solver.eref().data() );
	/*
	Enz* enz = reinterpret_cast< Enz* >( orig->dataHandler()->data( 0 ) );
	Eref oer( orig, 0 );

	double concK1 = enz->getConcK1( oer, 0 );
	*/

	vector< Id > pools;
	unsigned int numReactants = orig.element()->getNeighbours( pools, enzFinfo ); 
	assert( numReactants == 1 );
	Id enzMolId = pools[0];
	/*
	ZeroOrder* r1 = z->makeHalfReaction( orig, enz->getK1( oer, 0 ), sub, enzMolId );
	ZeroOrder* r2 = z->makeHalfReaction( orig, enz->getK2(), cplx, Id() );
	ZeroOrder* r3 = z->makeHalfReaction( orig, enz->getK3(), cplx, Id() );
	*/
	ZeroOrder* r1 = makeHalfReaction( orig.element(), 1, sub, enzMolId );
	ZeroOrder* r2 = makeHalfReaction( orig.element(), 1, cplx, Id() );
	ZeroOrder* r3 = makeHalfReaction( orig.element(), 1, cplx, Id() );

	numReactants = orig.element()->getNeighbours( pools, prd ); 
	stoich_->installEnzyme( r1, r2, r3, orig, enzMolId, pools );

	/*
	orig->zombieSwap( ZombieEnz::initCinfo(), dh );
	z->concK1_ = concK1;
	z->stoich_->setEnzK1( Eref( orig, 0 ), concK1 );
	*/
}
