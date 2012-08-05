/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"
#include "ReacBase.h"
#include "ZombieReac.h"
#include "Reac.h"
#include "ElementValueFinfo.h"
#include "DataHandlerWrapper.h"

const Cinfo* ZombieReac::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions: All inherited.
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions: All inherited.
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions: All inherited
		//////////////////////////////////////////////////////////////

	static Cinfo zombieReacCinfo (
		"ZombieReac",
		ReacBase::initCinfo(),
		0,
		0,
		new Dinfo< ZombieReac >()
	);

	return &zombieReacCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieReacCinfo = ZombieReac::initCinfo();

static const SrcFinfo2< double, double >* toSub = 
 	dynamic_cast< const SrcFinfo2< double, double >* >(
					zombieReacCinfo->findFinfo( "toSub" ) );

static const SrcFinfo2< double, double >* toPrd = 
 	dynamic_cast< const SrcFinfo2< double, double >* >(
					zombieReacCinfo->findFinfo( "toPrd" ) );

ZombieReac::ZombieReac()
{;}

ZombieReac::~ZombieReac()
{;}


//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void ZombieReac::vRemesh( const Eref& e, const Qinfo* q )
{
	stoich_->setReacKf( e, concKf_ );
	stoich_->setReacKb( e, concKb_ );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

// This conversion is deprecated, used mostly for kkit conversions.
void ZombieReac::vSetNumKf( const Eref& e, const Qinfo* q, double v )
{
	double volScale = convertConcToNumRateUsingMesh( e, toSub, 0 );
	concKf_ = v * volScale;
	stoich_->setReacKf( e, concKf_ );
}

double ZombieReac::vGetNumKf( const Eref& e, const Qinfo* q ) const
{
	// Return value for voxel 0. Conceivably I might want to use the
	// DataId part to specify which voxel to use, but that isn't in the
	// current definition for Reacs as being a single entity for the entire
	// compartment.
	return stoich_->getR1( stoich_->convertIdToReacIndex( e.id() ), 0 );
	// return rates_[ convertIdToReacIndex( e.id() ) ]->getR1();
}

// Deprecated, used for kkit conversion backward compatibility
void ZombieReac::vSetNumKb( const Eref& e, const Qinfo* q, double v )
{
	double volScale = convertConcToNumRateUsingMesh( e, toPrd, 0 );
	concKb_ = v * volScale;
	stoich_->setReacKb( e, concKb_ );
}

double ZombieReac::vGetNumKb( const Eref& e, const Qinfo* q ) const
{
	if ( stoich_->getOneWay() ) {
		return stoich_->getR1( stoich_->convertIdToReacIndex( e.id() ) + 1, 0 );
	} else {
		return stoich_->getR2( stoich_->convertIdToReacIndex( e.id() ), 0 );
	}
}

void ZombieReac::vSetConcKf( const Eref& e, const Qinfo* q, double v )
{
	concKf_ = v;
	stoich_->setReacKf( e, v );
}

double ZombieReac::vGetConcKf( const Eref& e, const Qinfo* q ) const
{
	return concKf_;
}

void ZombieReac::vSetConcKb( const Eref& e, const Qinfo* q, double v )
{
	concKb_ = v;
	stoich_->setReacKb( e, v );
}

double ZombieReac::vGetConcKb( const Eref& e, const Qinfo* q ) const
{
	return concKb_;
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

ZeroOrder* ZombieReac::makeHalfReaction( 
	Element* orig, double rate, const SrcFinfo* finfo ) const
{
	vector< Id > mols;
	unsigned int numReactants = orig->getNeighbours( mols, finfo ); 
	ZeroOrder* rateTerm = 0;
	if ( numReactants == 1 ) {
		rateTerm = 
			new FirstOrder( rate, stoich_->convertIdToPoolIndex( mols[0] ) );
	} else if ( numReactants == 2 ) {
		rateTerm = new SecondOrder( rate,
				stoich_->convertIdToPoolIndex( mols[0] ), 
				stoich_->convertIdToPoolIndex( mols[1] ) );
	} else if ( numReactants > 2 ) {
		vector< unsigned int > v;
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			v.push_back( stoich_->convertIdToPoolIndex( mols[i] ) );
		}
		rateTerm = new NOrder( rate, v );
	} else {
		cout << "Error: ZombieReac::makeHalfReaction: zero reactants\n";
	}
	return rateTerm;
}

// Virtual func called in zombify before fields are assigned.
void ZombieReac::setSolver( Id solver, Id orig )
{
		/*
	static const SrcFinfo* sub = dynamic_cast< const SrcFinfo* >(
		Reac::initCinfo()->findFinfo( "toSub" ) );
	static const SrcFinfo* prd = dynamic_cast< const SrcFinfo* >(
		Reac::initCinfo()->findFinfo( "toPrd" ) );
	assert( sub );
	assert( prd );
	*/
	assert( solver != Id() );

	stoich_ = reinterpret_cast< Stoich* >( solver.eref().data( ) );
	/*
	ReacBase* reac = reinterpret_cast< ReacBase* >( orig.eref()->data() );

	double concKf = reac->getConcKf( orig.eref(), 0 );
	double concKb = reac->getConcKb( orig.eref(), 0 );
	ZeroOrder* forward = makeHalfReaction( orig.element(), concKf, sub );
	ZeroOrder* reverse = makeHalfReaction( orig.element(), concKb, prd );
	*/
	// Values will be filled in later by the zombify function.
	ZeroOrder* forward = makeHalfReaction( orig.element(), 0, toSub );
	ZeroOrder* reverse = makeHalfReaction( orig.element(), 0, toPrd );

	stoich_->installReaction( forward, reverse, orig );
}

/*
// static func
void ZombieReac::zombify( Element* solver, Element* orig )
{

	DataHandler* dh = orig->dataHandler()->copyUsingNewDinfo( ZombieReac::initCinfo()->dinfo() );

	Reac* reac = reinterpret_cast< Reac* >( orig->dataHandler()->data( 0 ));
	double concKf = reac->getConcKf( Eref( orig, 0 ), 0 );
	double concKb = reac->getConcKb( Eref( orig, 0 ), 0 );

	ZombieReac* zr = reinterpret_cast< ZombieReac* >( dh->data( 0 ) );
	zr->concKf_ = concKf;
	zr->concKb_ = concKb;
	zr->stoich_ = reinterpret_cast< Stoich* >( solver->dataHandler()->data( 0 ) );


	ZeroOrder* forward = zr->makeHalfReaction( orig, concKf, sub );
	ZeroOrder* reverse = zr->makeHalfReaction( orig, concKb, prd );

	zr->stoich_->installReaction( forward, reverse, orig->id() );

	orig->zombieSwap( ZombieReac::initCinfo(), dh );

	zr->stoich_->setReacKf( Eref( orig, 0 ), concKf );
	zr->stoich_->setReacKb( Eref( orig, 0 ), concKb );
}
*/
