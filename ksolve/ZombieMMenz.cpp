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
#include "ZombieMMenz.h"
#include "MMenz.h"
#include "DataHandlerWrapper.h"

const Cinfo* ZombieMMenz::initCinfo()
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

	static Cinfo zombieMMenzCinfo (
		"ZombieMMenz",
		EnzBase::initCinfo(),
		0,
		0,
		new Dinfo< ZombieMMenz >()
	);

	return &zombieMMenzCinfo;
}

//////////////////////////////////////////////////////////////

static const Cinfo* zombieMMenzCinfo = ZombieMMenz::initCinfo();

static const SrcFinfo2< double, double >* toSub = 
    dynamic_cast< const SrcFinfo2< double, double >* >(
	zombieMMenzCinfo->findFinfo( "toSub" ) );

static const SrcFinfo2< double, double >* toPrd = 
	dynamic_cast< const SrcFinfo2< double, double >* >(
	zombieMMenzCinfo->findFinfo( "toPrd" ) );

//////////////////////////////////////////////////////////////
// ZombieMMenz internal functions
//////////////////////////////////////////////////////////////


ZombieMMenz::ZombieMMenz( )
	: Km_( 0.005 )
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void ZombieMMenz::vRemesh( const Eref& e, const Qinfo* q )
{
	// cout << "ZombieMMenz::remesh for " << e << endl;
	stoich_->setMMenzKm( e, Km_ );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZombieMMenz::vSetKm( const Eref& e, const Qinfo* q, double v )
{
	Km_ = v;
	stoich_->setMMenzKm( e, v );
}

double ZombieMMenz::vGetKm( const Eref& e, const Qinfo* q ) const
{
	return Km_;
}

void ZombieMMenz::vSetNumKm( const Eref& e, const Qinfo* q, double v )
{
	double volScale = convertConcToNumRateUsingMesh( e, toSub, 1 );
	Km_ = v / volScale;
	setKm( e, q, Km_ );
}

double ZombieMMenz::vGetNumKm( const Eref& e, const Qinfo* q ) const
{
	double numKm = stoich_->getR1( stoich_->convertIdToPoolIndex( e.id() ), 0 );
	
	return numKm;
}

void ZombieMMenz::vSetKcat( const Eref& e, const Qinfo* q, double v )
{
	stoich_->setMMenzKcat( e, v );
}

double ZombieMMenz::vGetKcat( const Eref& e, const Qinfo* q ) const
{
	double kcat = stoich_->getR2( stoich_->convertIdToPoolIndex( e.id() ), 0 );
	return kcat;
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

void ZombieMMenz::setSolver( Id solver, Id orig )
{
	static const DestFinfo* enz = dynamic_cast< const DestFinfo* >(
		EnzBase::initCinfo()->findFinfo( "enzDest" ) );
	static const SrcFinfo* sub = dynamic_cast< const SrcFinfo* >(
		EnzBase::initCinfo()->findFinfo( "toSub" ) );
	static const SrcFinfo* prd = dynamic_cast< const SrcFinfo* >(
		EnzBase::initCinfo()->findFinfo( "toPrd" ) );
	assert( enz );
	assert( sub );
	assert( prd );

	stoich_ = reinterpret_cast< Stoich* >( solver.eref().data() );

	/// Now set up the RateTerm
	vector< Id > subvec;
	vector< Id > prdvec;
	unsigned int rateIndex = stoich_->convertIdToReacIndex( orig );
	unsigned int num = orig.element()->getNeighbours( subvec, enz );
	unsigned int enzIndex = stoich_->convertIdToPoolIndex( subvec[0] );
	MMEnzymeBase* meb;

	double numKm = 1.0; // Dummy default initial values, later to be reset
	double kcat = 1.0;
	/*
	double numKm = base->zGetNumKm( orig.eref(), 0 );
	double kcat = base->zGetKcat( orig.eref(), 0 );
	*/

	num = orig.element()->getNeighbours( subvec, sub );
	if ( num == 1 ) {
		unsigned int subIndex = stoich_->convertIdToPoolIndex( subvec[0] );
		meb = new MMEnzyme1( numKm, kcat, enzIndex, subIndex );
	} else if ( num > 1 ) {
		vector< unsigned int > v;
		for ( unsigned int i = 0; i < num; ++i )
			v.push_back( stoich_->convertIdToPoolIndex( subvec[i] ) );
		ZeroOrder* rateTerm = new NOrder( 1.0, v );
		meb = new MMEnzyme( numKm, kcat, enzIndex, rateTerm );
	} else {
		cout << "Error: ZombieMMenz::zombify: No substrates for "  <<
			orig.path() << endl;
		cout << "Will ignore and continue, but don't be surprised if "
		"simulation fails.\n";
		// assert( 0 );
		return;
	}
	num = orig.element()->getNeighbours( prdvec, prd );
	stoich_->installMMenz( meb, rateIndex, subvec, prdvec );
}

/*
void ZombieMMenz::zombify( Element* solver, Element* orig )
{
	static const DestFinfo* enz = dynamic_cast< const DestFinfo* >(
		MMenz::initCinfo()->findFinfo( "enz" ) );
	static const SrcFinfo* sub = dynamic_cast< const SrcFinfo* >(
		MMenz::initCinfo()->findFinfo( "toSub" ) );
	static const SrcFinfo* prd = dynamic_cast< const SrcFinfo* >(
		MMenz::initCinfo()->findFinfo( "toPrd" ) );
	assert( enz );
	assert( sub );
	assert( prd );

	DataHandler* dh = orig->dataHandler()->copyUsingNewDinfo( 
		ZombieMMenz::initCinfo()->dinfo() );
	MMenz* mmEnz = reinterpret_cast< MMenz* >( 
		orig->dataHandler()->data( 0 ) );

	Eref oer( orig, 0 );
	double Km = mmEnz->getKm( oer, 0 );

	ZombieMMenz* z = reinterpret_cast< ZombieMMenz* >( dh->data( 0 ) );
	z->Km_ = Km;
	z->stoich_ = reinterpret_cast< Stoich* >( 	
		solver->dataHandler()->data( 0 ) );


	/// Now set up the RateTerm
	vector< Id > subvec;
	vector< Id > prdvec;
	unsigned int rateIndex = z->stoich_->convertIdToReacIndex( orig->id() );
	unsigned int num = orig->getNeighbours( subvec, enz );
	unsigned int enzIndex = z->stoich_->convertIdToPoolIndex( subvec[0] );
	MMEnzymeBase* meb;

	num = orig->getNeighbours( subvec, sub );
	if ( num == 1 ) {
		unsigned int subIndex = z->stoich_->convertIdToPoolIndex( subvec[0] );
		meb = new MMEnzyme1( mmEnz->getNumKm( oer, 0 ), mmEnz->getKcat(),
			enzIndex, subIndex );
	} else if ( num > 1 ) {
		vector< unsigned int > v;
		for ( unsigned int i = 0; i < num; ++i )
			v.push_back( z->stoich_->convertIdToPoolIndex( subvec[i] ) );
		ZeroOrder* rateTerm = new NOrder( 1.0, v );
		meb = new MMEnzyme( mmEnz->getNumKm( oer, 0 ), mmEnz->getKcat(),
			enzIndex, rateTerm );
	} else {
		cout << "Error: ZombieMMenz::zombify: No substrates for "  <<
			orig->id().path() << endl;
		cout << "Will ignore and continue, but don't be surprised if "
		"simulation fails.\n";
		// assert( 0 );
		delete dh;
		return;
	}

	num = orig->getNeighbours( prdvec, prd );

	z->stoich_->installMMenz( meb, rateIndex, subvec, prdvec );

	orig->zombieSwap( ZombieMMenz::initCinfo(), dh );
	z->stoich_->setMMenzKm( Eref( orig, 0 ), Km );
}
*/
