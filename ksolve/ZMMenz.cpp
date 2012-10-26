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
#include "ZMMenz.h"
#include "DataHandlerWrapper.h"

const Cinfo* ZMMenz::initCinfo()
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
		"ZMMenz",
		EnzBase::initCinfo(),
		0,
		0,
		new Dinfo< ZMMenz >()
	);

	return &zombieMMenzCinfo;
}

//////////////////////////////////////////////////////////////

static const Cinfo* zombieMMenzCinfo = ZMMenz::initCinfo();

static const SrcFinfo2< double, double >* toSub = 
    dynamic_cast< const SrcFinfo2< double, double >* >(
	zombieMMenzCinfo->findFinfo( "toSub" ) );

static const SrcFinfo2< double, double >* toPrd = 
	dynamic_cast< const SrcFinfo2< double, double >* >(
	zombieMMenzCinfo->findFinfo( "toPrd" ) );

//////////////////////////////////////////////////////////////
// ZMMenz internal functions
//////////////////////////////////////////////////////////////


ZMMenz::ZMMenz( )
	: Km_( 0.005 )
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void ZMMenz::vRemesh( const Eref& e, const Qinfo* q )
{
	// cout << "ZMMenz::remesh for " << e << endl;
	stoich_->setMMenzKm( e, Km_ );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZMMenz::vSetKm( const Eref& e, const Qinfo* q, double v )
{
	Km_ = v;
	stoich_->setMMenzKm( e, v );
}

double ZMMenz::vGetKm( const Eref& e, const Qinfo* q ) const
{
	return Km_;
}

void ZMMenz::vSetNumKm( const Eref& e, const Qinfo* q, double v )
{
	double volScale = convertConcToNumRateUsingMesh( e, toSub, 1 );
	Km_ = v / volScale;
	setKm( e, q, Km_ );
}

double ZMMenz::vGetNumKm( const Eref& e, const Qinfo* q ) const
{
	double numKm = stoich_->getR1( stoich_->convertIdToPoolIndex( e.id() ), 0 );
	
	return numKm;
}

void ZMMenz::vSetKcat( const Eref& e, const Qinfo* q, double v )
{
	stoich_->setMMenzKcat( e, v );
}

double ZMMenz::vGetKcat( const Eref& e, const Qinfo* q ) const
{
	double kcat = stoich_->getR2( stoich_->convertIdToPoolIndex( e.id() ), 0 );
	return kcat;
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

void ZMMenz::setSolver( Id solver, Id orig )
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

	stoich_ = reinterpret_cast< StoichCore* >( solver.eref().data() );

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
		cout << "Error: ZMMenz::zombify: No substrates for "  <<
			orig.path() << endl;
		cout << "Will ignore and continue, but don't be surprised if "
		"simulation fails.\n";
		// assert( 0 );
		return;
	}
	num = orig.element()->getNeighbours( prdvec, prd );
	stoich_->installMMenz( meb, rateIndex, subvec, prdvec );
}
