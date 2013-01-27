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
	solver_->setMMenzKm( e, Km_ );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZMMenz::vSetKm( const Eref& e, const Qinfo* q, double v )
{
	Km_ = v;
	solver_->setMMenzKm( e, v );
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
	return solver_->getMMenzNumKm( e );
}

void ZMMenz::vSetKcat( const Eref& e, const Qinfo* q, double v )
{
	solver_->setMMenzKcat( e, v );
}

double ZMMenz::vGetKcat( const Eref& e, const Qinfo* q ) const
{
	return solver_->getMMenzKcat( e );
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

void ZMMenz::setSolver( Id solver, Id enzId )
{
	static const DestFinfo* enzFinfo = dynamic_cast< const DestFinfo* >(
		EnzBase::initCinfo()->findFinfo( "enzDest" ) );
	static const SrcFinfo* subFinfo = dynamic_cast< const SrcFinfo* >(
		EnzBase::initCinfo()->findFinfo( "toSub" ) );
	static const SrcFinfo* prdFinfo = dynamic_cast< const SrcFinfo* >(
		EnzBase::initCinfo()->findFinfo( "toPrd" ) );
	assert( enzFinfo );
	assert( subFinfo );
	assert( prdFinfo );

	solver_ = reinterpret_cast< SolverBase* >( solver.eref().data() );

	/// Now set up the RateTerm
	vector< Id > enzvec;
	vector< Id > subvec;
	vector< Id > prdvec;
	unsigned int num = enzId.element()->getNeighbours( enzvec, enzFinfo );
	assert( num == 1 );
	num = enzId.element()->getNeighbours( subvec, subFinfo );
	assert( num > 0 );
	num = enzId.element()->getNeighbours( prdvec, prdFinfo );
	assert( num > 0 );
	solver_->installMMenz( enzId, enzvec[0], subvec, prdvec );
}
