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
#include "ZEnz.h"
#include "Enz.h"
#include "DataHandlerWrapper.h"

const Cinfo* ZEnz::initCinfo()
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
		"ZEnz",
		CplxEnzBase::initCinfo(),
		0,
		0,
		new Dinfo< ZEnz >()
	);

	return &zombieEnzCinfo;
}
//////////////////////////////////////////////////////////////

static const Cinfo* zombieEnzCinfo = ZEnz::initCinfo();

static const SrcFinfo2< double, double >* toSub =
	dynamic_cast< const SrcFinfo2< double, double >* >(
	zombieEnzCinfo->findFinfo( "toSub" ) );

//////////////////////////////////////////////////////////////
// ZEnz internal functions
//////////////////////////////////////////////////////////////

ZEnz::ZEnz( )
		: 
				solver_( 0 ),
				concK1_( 1.0 )
{ ; }

ZEnz::~ZEnz( )
{ ; }

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void ZEnz::vRemesh( const Eref& e, const Qinfo* q )
{   
	solver_->setEnzK1( e, concK1_ );
}   


//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

// v is in number units.
void ZEnz::vSetK1( const Eref& e, const Qinfo* q, double v )
{
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub, 1 );

	concK1_ = v / volScale;
	solver_->setEnzK1( e, concK1_ );
}

// v is In number units.
double ZEnz::vGetK1( const Eref& e, const Qinfo* q ) const
{
	return solver_->getEnzNumK1( e );
}

void ZEnz::vSetK2( const Eref& e, const Qinfo* q, double v )
{
	solver_->setEnzK2( e, v );
}

double ZEnz::vGetK2( const Eref& e, const Qinfo* q ) const
{
	return solver_->getEnzK2( e );
}

void ZEnz::vSetKcat( const Eref& e, const Qinfo* q, double v )
{
	solver_->setEnzK3( e, v );
}

double ZEnz::vGetKcat( const Eref& e, const Qinfo* q ) const
{
	return solver_->getEnzK3( e );
}


void ZEnz::vSetKm( const Eref& e, const Qinfo* q, double v )
{
	double k2 = getK2( e, q );
	double k3 = getKcat( e, q );
	concK1_ = ( k2 + k3 ) / v;
	solver_->setEnzK1( e, concK1_ );
}

double ZEnz::vGetKm( const Eref& e, const Qinfo* q ) const
{
	double k2 = getK2( e, q );
	double k3 = getKcat( e, q );

	return ( k2 + k3 ) / concK1_;
}

void ZEnz::vSetNumKm( const Eref& e, const Qinfo* q, double v )
{
	double k2 = getK2( e, q );
	double k3 = getKcat( e, q );
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub, 1 );
	concK1_ = volScale * ( k2 + k3 ) / v;

	solver_->setEnzK1( e, concK1_ );
}

double ZEnz::vGetNumKm( const Eref& e, const Qinfo* q ) const
{
	double k2 = vGetK2( e, q );
	double k3 = vGetKcat( e, q );
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub, 1 );

	return ( k2 + k3 ) / ( concK1_ * volScale );
}

void ZEnz::vSetRatio( const Eref& e, const Qinfo* q, double v )
{
	double Km = getKm( e, q );
	double k2 = getK2( e, q );
	double k3 = getKcat( e, q );

	k2 = v * k3;

	solver_->setEnzK2( e, k2 );
	double k1 = ( k2 + k3 ) / Km;

	setConcK1( e, q, k1 );
}

double ZEnz::vGetRatio( const Eref& e, const Qinfo* q ) const
{
	double k2 = getK2( e, q );
	double k3 = getKcat( e, q );
	return k2 / k3;
}

void ZEnz::vSetConcK1( const Eref& e, const Qinfo* q, double v )
{
	concK1_ = v;
	solver_->setEnzK1( e, v );
}

double ZEnz::vGetConcK1( const Eref& e, const Qinfo* q ) const
{
	return concK1_;
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

// static func
void ZEnz::setSolver( Id solver, Id enz )
{
	static const SrcFinfo* subFinfo = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toSub" ) );
	static const SrcFinfo* prdFinfo = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toPrd" ) );
	static const SrcFinfo* enzFinfo = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toEnz" ) );
	static const SrcFinfo* cplxFinfo = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toCplx" ) );

	assert( subFinfo );
	assert( prdFinfo );
	assert( enzFinfo );
	assert( cplxFinfo );
	vector< Id > temp;
	unsigned int numReactants;
	numReactants = enz.element()->getNeighbours( temp, enzFinfo ); 
	assert( numReactants == 1 );
	Id enzMol = temp[0];
	vector< Id > subs;
	numReactants = enz.element()->getNeighbours( subs, subFinfo ); 
	assert( numReactants > 0 );
	numReactants = enz.element()->getNeighbours( temp, cplxFinfo ); 
	assert( numReactants == 1 );
	Id cplx = temp[0];
	vector< Id > prds;
	numReactants = enz.element()->getNeighbours( prds, prdFinfo ); 
	assert( numReactants > 0 );

	solver_ = reinterpret_cast< SolverBase* >( solver.eref().data() );
	solver_->installEnzyme( enz, enzMol, cplx, subs, prds );
}
