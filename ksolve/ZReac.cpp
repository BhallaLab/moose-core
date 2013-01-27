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
#include "ZReac.h"
// #include "Reac.h"
// #include "ElementValueFinfo.h"
// #include "DataHandlerWrapper.h"

const Cinfo* ZReac::initCinfo()
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
		"ZReac",
		ReacBase::initCinfo(),
		0,
		0,
		new Dinfo< ZReac >()
	);

	return &zombieReacCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieReacCinfo = ZReac::initCinfo();

static const SrcFinfo2< double, double >* toSub = 
 	dynamic_cast< const SrcFinfo2< double, double >* >(
					zombieReacCinfo->findFinfo( "toSub" ) );

static const SrcFinfo2< double, double >* toPrd = 
 	dynamic_cast< const SrcFinfo2< double, double >* >(
					zombieReacCinfo->findFinfo( "toPrd" ) );

ZReac::ZReac()
		: solver_( 0 )
{;}

ZReac::~ZReac()
{;}


//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void ZReac::vRemesh( const Eref& e, const Qinfo* q )
{
	solver_->setReacKf( e, concKf_ );
	solver_->setReacKb( e, concKb_ );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

// This conversion is deprecated, used mostly for kkit conversions.
void ZReac::vSetNumKf( const Eref& e, const Qinfo* q, double v )
{
	double volScale = convertConcToNumRateUsingMesh( e, toSub, 0 );
	concKf_ = v * volScale;
	solver_->setReacKf( e, concKf_ );
}

double ZReac::vGetNumKf( const Eref& e, const Qinfo* q ) const
{
	// Return value for voxel 0. Conceivably I might want to use the
	// DataId part to specify which voxel to use, but that isn't in the
	// current definition for Reacs as being a single entity for the entire
	// compartment.
	return solver_->getReacNumKf( e );
}

// Deprecated, used for kkit conversion backward compatibility
void ZReac::vSetNumKb( const Eref& e, const Qinfo* q, double v )
{
	double volScale = convertConcToNumRateUsingMesh( e, toPrd, 0 );
	concKb_ = v * volScale;
	solver_->setReacKb( e, concKb_ );
}

double ZReac::vGetNumKb( const Eref& e, const Qinfo* q ) const
{
	return solver_->getReacNumKb( e );
}

void ZReac::vSetConcKf( const Eref& e, const Qinfo* q, double v )
{
	concKf_ = v;
	solver_->setReacKf( e, v );
}

double ZReac::vGetConcKf( const Eref& e, const Qinfo* q ) const
{
	return concKf_;
}

void ZReac::vSetConcKb( const Eref& e, const Qinfo* q, double v )
{
	concKb_ = v;
	solver_->setReacKb( e, v );
}

double ZReac::vGetConcKb( const Eref& e, const Qinfo* q ) const
{
	return concKb_;
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

// Virtual func called in zombify before fields are assigned.
void ZReac::setSolver( Id solver, Id orig )
{
	assert( solver != Id() );

	solver_ = reinterpret_cast< SolverBase* >( solver.eref().data( ) );
	vector< Id > sub;
	vector< Id > prd;
	orig.element()->getNeighbours( sub, toSub );
	orig.element()->getNeighbours( prd, toPrd );

	solver_->installReaction( orig, sub, prd );
}
