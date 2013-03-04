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
#include "lookupSizeFromMesh.h"
#include "ReacBase.h"
#include "Reac.h"

#define EPSILON 1e-15
const Cinfo* Reac::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions: All inherited from ReacBase
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions: All inherited
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions: All inherited.
		//////////////////////////////////////////////////////////////
	static Cinfo reacCinfo (
		"Reac",
		ReacBase::initCinfo(),
		0,
		0,
		new Dinfo< Reac >()
	);

	return &reacCinfo;
}

static const Cinfo* reacCinfo = Reac::initCinfo();

static const SrcFinfo2< double, double >* toSub = 
 	dynamic_cast< const SrcFinfo2< double, double >* >(
					reacCinfo->findFinfo( "toSub" ) );

static const SrcFinfo2< double, double >* toPrd = 
 	dynamic_cast< const SrcFinfo2< double, double >* >(
					reacCinfo->findFinfo( "toPrd" ) );

//////////////////////////////////////////////////////////////
// Reac internal functions
//////////////////////////////////////////////////////////////


Reac::Reac( )
		: kf_( 0.1 ), kb_( 0.2 ), sub_( 0.0 ), prd_( 0.0 )
{
	;
}

/*
Reac::Reac( double kf, double kb )
	: kf_( kf ), kb_( kb ), concKf_( 0.1 ), concKb_( 0.2 ),
		sub_( 0.0 ), prd_( 0.0 )
{
	;
}
*/

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Reac::vSub( double v )
{
	sub_ *= v;
}

void Reac::vPrd( double v )
{
	prd_ *= v;
}

void Reac::vProcess( const Eref& e, ProcPtr p )
{
	toPrd->send( e, p->threadIndexInGroup, sub_, prd_ );
	toSub->send( e, p->threadIndexInGroup, prd_, sub_ );
	
	sub_ = kf_;
	prd_ = kb_;
}

void Reac::vReinit( const Eref& e, ProcPtr p )
{
	sub_ = kf_ = concKf_ *
		convertConcToNumRateUsingMesh( e, toSub, 0 );
	prd_ = kb_ = concKb_ * 
		convertConcToNumRateUsingMesh( e, toPrd, 0 );
}

void Reac::vRemesh( const Eref& e, const Qinfo* q )
{
	kf_ = concKf_ / convertConcToNumRateUsingMesh( e, toSub, 0 );
	kb_ = concKb_ / convertConcToNumRateUsingMesh( e, toPrd, 0 );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Reac::vSetNumKf( const Eref& e, const Qinfo* q, double v )
{
	sub_ = kf_ = v;
	double volScale = convertConcToNumRateUsingMesh( e, toSub, false );
	concKf_ = kf_ * volScale;
}

double Reac::vGetNumKf( const Eref& e, const Qinfo* q) const
{
	double kf = concKf_ / convertConcToNumRateUsingMesh( e, toSub, false );
	return kf;
}

void Reac::vSetNumKb( const Eref& e, const Qinfo* q, double v )
{
	prd_ = kb_ = v;
	/*
	double volScale = convertConcToNumRateUsingMesh( e, toPrd, true );
	vector< double > vols;
	getReactantVols( e, toSub, vols );
	assert( vols.size() > 0 );
	volScale /= (vols[0] * NA);
	*/

	double volScale = convertConcToNumRateUsingMesh( e, toPrd, false );
	concKb_ = kb_ * volScale;
}

double Reac::vGetNumKb( const Eref& e, const Qinfo* q ) const
{
	double kb = concKb_ / convertConcToNumRateUsingMesh( e, toPrd, 0 );
	return kb;
}

void Reac::vSetConcKf( const Eref& e, const Qinfo* q, double v )
{
	concKf_ = v;
	sub_ = kf_ = v / convertConcToNumRateUsingMesh( e, toSub, 0 );
}

double Reac::vGetConcKf( const Eref& e, const Qinfo* q ) const
{
	return concKf_;
}

void Reac::vSetConcKb( const Eref& e, const Qinfo* q, double v )
{
	concKb_ = v;
	prd_ = kb_ = v / convertConcToNumRateUsingMesh( e, toPrd, 0 );
}

double Reac::vGetConcKb( const Eref& e, const Qinfo* q ) const
{
	return concKb_;
}
