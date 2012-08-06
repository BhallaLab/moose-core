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
#include "EnzBase.h"
#include "MMenz.h"

#define EPSILON 1e-15

const Cinfo* MMenz::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions: Inherited from base class.
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions: Inherited from base class.
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions: inherited
		//////////////////////////////////////////////////////////////

	static Cinfo mmEnzCinfo (
		"MMenz",
		EnzBase::initCinfo(),
		0,
		0,
		new Dinfo< MMenz >()
	);

	return &mmEnzCinfo;
}
//////////////////////////////////////////////////////////////

static const Cinfo* mmEnzCinfo = MMenz::initCinfo();

static const SrcFinfo2< double, double >* toSub = 
    dynamic_cast< const SrcFinfo2< double, double >* >(
	mmEnzCinfo->findFinfo( "toSub" ) );

static const SrcFinfo2< double, double >* toPrd = 
	dynamic_cast< const SrcFinfo2< double, double >* >(
	mmEnzCinfo->findFinfo( "toPrd" ) );


//////////////////////////////////////////////////////////////
// MMenz internal functions
//////////////////////////////////////////////////////////////

MMenz::MMenz( )
	: Km_( 0.005 ), kcat_( 0.1 ), sub_( 0.0 ), enz_( 0.0 )
{
	;
}

MMenz::~MMenz()
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////


void MMenz::vSub( double n )
{
	sub_ *= n;
}

void MMenz::vEnz( double n )
{
	enz_ = n;
}

void MMenz::vProcess( const Eref& e, ProcPtr p )
{
	double rate = kcat_ * enz_ * sub_ / ( numKm_ + sub_ );
	toSub->send( e, p->threadIndexInGroup, 0, rate );
	toPrd->send( e, p->threadIndexInGroup, rate, 0 );
	
	sub_ = 1.0;
}

void MMenz::vReinit( const Eref& e, ProcPtr p )
{
	sub_ = 1.0;
	enz_ = 0.0;
}

void MMenz::vRemesh( const Eref& e, const Qinfo* q )
{
//	cout << "MMenz::remesh for " << e << endl;
	double volScale = convertConcToNumRateUsingMesh( e, toSub, 1 );
	numKm_ = Km_ * volScale;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void MMenz::vSetKm( const Eref& enz, const Qinfo* q, double v )
{
	Km_ = v;
	double volScale = convertConcToNumRateUsingMesh( enz, toSub, 1 );
	numKm_ = v * volScale;
}

double MMenz::vGetKm( const Eref& enz, const Qinfo* q ) const
{
	return Km_;
}

void MMenz::vSetNumKm( const Eref& enz, const Qinfo* q, double v )
{
	double volScale = convertConcToNumRateUsingMesh( enz, toSub, 1 );
	numKm_ = v;
	Km_ = v / volScale;
}

double MMenz::vGetNumKm( const Eref& enz, const Qinfo* q ) const
{
	double volScale = convertConcToNumRateUsingMesh( enz, toSub, 1 );
	return Km_ * volScale;
}


void MMenz::vSetKcat( const Eref& e, const Qinfo* q, double v )
{
	kcat_ = v;
}

double MMenz::vGetKcat( const Eref& e, const Qinfo* q ) const
{
	return kcat_;
}
